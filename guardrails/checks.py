"""Purpose of this program: rule-based guardrail checks on questions and answers.

Input checks run before the coordinator sees anything:
  - PII redaction: masks SSNs, phone numbers, emails and dates of birth
  - Prompt-injection detection: blocks "ignore previous instructions" etc.
Output checks run after the specialist answers:
  - Medical-advice blocking: replaces advice with a safe message
  - Missing-sources flag: warns when a real answer cites nothing

Every check returns a GuardrailResult: allowed or blocked, the reason, the
cleaned text, and a list of flags describing what was found.
"""

import re  # regular expressions: patterns for finding text
from dataclasses import dataclass, field


@dataclass
class GuardrailResult:
    allowed: bool  # False means the text was blocked
    reason: str  # short explanation, "ok" when nothing was found
    text: str  # the cleaned (or replacement) text to use from here on
    flags: list[str] = field(default_factory=list)  # what was found, if anything


# --- PII redaction ---
# Each entry is (label, pattern). Order matters: SSN runs before phone so a
# number like 123-45-6789 is labelled as an SSN, not part of a phone number.
PII_PATTERNS = [
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    # Phone: optional +1, optional brackets round the area code, then
    # 3-3-4 digits split by dashes, dots or spaces, e.g. (312) 555-0100.
    # (?<!\d) and (?!\d) stop it matching part of a longer number.
    ("PHONE", re.compile(r"(?<!\d)(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}(?!\d)")),
    # Dates such as 04/12/1985, 4-12-85 or 1985-04-12. Every date is masked,
    # because a rule can't tell a birth date from any other date. Safer to
    # over-mask than to leak a date of birth.
    ("DOB", re.compile(r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})\b")),
]


def redact_pii(text: str) -> tuple[str, list[str]]:
    """Replace each piece of PII with a label like [SSN]. Returns the cleaned
    text and the labels that were found."""
    found = []
    for label, pattern in PII_PATTERNS:
        text, count = pattern.subn(f"[{label}]", text)  # subn also counts matches
        if count:
            found.append(label)
    return text, found


# --- Prompt-injection detection ---
# Phrases people use to try to override the assistant's rules.
# re.IGNORECASE (below) means capital letters don't matter.
INJECTION_PATTERNS = [
    r"\b(ignore|disregard|forget)\b.{0,20}\b(previous|prior|above|earlier|your|all)\b.{0,20}\b(instructions|rules|prompts?)\b",
    r"\b(reveal|show|print|repeat|display|tell me)\b.{0,20}\b(system prompt|your (instructions|rules|prompt))\b",
    r"\byou are now\b",
    # Not plain "act as": "can my spouse act as my representative?" is fine.
    r"\bpretend (to be|you are|you're)\b",
    r"\b(developer mode|jailbreak)\b",
]
INJECTION_RE = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE)


def check_input(text: str) -> GuardrailResult:
    """Run the input checks: redact PII first, then look for injection."""
    cleaned, pii_found = redact_pii(text)
    flags = [f"redacted {label}" for label in pii_found]

    if INJECTION_RE.search(cleaned):
        return GuardrailResult(
            allowed=False,
            reason="Possible prompt injection: the message tries to change or reveal the assistant's instructions.",
            text=cleaned,
            flags=flags + ["prompt injection"],
        )

    reason = "PII redacted" if pii_found else "ok"
    return GuardrailResult(allowed=True, reason=reason, text=cleaned, flags=flags)


# --- Output checks ---
# Phrases that sound like telling someone what treatment or dose to use.
MEDICAL_ADVICE_PATTERNS = [
    r"\byou should (take|stop taking|start taking|switch to)\b",
    r"\bI (would )?(recommend|suggest) (taking|you take|this (treatment|medication|drug))\b",
    r"\b(increase|decrease|lower|raise|double) your (dose|dosage)\b",
    r"\b\d+\s?(mg|milligrams)\b",  # a specific dose
]
MEDICAL_ADVICE_RE = re.compile("|".join(MEDICAL_ADVICE_PATTERNS), re.IGNORECASE)

SAFE_MEDICAL_MESSAGE = (
    "I can't give medical advice. Please talk to your doctor or pharmacist "
    "about which treatment is right for you. I can help with what your plan "
    "covers and what it costs."
)

# Phrases that mean the assistant declined, couldn't find an answer, or is
# asking the member for their id. Those answers don't need sources.
REFUSAL_RE = re.compile(
    r"I can only help|I can't give medical advice|I don't have that|"
    r"couldn't find|could not find|unable to find|not able to help|"
    r"(need|provide) (your|a) member id",
    re.IGNORECASE,
)


def check_output(answer: str, sources: list[dict]) -> GuardrailResult:
    """Run the output checks on a specialist's answer."""
    if MEDICAL_ADVICE_RE.search(answer):
        return GuardrailResult(
            allowed=False,
            reason="Answer looked like medical advice, so it was replaced.",
            text=SAFE_MEDICAL_MESSAGE,
            flags=["medical advice"],
        )

    # A real answer (not a refusal) with nothing behind it might be made up.
    # It is still shown, but flagged so a person can review it.
    if not sources and not REFUSAL_RE.search(answer):
        return GuardrailResult(
            allowed=True,
            reason="Answer has no sources; it may not be backed by the plan documents.",
            text=answer,
            flags=["no sources"],
        )

    return GuardrailResult(allowed=True, reason="ok", text=answer)
