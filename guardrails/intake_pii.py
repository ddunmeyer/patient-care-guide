"""Purpose of this program: mask personal details in intake answers before
anything is sent to OpenAI or Langfuse.

Builds on redact_pii() in guardrails/checks.py (SSNs, emails, US phone
numbers, dates) and adds what an intake needs:
  - NAME: any name the caregiver typed into the app (the patient's or their
    own), plus names after phrases like "my name is" or "Mr." / "Mrs."
  - PHONE: international numbers such as +254 712 345 678
  - ADDRESS: street addresses ("12 Market Road"), P.O. boxes, house/plot numbers

The model only ever sees the patient_id, never the patient's name.
"""

import re

from guardrails.checks import redact_pii

# "my name is Amina", "her name's Joseph", "called Grace", "Mrs. Otieno".
# The name part must start with a capital letter, so plain words are left alone.
NAME_PHRASE_RE = re.compile(
    r"\b(?:(?:my|his|her|their) name(?: is|'s)|(?:I am|I'm) called|called|named)\s+"
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)"
    r"|\b(?:Mr|Mrs|Ms|Miss|Dr|Mama|Baba)\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)")

# A run of 9 or more digits with optional + and separators: phone numbers in
# any country. Blood pressure ("120/80") and temperatures are too short to match.
INTL_PHONE_RE = re.compile(r"\+?\d(?:[\s().-]?\d){8,}")

STREET_WORDS = (r"street|st|road|rd|avenue|ave|lane|ln|drive|close|court|way|crescent|"
                r"boulevard|blvd|highway|estate")
ADDRESS_RE = re.compile(
    rf"\b\d{{1,5}}[A-Za-z]?\s+(?:[A-Z][a-z]+\s+){{1,3}}(?:{STREET_WORDS})\b\.?"  # 12 Market Road
    r"|\bP\.?\s?O\.?\s+Box\s+\d+"  # P.O. Box 123
    r"|\b(?:house|plot|plot no\.?|house no\.?)\s+\d+\w*",  # house 45, plot 7B
    re.IGNORECASE)


def mask_intake_text(text: str, known_names: list[str] | None = None) -> tuple[str, list[str]]:
    """Mask personal details. Returns (masked text, labels found).
    known_names: names typed in the app (e.g. the patient's name), always masked."""
    text, found = redact_pii(text)  # SSN, EMAIL, PHONE (US), DOB

    for name in known_names or []:
        # Mask the full name and each part of it ("Amina Hassan", "Amina", "Hassan").
        for part in sorted({name.strip(), *name.split()}, key=len, reverse=True):
            if len(part) >= 2:
                text, n = re.subn(rf"\b{re.escape(part)}\b", "[NAME]", text, flags=re.IGNORECASE)
                if n and "NAME" not in found:
                    found.append("NAME")

    def mask_name(match: re.Match) -> str:
        name = match.group(1) or match.group(2)
        return match.group(0).replace(name, "[NAME]")

    text, n = NAME_PHRASE_RE.subn(mask_name, text)
    if n and "NAME" not in found:
        found.append("NAME")
    for label, pattern in (("PHONE", INTL_PHONE_RE), ("ADDRESS", ADDRESS_RE)):
        text, n = pattern.subn(f"[{label}]", text)
        if n and label not in found:
            found.append(label)
    return text, found
