"""Purpose of this program: the urgent signs check (an input guardrail).

Plain Python rules, NOT the AI model. After every intake answer the
coordinator and the app call check_urgent(). If any urgent sign is found, the
intake stops and the app shows a big "Get emergency care now" banner with
simple first-aid steps. Nothing the model writes can switch this off: the
check reads only the caregiver's answers and the vitals, and the coordinator
ends the flow before any AI step when it fires.

How a sign is found:
  1. Words: each rule has phrases ("hard to wake", "convulsion"...). The
     answers are split into short clauses (at , ; . and "but"). A phrase
     counts unless that clause says "no", "not", "denies" or "without" before
     it, so "no chest pain" does not trigger but "chest pain since morning"
     does.
  2. Numbers: some vitals are urgent on their own (very fast breathing for
     the age, very low blood pressure).

The rules lean towards safety: when in doubt they fire. A false alarm costs a
trip to a clinic; a missed sign can cost a life.

Try it:  python -m triage.urgent_signs
"""

import re
from dataclasses import dataclass, field

from agents.intake_models import IntakeRecord

BANNER_TITLE = "Get emergency care now"
BANNER_TEXT = ("This patient has a danger sign. Take them to the nearest clinic or hospital, "
               "or call for emergency help, right away. Do not wait for this app.")


@dataclass
class UrgentSign:
    name: str  # plain-language name of the danger sign
    reason: str  # what in the intake triggered it (shown to the caregiver)
    first_aid: list[str]  # simple steps while getting help


@dataclass
class UrgentResult:
    urgent: bool
    signs: list[UrgentSign] = field(default_factory=list)

    @property
    def first_aid(self) -> list[str]:
        """All first-aid steps for the signs found, without repeats."""
        steps = []
        for sign in self.signs:
            steps += [s for s in sign.first_aid if s not in steps]
        return steps


# --- First-aid steps: simple, safe actions only. No medicines, no doses. ---
CALL_HELP = "Send someone to get emergency help or transport now."
STAY = "Stay with the patient and watch their breathing."
NOTHING_BY_MOUTH = "Do not give food or drink if they are not fully awake."

FIRST_AID = {
    "breathing": [CALL_HELP, "Help them sit up, leaning slightly forward.",
                  "Loosen tight clothing. Keep them calm and still.", STAY],
    "unconscious": [CALL_HELP, "Lay them on their side (recovery position) so they don't choke.",
                    NOTHING_BY_MOUTH, STAY],
    "convulsions": [CALL_HELP, "Move hard or sharp things away. Do not hold them down.",
                    "Do not put anything in their mouth.",
                    "When the shaking stops, lay them on their side.",
                    "Note how long the fit lasted."],
    "chest_pain": [CALL_HELP, "Keep them sitting or lying still. Do not let them walk.",
                   "Loosen tight clothing.", STAY],
    "bleeding": [CALL_HELP, "Press hard on the bleeding with a clean cloth and keep pressing.",
                 "If the cloth soaks through, add more on top. Do not lift it off.",
                 "Lay them down and keep them warm."],
    "cannot_drink": [CALL_HELP, "If they can swallow, give small sips of clean water often.",
                     "Keep them lying on their side if they are vomiting."],
    "shock": [CALL_HELP, "Lay them flat and raise their legs a little, unless this hurts.",
              "Keep them warm with a blanket.", NOTHING_BY_MOUTH],
    "pregnancy_bleeding": [CALL_HELP, "Help her lie down on her left side.",
                           "Keep her warm. Do not put anything inside the vagina.",
                           "Bring any pregnancy card or clinic papers."],
}

# --- Word rules: (key, plain name, phrases) ---
# Phrases are regular expressions; \b keeps "fits" from matching "benefits".
WORD_RULES = [
    ("breathing", "Severe trouble breathing", [
        r"can'?t breathe", r"cannot breathe", r"unable to breathe", r"struggl\w* to breathe",
        r"gasping", r"blue lips", r"lips (are |turning )?blue",
        r"(severe|very bad|serious) (shortness of breath|trouble breathing|difficulty breathing)",
        r"(fighting|struggling) for breath", r"chest (is )?(pulling|sucking) in"]),
    ("unconscious", "Unconscious or very hard to wake", [
        r"unconscious", r"unresponsive", r"not responding", r"won'?t wake", r"can'?t wake",
        r"cannot wake", r"hard to wake", r"difficult to wake", r"passed out", r"fainted",
        r"collapsed", r"very (drowsy|sleepy)"]),
    ("convulsions", "Convulsions (fits)", [
        r"convuls\w*", r"seizures?", r"\bfits?\b", r"\bfitting\b",
        r"shaking (all over|uncontrollably)"]),
    ("bleeding", "Heavy bleeding", [
        r"heavy bleeding", r"bleeding (heavily|a lot|badly|won'?t stop|that won'?t stop)",
        r"won'?t stop bleeding", r"lots of blood", r"losing (a lot of )?blood",
        r"vomiting blood", r"soak\w* (through )?(pads?|cloths?)"]),
    ("cannot_drink", "Cannot drink or keep anything down", [
        r"can'?t (drink|keep anything down|keep (food|water|fluids|anything) down)",
        r"cannot (drink|keep anything down)", r"unable to (drink|keep anything down)",
        r"not able to drink", r"vomit\w* (everything|all)", r"throws? up everything",
        r"not drinking"]),
    ("shock", "Signs of shock", [
        r"cold,? clammy", r"clammy", r"cold (hands|feet|skin) and (weak|fast)",
        r"(weak|faint),? fast pulse", r"fast,? weak pulse", r"very pale and (cold|weak)"]),
]
NEGATION_RE = re.compile(r"\b(no|not|denies|denied|without|never|none)\b", re.IGNORECASE)
CHEST_PAIN_RE = re.compile(r"chest (pain|tightness|pressure)|pain in (the |his |her )?chest",
                           re.IGNORECASE)
SWEATING_RE = re.compile(r"sweat\w*|clammy", re.IGNORECASE)
BLEEDING_RE = re.compile(r"bleed\w*|blood", re.IGNORECASE)


def _clauses(text: str) -> list[str]:
    """Split text into short clauses, so "no fever, but chest pain" becomes
    ["no fever", "chest pain"] and the "no" doesn't cancel the chest pain."""
    return [c.strip() for c in re.split(r"[,;.\n]|\bbut\b", text, flags=re.IGNORECASE)
            if c.strip()]


def _find(pattern: str | re.Pattern, text: str) -> str | None:
    """Return the first clause where the pattern appears and is not negated."""
    regex = re.compile(pattern, re.IGNORECASE) if isinstance(pattern, str) else pattern
    for clause in _clauses(text):
        match = regex.search(clause)
        if match and not NEGATION_RE.search(clause[:match.start()]):
            return clause
    return None


def _intake_text(record: IntakeRecord, answers: dict[str, str] | None) -> str:
    """Everything the caregiver typed, as one block of text to scan."""
    parts = [record.chief_complaint or "", record.caregiver_notes or ""]
    parts += [f"{s.name} {s.duration or ''}" for s in record.symptoms]
    parts += list((answers or {}).values())  # the masked answers, word for word
    return "\n".join(p for p in parts if p)


def _fast_breathing(age: int | None, rate: int) -> bool:
    """Breathing much too fast or too slow for the age (WHO-style cut-offs)."""
    if rate < 8:
        return True
    if age is None or age >= 12:
        return rate >= 30
    if age >= 5:
        return rate >= 40
    if age >= 1:
        return rate >= 50
    return rate >= 60  # babies under 1 year


def check_urgent(record: IntakeRecord, answers: dict[str, str] | None = None) -> UrgentResult:
    """Look for urgent signs in the intake. answers is the dict of masked
    answer text from the IntakeSession (field -> text)."""
    text = _intake_text(record, answers)
    signs: list[UrgentSign] = []

    def add(key: str, name: str, reason: str) -> None:
        if all(s.name != name for s in signs):  # one entry per sign
            signs.append(UrgentSign(name, reason, FIRST_AID[key]))

    # 1. Word rules
    for key, name, phrases in WORD_RULES:
        for phrase in phrases:
            if clause := _find(phrase, text):
                add(key, name, f'Intake says: "{clause}"')
                break

    # 2. Chest pain WITH sweating (both must be there, anywhere in the intake)
    pain = _find(CHEST_PAIN_RE, text)
    sweat = _find(SWEATING_RE, text)
    if pain and sweat:
        add("chest_pain", "Chest pain with sweating", f'Intake says: "{pain}" and "{sweat}"')

    # 3. Pregnant AND any bleeding: always urgent, even if not called "heavy"
    if record.pregnant and (clause := _find(BLEEDING_RE, text)):
        add("pregnancy_bleeding", "Pregnant and bleeding", f'Patient may be pregnant; intake '
            f'says: "{clause}"')

    # 4. Vitals
    v = record.vitals
    if v.breathing_rate is not None and _fast_breathing(record.age, v.breathing_rate):
        add("breathing", "Severe trouble breathing",
            f"Breathing rate {v.breathing_rate} per minute is outside the safe range for the age")
    if v.bp_systolic is not None and v.bp_systolic < 90 and (record.age is None or record.age >= 12):
        add("shock", "Signs of shock", f"Blood pressure top number {v.bp_systolic} is very low")
    if v.pulse is not None and (v.pulse >= 140 or v.pulse < 40) and (record.age is None or record.age >= 12):
        add("shock", "Signs of shock", f"Pulse {v.pulse} is outside the safe range")

    return UrgentResult(urgent=bool(signs), signs=signs)


if __name__ == "__main__":
    from agents.intake_models import Symptom, Vitals
    tests = {
        "no chest pain, mild cough": IntakeRecord(patient_id="T1", age=40,
                                                  chief_complaint="no chest pain, mild cough"),
        "chest pain and sweating": IntakeRecord(patient_id="T2", age=55,
                                                chief_complaint="chest pain since morning",
                                                symptoms=[Symptom(name="sweating", duration="1 hour")]),
        "pregnant, spotting blood": IntakeRecord(patient_id="T3", age=24, pregnant=True,
                                                 chief_complaint="some blood when I go to toilet"),
        "child breathing 55": IntakeRecord(patient_id="T4", age=3, chief_complaint="cough",
                                           vitals=Vitals(breathing_rate=55)),
        "had a seizure": IntakeRecord(patient_id="T5", age=6, chief_complaint="fever and had a seizure"),
    }
    for label, rec in tests.items():
        r = check_urgent(rec)
        print(f"{label:<28} urgent={r.urgent}  {[s.name for s in r.signs]}")
