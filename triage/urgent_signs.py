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
    "stroke": [CALL_HELP, "Note the time the signs started. Doctors need to know.",
               "Lay them on their side if they are drowsy or vomiting.", NOTHING_BY_MOUTH, STAY],
    "allergy": [CALL_HELP, "Help them sit up if breathing is hard, or lie flat if they feel faint.",
                "If they have their own allergy pen, help them use it.", STAY],
    "self_harm": [CALL_HELP, "Do not leave them alone.",
                  "Keep pills, poisons and sharp objects out of reach.",
                  "If they swallowed something, bring the bottle or packet with you."],
    "fever_baby": [CALL_HELP, "Keep the baby lightly dressed, not wrapped up tightly.",
                   "Keep breastfeeding or giving feeds if the baby will take them."],
    "high_fever": [CALL_HELP, "Remove extra clothing and keep them cool.",
                   "Give small sips of water if they are fully awake.", STAY],
}

# --- Word rules: (key, plain name, phrases) ---
# Phrases are regular expressions; \b keeps "fits" from matching "benefits".
# Widened after testing: any trouble breathing (not only "severe"), chest pain on
# its own in adults, stroke, allergic swelling, self-harm and fever in babies,
# plus common misspellings ("cant breath", "siezure").
WORD_RULES = [
    ("breathing", "Trouble breathing", [
        r"can'?t breathe?\b", r"cannot breathe?\b", r"unable to breathe?\b", r"struggl\w* to breathe?\b",
        r"gasping", r"blue lips", r"lips (are |turning )?blue",
        r"short(ness)? of breath", r"\bbreathless", r"out of breath",
        r"(trouble|difficulty|problems?|hard|difficult) (with )?breath\w*", r"(hard|difficult) to breathe?\b",
        r"can'?t (catch|get) (his |her |my |their )?breath",
        r"(fighting|struggling) for breath", r"chest (is )?(pulling|sucking) in",
        r"breathing (very )?(fast|hard|badly|noisy)"]),
    ("unconscious", "Unconscious or very hard to wake", [
        r"unconscious", r"unresponsive", r"not responding", r"won'?t wake", r"can'?t wake",
        r"cannot wake", r"hard to wake", r"difficult to wake", r"passed out", r"fainted",
        r"blacked out", r"\bout cold\b", r"collapsed", r"very (drowsy|sleepy)"]),
    ("convulsions", "Convulsions (fits)", [
        r"convuls\w*", r"s(ei|ie|i)[sz]ures?", r"(?<!coughing )\bfits?\b(?! of cough)(?! and well)",
        r"\bfitting\b", r"shaking (all over|uncontrollably)"]),
    ("stroke", "Signs of a stroke", [
        r"face (is )?(drooping|droops|dropped|fallen)", r"droop\w* (face|mouth|on one side)",
        r"slurr\w* (speech|words|talking)", r"speech (is )?slurred", r"can'?t (speak|talk) properly",
        r"(weak|numb|can'?t move|can'?t feel)\w* (on |in )?(one|his|her|their|the) (left |right )?(side|arm|leg)",
        r"one side (of (the|his|her|their) body )?(is )?(weak|numb)",
        r"sudden (weakness|numbness|confusion)", r"worst headache"]),
    ("bleeding", "Heavy bleeding", [
        r"heavy bleeding", r"bleeding (heavily|a lot|badly|won'?t stop|that won'?t stop|nonstop)",
        r"won'?t stop bleeding", r"lots of blood", r"losing (a lot of )?blood",
        r"blood (is )?(pouring|gushing|spurting)", r"vomit\w* (up )?blood", r"cough\w* (up )?blood",
        r"soak\w* (through )?(pads?|cloths?)"]),
    ("allergy", "Severe allergic reaction", [
        r"(lips?|tongue|throat|face) (is |are )?(swelling|swollen|swelled)",
        r"swell\w* (of |in )?(the |his |her |their )?(lips?|tongue|throat|face)",
        r"throat (is )?(closing|tight)", r"anaphyla\w*"]),
    ("self_harm", "Self-harm or poisoning", [
        r"want\w* to die", r"kill (myself|himself|herself|themselves)", r"suicid\w*",
        r"overdos\w*", r"too many (pills|tablets)", r"(?<!food )poison\w*",
        r"(drank|swallowed) (bleach|paraffin|kerosene|pesticide|rat poison)"]),
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
# "Never had pain like this before" is not a negation: it makes the sign worse.
NOT_A_NEGATION_RE = re.compile(r"\bnever\b.*\b(like (this|that)|this bad|so bad|before)\b",
                               re.IGNORECASE)
CHEST_PAIN_RE = re.compile(r"chest (pain|tightness|pressure|hurts?)|pain in (the |his |her |their )?chest"
                           r"|heart attack", re.IGNORECASE)
SWEATING_RE = re.compile(r"sweat\w*|clammy", re.IGNORECASE)
BLEEDING_RE = re.compile(r"bleed\w*|blood", re.IGNORECASE)
FEVER_RE = re.compile(r"fever\w*|feverish|\bhot\b|high temp\w*", re.IGNORECASE)


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
        if not match:
            continue
        negated = NEGATION_RE.search(clause[:match.start()]) and not NOT_A_NEGATION_RE.search(clause)
        if not negated:
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

    adult = record.age is None or record.age >= 12

    # 2. Chest pain in anyone 12 or older (or age unknown). It can be a heart
    #    attack, and a caregiver cannot rule that out. Sweating is noted if present.
    if adult and (pain := _find(CHEST_PAIN_RE, text)):
        sweat = _find(SWEATING_RE, text)
        add("chest_pain", "Chest pain", f'Intake says: "{pain}"' + (f' and "{sweat}"' if sweat else ""))

    # 3. Pregnant AND any bleeding: always urgent, even if not called "heavy"
    if record.pregnant and (clause := _find(BLEEDING_RE, text)):
        add("pregnancy_bleeding", "Pregnant and bleeding", f'Patient may be pregnant; intake '
            f'says: "{clause}"')

    # 4. Any fever in a baby under 1 year (age 0). Young babies with fever can
    #    get very sick quickly.
    if record.age == 0 and (clause := _find(FEVER_RE, text)):
        add("fever_baby", "Fever in a baby", f'Baby under 1 year; intake says: "{clause}"')

    # 5. Vitals
    v = record.vitals
    if v.breathing_rate is not None and _fast_breathing(record.age, v.breathing_rate):
        add("breathing", "Trouble breathing",
            f"Breathing rate {v.breathing_rate} per minute is outside the safe range for the age")
    if v.temp_c is not None and (v.temp_c >= 40 or v.temp_c < 35):
        add("high_fever", "Very high or very low temperature",
            f"Temperature {v.temp_c} C is outside the safe range")
    if record.age == 0 and v.temp_c is not None and v.temp_c >= 38:
        add("fever_baby", "Fever in a baby", f"Baby under 1 year with temperature {v.temp_c} C")
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
