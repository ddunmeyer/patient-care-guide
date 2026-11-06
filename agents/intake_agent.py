"""Purpose of this program: the IntakeAgent, which runs a structured patient
intake one plain-language question at a time and fills an IntakeRecord.

How it works:
  - QUESTIONS is a fixed, ordered list. Each question says which record field
    it fills and when it applies (pregnancy only for women and girls of
    child-bearing age, smoking only from age 12). Plain Python picks the next
    question, so the order is predictable and easy to test.
  - Every answer is masked first (guardrails/intake_pii.py): names, phone
    numbers and addresses never reach OpenAI.
  - Simple answers (age, sex, yes/no, weight, lists, vitals) are read with
    plain Python. Only free-text answers that need understanding (symptoms
    with how long they've lasted, lifestyle, a condition the patient was told
    about) go to gpt-4o-mini, which fills a small Pydantic model.
  - The caregiver can change any answer: edit_answer() re-reads it.
  - "skip", "don't know" or an empty answer leaves the field empty.

The urgent signs check (triage/urgent_signs.py) runs after every answer; the
coordinator and the app do that, not this file.

Try it:  python -m agents.intake_agent
"""

import re
import uuid
from dataclasses import dataclass, field
from typing import Callable

from pydantic import BaseModel, Field

from agents.intake_models import IntakeRecord, Lifestyle, Symptom, Vitals
from agents.specialists import llm  # the same gpt-4o-mini, temperature 0
from guardrails.intake_pii import mask_intake_text

SKIP_RE = re.compile(r"^\s*(skip|n/?a|not sure|unknown|don'?t know|do not know|"
                     r"dont know|-|\?)?\s*[.!]?\s*$", re.IGNORECASE)
NONE_RE = re.compile(r"^\s*(no|none|nothing|nil|no\s+\w+|not that I know)\s*[.!]?\s*$",
                     re.IGNORECASE)


# --- The questions ---
@dataclass
class Question:
    field: str  # the IntakeRecord field it fills
    text: str  # what the caregiver reads out, in plain words
    hint: str = ""  # an example answer, shown under the box
    applies: Callable[[IntakeRecord], bool] = lambda r: True  # skip when False


def _could_be_pregnant(r: IntakeRecord) -> bool:
    if r.sex == "male":
        return False
    return r.age is None or 10 <= r.age <= 55


QUESTIONS = [
    Question("preferred_language", "Which language does the patient prefer?",
             "English, Swahili, French, Spanish..."),
    Question("age", "How old is the patient?", "34, or '8 months' for a baby"),
    Question("sex", "Is the patient female or male?", "female / male / other"),
    Question("pregnant", "Is she pregnant, or could she be pregnant?", "yes / no / not sure",
             applies=_could_be_pregnant),
    Question("weight_kg", "About how much does the patient weigh, in pounds?", "130, or 'skip'"),
    Question("chief_complaint", "What is the main problem today? Use the patient's own words.",
             "'I have had a fever and headache'"),
    Question("symptoms", "What other signs or symptoms are there, and how long has each "
             "lasted?", "'cough for 2 weeks, tired since Monday'"),
    Question("known_conditions", "Does the patient have any long-term health problems?",
             "diabetes, HIV, high blood pressure, asthma... or 'none'"),
    Question("current_meds", "What medicines, herbs or supplements does the patient take now?",
             "'metformin, herbal tea' or 'none'"),
    Question("allergies", "Is the patient allergic to any medicines or foods?",
             "'penicillin' or 'none'"),
    Question("past_surgeries_hospital_stays", "Has the patient had any operations or hospital "
             "stays before?", "'appendix out in 2019' or 'none'"),
    Question("family_history", "Do any health problems run in the family?",
             "'mother has diabetes' or 'none'"),
    Question("lifestyle", "Does the patient smoke or drink alcohol? How much?",
             "'no smoking, beer on weekends'", applies=lambda r: r.age is None or r.age >= 12),
    Question("vitals", "If you measured them: temperature, pulse, blood pressure and "
             "breathing rate.", "'temp 38.5, pulse 110, BP 120/80, breathing 24' or 'skip'"),
    Question("caregiver_notes", "Anything else you noticed, or want the care team to know?",
             "'very tired, not eating well' or 'no'"),
]
QUESTION_BY_FIELD = {q.field: q for q in QUESTIONS}


# --- The intake session: the record plus what was asked and answered ---
@dataclass
class IntakeSession:
    record: IntakeRecord
    answers: dict[str, str] = field(default_factory=dict)  # field -> masked answer text
    known_names: list[str] = field(default_factory=list)  # masked wherever they appear
    pii_found: list[str] = field(default_factory=list)  # labels masked so far


def new_session(patient_id: str | None = None, known_names: list[str] | None = None) -> IntakeSession:
    """Start an intake. patient_id is made up if not given."""
    pid = patient_id or f"P-{uuid.uuid4().hex[:6].upper()}"
    return IntakeSession(record=IntakeRecord(patient_id=pid),
                         known_names=[n for n in (known_names or []) if n.strip()])


def next_question(session: IntakeSession) -> Question | None:
    """The first question not yet answered that applies, or None when done."""
    for q in QUESTIONS:
        if q.field not in session.answers and q.applies(session.record):
            return q
    return None


def progress(session: IntakeSession) -> tuple[int, int]:
    """(answered, total that apply), for a progress bar."""
    applicable = [q for q in QUESTIONS if q.applies(session.record)]
    return sum(1 for q in applicable if q.field in session.answers), len(applicable)


# --- Reading answers with plain Python ---
def _number(text: str) -> float | None:
    match = re.search(r"\d+(?:\.\d+)?", text)
    return float(match.group()) if match else None


def _parse_age(text: str) -> int | None:
    n = _number(text)
    if n is None:
        return None
    if re.search(r"month|week|day", text, re.IGNORECASE):
        return 0  # babies under a year count as 0 years
    return int(n)


def _parse_sex(text: str):
    t = text.lower()
    if re.search(r"\b(f|female|woman|girl|she)\b", t):
        return "female"
    if re.search(r"\b(m|male|man|boy|he)\b", t):
        return "male"
    return "other" if t.strip() else None


def _parse_yes_no(text: str) -> bool | None:
    t = text.lower()
    if re.search(r"\b(yes|yeah|yep|pregnant|maybe|possibly|could be)\b", t) and \
            not re.search(r"\bnot pregnant\b", t):
        return True  # "could be" counts as yes: safer for medicines and warnings
    if re.search(r"\b(no|not|nope)\b", t):
        return False
    return None


def _parse_list(text: str) -> list[str]:
    """'metformin, aspirin and herbal tea' -> ['metformin', 'aspirin', 'herbal tea']."""
    if NONE_RE.match(text):
        return []
    parts = re.split(r",|;|\band\b|\n", text)
    return [p.strip(" .") for p in parts if p.strip(" .")]


def _parse_vitals(text: str) -> Vitals:
    """Read labelled numbers: temp 38.5 (or 101 F), pulse 110, BP 120/80, breathing 24."""
    t = text.lower()
    vitals = Vitals()
    if m := re.search(r"(?:temp\w*|t)\s*[:=]?\s*(\d{2,3}(?:\.\d)?)\s*(°?\s*[cf])?", t):
        value = float(m.group(1))
        is_f = (m.group(2) or "").strip(" °") == "f" or value > 45  # 101 must be Fahrenheit
        vitals.temp_c = round((value - 32) * 5 / 9, 1) if is_f else value
    if m := re.search(r"(?:pulse|heart rate|hr)\s*[:=]?\s*(\d{2,3})", t):
        vitals.pulse = int(m.group(1))
    if m := re.search(r"(\d{2,3})\s*/\s*(\d{2,3})", t):
        vitals.bp_systolic, vitals.bp_diastolic = int(m.group(1)), int(m.group(2))
    if m := re.search(r"(?:breathing|breaths|resp\w*|rr)\D{0,12}(\d{1,3})", t):
        vitals.breathing_rate = int(m.group(1))
    return vitals


# --- Reading free-text answers with the model ---
class SymptomList(BaseModel):
    symptoms: list[Symptom] = Field(description="Each symptom mentioned, with how long it has "
                                                "lasted if the answer says so.")


class ConditionAnswer(BaseModel):
    condition: str | None = Field(description="The illness the patient says a health worker "
                                              "told them they have, in plain words, or null.")


EXTRACT_PROMPT = """You copy facts from a caregiver's answer into fields.
Use only what the answer says. Never add, guess or diagnose. If the answer
doesn't give a value, use null (or an empty list)."""


def _extract(model: type[BaseModel], question: str, answer: str) -> BaseModel:
    """Ask gpt-4o-mini to fill one small model from one (masked) answer."""
    return llm.with_structured_output(model).invoke(
        [("system", EXTRACT_PROMPT), ("human", f"Question: {question}\nAnswer: {answer}")],
        config={"run_name": f"intake_extract_{model.__name__}"})


def _read_answer(question: Question, answer: str):
    """Turn one masked answer into the value for question.field."""
    f = question.field
    if f == "preferred_language":
        return answer.strip().title() or "English"
    if f == "age":
        return _parse_age(answer)
    if f == "sex":
        return _parse_sex(answer)
    if f == "pregnant":
        return _parse_yes_no(answer)
    if f == "weight_kg":
        # Asked in pounds; stored in kg (the record field stays weight_kg).
        # If the caregiver writes "kg", the number is already kilograms.
        n = _number(answer)
        return n if not n or re.search(r"kg|kilo", answer, re.I) else round(n / 2.205, 1)
    if f in ("chief_complaint", "caregiver_notes"):
        return None if NONE_RE.match(answer) else answer.strip()
    if f == "vitals":
        return _parse_vitals(answer)
    if f in ("known_conditions", "current_meds", "allergies",
             "past_surgeries_hospital_stays", "family_history"):
        return _parse_list(answer)
    if f == "symptoms":
        if NONE_RE.match(answer):
            return []
        return _extract(SymptomList, question.text, answer).symptoms
    if f == "lifestyle":
        return _extract(Lifestyle, question.text, answer)
    if f == "reported_condition":
        if NONE_RE.match(answer):
            return None
        return _extract(ConditionAnswer, question.text, answer).condition
    raise ValueError(f"No reader for field {f}")


def answer_question(session: IntakeSession, field_name: str, raw_answer: str) -> IntakeSession:
    """Mask the answer, read it into the record and remember it.
    Used both for new answers and for edits."""
    question = QUESTION_BY_FIELD[field_name]
    masked, found = mask_intake_text(raw_answer, session.known_names)
    session.pii_found = sorted(set(session.pii_found) | set(found))
    session.answers[field_name] = masked

    if SKIP_RE.match(masked):
        value = IntakeRecord.model_fields[field_name].get_default(call_default_factory=True)
    else:
        value = _read_answer(question, masked)
    # model_copy(update=...) keeps the rest of the record and sets this field.
    session.record = session.record.model_copy(update={field_name: value})

    # If an earlier answer changes who the questions apply to (e.g. sex edited
    # to male), forget answers to questions that no longer apply.
    for q in QUESTIONS:
        if q.field in session.answers and not q.applies(session.record):
            del session.answers[q.field]
            session.record = session.record.model_copy(
                update={q.field: IntakeRecord.model_fields[q.field].get_default(
                    call_default_factory=True)})
    return session


edit_answer = answer_question  # an edit is just a new answer to the same question


if __name__ == "__main__":
    demo = {
        "preferred_language": "swahili",
        "age": "29",
        "sex": "woman",
        "pregnant": "yes, about 5 months",
        "weight_kg": "skip",
        "chief_complaint": "My name is Amina, I have had fever and chills",
        "symptoms": "fever for 3 days, headache since yesterday, vomiting twice",
        "reported_condition": "the clinic said I have malaria",
        "known_conditions": "none",
        "current_meds": "paracetamol and some herbal tea",
        "allergies": "sulfa drugs",
        "past_surgeries_hospital_stays": "no",
        "family_history": "mother has high blood pressure",
        "lifestyle": "no smoking, no alcohol",
        "vitals": "temp 38.9, pulse 104, BP 110/70, breathing 22",
        "caregiver_notes": "lives at 12 Market Road, phone +254 712 345 678",
    }
    s = new_session(known_names=["Amina Hassan"])
    while (q := next_question(s)) is not None:
        print(f"Q: {q.text}\nA: {demo[q.field]}")
        answer_question(s, q.field, demo[q.field])
    print("\nMasked:", s.pii_found)
    print(s.record.model_dump_json(indent=2))
    edit_answer(s, "sex", "male")  # pregnancy no longer applies, so it is cleared
    print("\nAfter editing sex to male, pregnant =", s.record.pregnant)
