"""Purpose of this program: the PatientEducationAgent, which writes a
plain-language Patient Info Sheet about the condition the patient was told
they have.

Steps in make_patient_sheet():
  1. Personal notes are written by plain Python from the intake (pregnancy,
     allergies, medicines, other conditions, young child). The model never
     decides these, so a safety note can't be left out.
  2. gpt-4o-mini writes the sheet from the numbered evidence only, in the
     patient's preferred language, at about a 6th-grade reading level. Every
     point carries the ids of the evidence it came from.
  3. The grounding guardrail (guardrails/grounding.py) drops unsupported
     points, fake or non-recruiting trials and anything that diagnoses,
     prescribes or gives a dose, then fills in sources and trial details.
  4. For English sheets, if the reading level is above grade 8, the model is
     asked once to simplify, and the result is checked again.

Only the patient_id and non-identifying facts (age, sex, pregnancy, allergies,
conditions, medicines) are sent; intake answers were already masked.

Try it:  python -m agents.patient_education malaria
"""

import sys

import textstat  # reading-level formulas

from agents.care_evidence import EvidenceCatalog, build_catalog
from agents.care_models import PatientSheet
from agents.intake_models import IntakeRecord
from agents.specialists import llm  # the kit's gpt-4o-mini, temperature 0
from guardrails.grounding import ground_sheet

TARGET_GRADE = 6
MAX_GRADE = 8  # above this, ask the model to simplify once

SHEET_PROMPT = """You write a patient information sheet for someone who was
told by a health worker that they have: {condition}.
A caregiver will read it with the patient. Write it in {language}.

Rules:
- Use ONLY the numbered evidence below. After each point, list the evidence
  ids it comes from in "sources", e.g. ["M1", "P2"]. If the evidence does not
  cover a section, leave that section as an empty list. Never guess.
- Reading level: about grade 6. Short sentences (under 15 words). Everyday
  words. If you must use a medical word, explain it.
- 2 to 5 points per section.
- how_it_is_usually_treated: say in general terms how it is treated. Never
  give a dose, never say how much or how often to take anything, and never
  tell the patient to start, stop or change a medicine.
- Never diagnose. Do not say the patient has or might have any other illness.
- warning_signs_get_help_now: signs that mean "go to a clinic now".
- questions_to_ask_your_health_worker: 3 to 5 short questions.
- research_studies_you_can_ask_about: up to 3 trials from the evidence whose
  status is RECRUITING. Copy the NCT id exactly. One simple sentence each.
  If none are recruiting, leave it empty.
- personal_notes: translate these notes into {language}, keeping the same
  meaning and order (if {language} is English, copy them as they are):
{notes}
"""

SIMPLIFY_PROMPT = """Rewrite this patient information sheet so it reads at
about a grade {target} level: shorter sentences, simpler words. Keep every
fact, every source id and every NCT id exactly as they are. Do not add facts."""


def personal_notes(record: IntakeRecord) -> list[str]:
    """Safety notes from the intake, written by plain Python."""
    notes = []
    if record.pregnant:
        notes.append("Tell your health worker you are pregnant, or may be pregnant. "
                     "Some medicines are not safe in pregnancy.")
    if record.allergies:
        notes.append(f"You listed an allergy to {', '.join(record.allergies)}. "
                     "Tell every health worker before you take any medicine.")
    if record.current_meds:
        notes.append(f"You take: {', '.join(record.current_meds)}. Tell your health worker "
                     "about all medicines and herbs you take. Do not stop or change any "
                     "medicine on your own.")
    if record.known_conditions:
        notes.append(f"You also have: {', '.join(record.known_conditions)}. Tell your health "
                     "worker, because this can change your care.")
    if record.age is not None and record.age < 5:
        notes.append("This is for a young child. Young children can get very sick quickly. "
                     "Watch closely for the warning signs below.")
    return notes


def patient_context(record: IntakeRecord) -> str:
    """Non-identifying facts the model may use to keep the sheet relevant."""
    return (f"Patient {record.patient_id}: age {record.age if record.age is not None else 'unknown'}, "
            f"sex {record.sex or 'unknown'}, pregnant: {record.pregnant}, "
            f"other conditions: {record.known_conditions or 'none listed'}.")


def sheet_text(sheet: PatientSheet) -> str:
    """The sheet's own words (no titles or links), for the reading-level check."""
    parts = [p.text for section in (sheet.what_it_is, sheet.common_signs,
                                    sheet.how_it_is_usually_treated,
                                    sheet.what_you_can_do_at_home,
                                    sheet.warning_signs_get_help_now) for p in section]
    parts += sheet.questions_to_ask_your_health_worker
    parts += [s.plain_summary for s in sheet.research_studies_you_can_ask_about]
    # textstat counts sentences by end marks; make sure every item ends with one.
    return " ".join(p if p.rstrip().endswith((".", "?", "!")) else p + "." for p in parts)


def reading_grade(sheet: PatientSheet) -> float:
    """Flesch-Kincaid grade level of the sheet (meaningful for English only)."""
    return textstat.flesch_kincaid_grade(sheet_text(sheet))


def _write(messages: list) -> PatientSheet:
    return llm.with_structured_output(PatientSheet, method="function_calling").invoke(
        messages, config={"run_name": "patient_education_agent"})


def make_patient_sheet(condition: str, record: IntakeRecord,
                       catalog: EvidenceCatalog) -> tuple[PatientSheet, list[str]]:
    """Write, ground and (if needed) simplify the sheet.
    Returns (sheet, notes from the grounding guardrail)."""
    language = record.preferred_language or "English"
    notes = personal_notes(record)
    system = SHEET_PROMPT.format(condition=condition, language=language,
                                 notes="\n".join(f"  - {n}" for n in notes) or "  (none)")
    human = f"{patient_context(record)}\n\nEVIDENCE:\n{catalog.prompt_text()}"
    sheet = _write([("system", system), ("human", human)])

    # Simplify once if an English sheet reads above grade 8.
    if language.lower().startswith("en") and reading_grade(sheet) > MAX_GRADE:
        sheet = _write([("system", SIMPLIFY_PROMPT.format(target=TARGET_GRADE)),
                        ("human", sheet.model_dump_json())])

    # Python, not the model, has the last word on the safety notes.
    if language.lower().startswith("en") or len(sheet.personal_notes) != len(notes):
        sheet.personal_notes = notes
    sheet.condition, sheet.language = condition, language
    return ground_sheet(sheet, catalog)


if __name__ == "__main__":
    from tools.evidence import gather_evidence
    condition = " ".join(sys.argv[1:]) or "malaria"
    record = IntakeRecord(patient_id="P-DEMO1", age=29, sex="female", pregnant=True,
                          allergies=["sulfa drugs"], current_meds=["paracetamol"])
    sheet, report = make_patient_sheet(condition, record, build_catalog(gather_evidence(condition)))
    print(sheet.model_dump_json(indent=2))
    print("\nGrade level:", round(reading_grade(sheet), 1))
    print("Guardrail:", report or "nothing removed")
