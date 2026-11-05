"""Purpose of this program: the Pydantic shape of a patient intake record.

The IntakeAgent (agents/intake_agent.py) fills an IntakeRecord one question
at a time. Every field is optional, because questions can be skipped or
may not apply (pregnancy for a man, smoking for a small child).

The patient is identified by patient_id only. Names are never stored here,
so the record can be sent to the model safely.
"""

from typing import Literal

from pydantic import BaseModel, Field


class Symptom(BaseModel):
    name: str = Field(description="The symptom in plain words, e.g. 'fever'.")
    duration: str | None = Field(description="How long, e.g. '3 days', or null if not said.")


class Lifestyle(BaseModel):
    smoking: str | None = Field(description="e.g. 'no', '10 cigarettes a day', or null.")
    alcohol: str | None = Field(description="e.g. 'no', '2 beers on weekends', or null.")


class Vitals(BaseModel):
    temp_c: float | None = None  # body temperature in Celsius
    pulse: int | None = None  # beats per minute
    bp_systolic: int | None = None  # blood pressure, top number
    bp_diastolic: int | None = None  # blood pressure, bottom number
    breathing_rate: int | None = None  # breaths per minute


class IntakeRecord(BaseModel):
    patient_id: str
    preferred_language: str = "English"
    age: int | None = None  # years; babies under 1 year are 0
    sex: Literal["female", "male", "other"] | None = None
    pregnant: bool | None = None  # only asked when it could apply
    weight_kg: float | None = None
    chief_complaint: str | None = None  # the main problem, in the patient's words
    symptoms: list[Symptom] = []
    # What the patient was told ("I was told I have malaria"). Never a diagnosis
    # made by this app.
    reported_condition: str | None = None
    known_conditions: list[str] = []
    current_meds: list[str] = []
    allergies: list[str] = []
    past_surgeries_hospital_stays: list[str] = []
    family_history: list[str] = []
    lifestyle: Lifestyle = Lifestyle(smoking=None, alcohol=None)
    vitals: Vitals = Vitals()
    caregiver_notes: str | None = None
