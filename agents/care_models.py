"""Purpose of this program: the Pydantic shapes of the two outputs, the
Patient Info Sheet and the Care Team Briefing.

The key idea for grounding: every fact is a Point, a short text plus the ids
of the sources it came from (like "M1" or "P3", see agents/care_evidence.py).
The model fills the ids; guardrails/grounding.py then drops any Point whose
ids are not real, and Python (never the model) builds the source list, the
links and the trial details from the tool results.
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

NO_DATA = "No current data found."
DISCLAIMER = ("This is general information, not medical advice. "
              "Talk to a health worker about your care.")
BRIEFING_DISCLAIMER = ("Decision support only, built from public sources listed below. "
                       "It is not a diagnosis or a prescription. Follow local guidelines "
                       "and clinical judgement.")


class Point(BaseModel):
    text: str = Field(description="One short statement.")
    sources: list[str] = Field(description="Ids of the evidence items this statement comes "
                                           "from, e.g. ['M1', 'P2']. At least one.")


class SourceRef(BaseModel):
    """A source shown to the reader. Built by Python from the tool results."""
    id: str
    title: str
    url: str
    type: str  # "MedlinePlus", "PubMed", "openFDA label", "ClinicalTrials.gov", "Web"


# --- Patient Info Sheet ---
class StudyMention(BaseModel):
    nct_id: str = Field(description="NCT id copied exactly from the trial list.")
    plain_summary: str = Field(description="One simple sentence on what the study tests.")
    # Filled in by Python from ClinicalTrials.gov, not by the model:
    title: str = ""
    countries: list[str] = []
    status: str = ""
    url: str = ""


class PatientSheet(BaseModel):
    condition: str
    language: str = "English"
    # Every section defaults to empty: if the model leaves one out, the
    # grounding guardrail shows "No current data found." instead of crashing.
    what_it_is: list[Point] = []
    common_signs: list[Point] = []
    how_it_is_usually_treated: list[Point] = []
    what_you_can_do_at_home: list[Point] = []
    warning_signs_get_help_now: list[Point] = []
    questions_to_ask_your_health_worker: list[str] = []
    research_studies_you_can_ask_about: list[StudyMention] = []
    personal_notes: list[str] = []  # from the intake: pregnancy, allergies, medicines
    sources: list[SourceRef] = []
    disclaimer: str = DISCLAIMER


# --- Care Team Briefing ---
class StandardOfCare(BaseModel):
    first_line: list[Point] = []
    second_line: list[Point] = []
    supportive_care: list[Point] = []
    when_to_refer: list[Point] = []
    sources: list[SourceRef] = []


TREATMENT_TYPES = ("drug", "vaccine", "device", "gene therapy", "other")


class EmergingTreatment(BaseModel):
    name: str = Field(description="The intervention being tested, from the trial list.")
    type: Literal["drug", "vaccine", "device", "gene therapy", "other"]

    @field_validator("type", mode="before")
    @classmethod
    def _known_type(cls, value):
        """The model sometimes copies registry words ("biological"); anything
        not in the list becomes "other". Grounding then sets the real type."""
        value = str(value).lower().replace("_", " ")
        return value if value in TREATMENT_TYPES else "other"
    nct_id: str = Field(description="NCT id copied exactly from the trial list.")
    # Filled in by Python from ClinicalTrials.gov, not by the model:
    phase: str = ""
    status: str = ""
    sponsor: str = ""
    url: str = ""


class Player(BaseModel):
    name: str = Field(description="Exactly as written in the evidence.")
    role: str = Field(description="What they do for this condition, e.g. 'sponsors trial "
                                  "NCT01234567' or 'makes an approved drug'.")


class KeyPlayers(BaseModel):
    companies: list[Player] = []
    academic_institutions: list[Player] = []
    government_ngo: list[Player] = []


class Briefing(BaseModel):
    condition: str
    generated_at: str = ""  # set by Python
    # Defaults: a section the model leaves out becomes "No current data found."
    overview: list[Point] = Field(default=[], description="2-3 sentences, each with its sources.")
    standard_of_care: StandardOfCare = StandardOfCare()
    emerging_treatments: list[EmergingTreatment] = []
    key_players: KeyPlayers = KeyPlayers()
    field_relevance: str = Field(default="", description="2-3 lines: what this means in a clinic with "
                                             "few doctors, little equipment and few drugs.")
    sources: list[SourceRef] = []
    disclaimer: str = BRIEFING_DISCLAIMER
