"""Purpose of this program: the grounding guardrail for the Patient Info
Sheet and the Care Team Briefing.

It runs on the model's draft and makes sure what reaches people comes from
the lookup tools. Python, not the model, fills in every source title and
link and every trial's details from the tool results.
"""

from agents.care_evidence import EvidenceCatalog
from agents.care_models import NO_DATA, Briefing, PatientSheet, Point, SourceRef


def _sources_used(sections: list[list[Point]], catalog: EvidenceCatalog,
                  extra_ids: list[str] = ()) -> list[SourceRef]:
    """The sources cited in these sections, in catalog order, without repeats."""
    cited = {sid for section in sections for p in section for sid in p.sources} | set(extra_ids)
    return [ref for sid, ref in catalog.sources.items() if sid in cited]


def _no_data_if_empty(points: list[Point]) -> list[Point]:
    return points or [Point(text=NO_DATA, sources=[])]


def ground_sheet(sheet: PatientSheet, catalog: EvidenceCatalog) -> tuple[PatientSheet, list[str]]:
    """Fill in sources and trial details on a Patient Info Sheet.
    Returns (sheet, list of notes about what was changed)."""
    report: list[str] = []
    sections = ["what_it_is", "common_signs", "how_it_is_usually_treated",
                "what_you_can_do_at_home", "warning_signs_get_help_now"]
    for name in sections:
        setattr(sheet, name, _no_data_if_empty(getattr(sheet, name)))

    studies = []
    for s in sheet.research_studies_you_can_ask_about:
        trial = catalog.trials.get(s.nct_id.strip().upper())
        if trial:
            studies.append(s.model_copy(update={
                "nct_id": trial["nct_id"], "title": trial["title"],
                "countries": trial["countries"], "status": trial["status"],
                "url": trial["url"]}))
    sheet.research_studies_you_can_ask_about = studies

    sheet.sources = _sources_used([getattr(sheet, n) for n in sections], catalog,
                                  [s.nct_id for s in studies])
    return sheet, report


def ground_briefing(briefing: Briefing, catalog: EvidenceCatalog) -> tuple[Briefing, list[str]]:
    """Fill in trial details and sources on a Care Team Briefing.
    Returns (briefing, list of notes about what was changed)."""
    report: list[str] = []
    soc = briefing.standard_of_care
    soc_sections = ["first_line", "second_line", "supportive_care", "when_to_refer"]
    for name in soc_sections:
        setattr(soc, name, _no_data_if_empty(getattr(soc, name)))
    briefing.overview = _no_data_if_empty(briefing.overview)

    treatments = []
    for t in briefing.emerging_treatments:
        trial = catalog.trials.get(t.nct_id.strip().upper())
        if trial:
            treatments.append(t.model_copy(update={
                "nct_id": trial["nct_id"], "phase": trial["phase"], "status": trial["status"],
                "sponsor": trial["lead_sponsor"], "url": trial["url"]}))
    briefing.emerging_treatments = treatments

    soc.sources = _sources_used([getattr(soc, n) for n in soc_sections], catalog)
    briefing.sources = _sources_used([briefing.overview] + [getattr(soc, n) for n in soc_sections],
                                     catalog, [t.nct_id for t in treatments])
    return briefing, report
