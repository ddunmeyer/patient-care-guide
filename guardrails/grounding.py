"""Purpose of this program: the grounding guardrail (output guardrail) for
the Patient Info Sheet and the Care Team Briefing.

The model writes a draft; this file decides what is allowed to reach people.
It is plain Python, so the model cannot talk its way past it. Rules:

  1. Every fact point must cite at least one real evidence id (M1, P2, D3,
     W1, NCT...). Ids that are not in the evidence are removed; a point with
     no real id left is dropped.
  2. Every research study and emerging treatment must have an NCT id that
     came back from clinicaltrials_search. Others are dropped. Studies on the
     patient sheet must also be RECRUITING. Phase, status, sponsor, title,
     countries, type and link are copied from the trial record; the name
     must be one of that trial's interventions, with any dose cut off.
  3. Every company or institution must appear in a tool result. Names from
     model memory are dropped. Names are re-sorted into companies /
     academic / government-NGO with simple keyword rules.
  4. Anything that diagnoses, prescribes or gives a dose is removed
     (safety_problem() below).
  5. A section left empty says "No current data found."
  6. The source list is built here from the cited ids, with real titles and
     links. The model never writes a URL.

Each ground_* function returns (fixed output, report), where report is a
list of plain notes like "Dropped 1 point with no real source (common_signs)".
"""

import re
from collections import Counter

from agents.care_evidence import EvidenceCatalog
from agents.care_models import (NO_DATA, Briefing, KeyPlayers, PatientSheet, Player, Point,
                                SourceRef)

REMOVED = "Removed by safety check."

# --- Rule 4: diagnose / prescribe / dose ---
SAFETY_RULES = {
    "gives a dose": [
        r"\b\d+(\.\d+)?\s?(mg|mcg|µg|g|ml|units?|iu|tablets?|pills?|capsules?|drops|puffs)\b",
        r"\bmg\s?/\s?kg\b|\bper kg\b",
        r"\b(once|twice|three times|four times|\d+ times) (a|per|each) (day|week)\b",
        r"\bevery \d+ (hours|hrs|days)\b",
        # "take it for 7 days" is a treatment length; "fever for 3 days" is fine.
        r"\b(take|taken|give|given|use|used)\b[^.]{0,40}\bfor \d+ (days|weeks|months)\b",
    ],
    "prescribes": [
        r"\byou should (take|start|stop|switch|use|begin)\b",
        r"\b(i|we) (recommend|prescribe|suggest) (that )?(you|the patient|this patient|taking)\b",
        r"\b(increase|decrease|double|reduce|raise|lower) (your|the|his|her|their) "
        r"(dose|dosage|medicine|medication|tablets)\b",
        r"\b(give|start) (him|her|them|the patient|this patient) (on )?(a |the |some )?"
        r"(dose|medicine|medication|drug|antibiotics?|tablets?|injection)\b",
    ],
    "diagnoses": [
        r"\byou (probably|likely|may|might|could|must) (also )?have\b",
        r"\b(the|this) patient (probably|likely|may|might|could) (also )?have\b",
        r"\b(i|we) (think|believe) (that )?(you|the patient) (have|has)\b",
        r"\byour diagnosis is\b",
        r"\b(these|your|the) symptoms (mean|show|prove) (that )?you have\b",
    ],
}
SAFETY_RES = {kind: re.compile("|".join(p), re.IGNORECASE) for kind, p in SAFETY_RULES.items()}
NEGATION_RE = re.compile(r"\b(do not|don't|never|not)\b", re.IGNORECASE)


def safety_problem(text: str) -> str | None:
    """Return "gives a dose" / "prescribes" / "diagnoses", or None if safe.
    "Do not stop your medicine" is allowed: a rule preceded by "do not" or
    "never" is a warning, not an instruction."""
    for kind, regex in SAFETY_RES.items():
        for match in regex.finditer(text):
            before = text[max(0, match.start() - 20):match.start()]
            if kind != "gives a dose" and NEGATION_RE.search(before):
                continue
            return kind
    return None


# --- Rule 1 + 4 + 5: fact points ---
def _clean_points(points: list[Point], catalog: EvidenceCatalog, section: str,
                  report: list[str]) -> list[Point]:
    kept, unsourced, unsafe = [], 0, Counter()
    for p in points:
        real_ids = [s.strip() for s in p.sources if s.strip() in catalog.sources]
        problem = safety_problem(p.text)
        if problem:
            unsafe[problem] += 1
        elif not real_ids:
            unsourced += 1
        else:
            kept.append(Point(text=p.text, sources=real_ids))
    if unsourced:
        report.append(f"Dropped {unsourced} point(s) with no real source ({section})")
    for kind, n in unsafe.items():
        report.append(f"Dropped {n} point(s) that {kind} ({section})")
    return kept or [Point(text=NO_DATA, sources=[])]


def _clean_texts(texts: list[str], section: str, report: list[str]) -> list[str]:
    """For lines that don't need a source (questions): safety check only."""
    kept = [t for t in texts if not safety_problem(t)]
    if len(kept) < len(texts):
        report.append(f"Dropped {len(texts) - len(kept)} line(s) that failed the safety "
                      f"check ({section})")
    return kept


def _sources_used(sections: list[list[Point]], catalog: EvidenceCatalog,
                  extra_ids: list[str] = ()) -> list[SourceRef]:
    """The sources cited in these sections, in catalog order, without repeats."""
    cited = {sid for section in sections for p in section for sid in p.sources} | set(extra_ids)
    return [ref for sid, ref in catalog.sources.items() if sid in cited]


# --- Rule 2: trials ---
def _trial(nct_id: str, catalog: EvidenceCatalog) -> dict | None:
    return catalog.trials.get(nct_id.strip().upper())


# Registry names often carry a dose or schedule ("Atorvastatin 40mg",
# "Rifapentine daily for 6 weeks"). Cut the name off where that starts.
NAME_DOSE_RE = re.compile(r"\s*(\b\d+(\.\d+)?\s?(mg|mcg|g|ml|units?|iu)\b|\b(daily|weekly|"
                          r"once|twice)\b|\btab(let)?s?\b|\bfor \d+).*$", re.IGNORECASE)


def _intervention_name(model_name: str, trial: dict) -> str:
    """Use the trial's own intervention name, without any dose. If the model's
    name matches one, keep that one; otherwise list the trial's real
    interventions (no placebo or control arms)."""
    real = []
    for i in trial["interventions"]:
        name = NAME_DOSE_RE.sub("", i["name"] or "").strip(" ,-")
        if name and name not in real and \
                not re.search(r"placebo|control|standard of care", name, re.I):
            real.append(name)
    for name in real:
        if name.lower() in model_name.lower() or model_name.lower() in name.lower():
            return name
    return " + ".join(real[:3]) or trial["title"]


def _treatment_type(model_type: str, trial: dict) -> str:
    """The type from the registry's own intervention types, not the model.
    ClinicalTrials.gov says DRUG, BIOLOGICAL, DEVICE, GENETIC, ..."""
    types = {(i.get("type") or "").upper() for i in trial["interventions"]}
    text = (trial["title"] + " " + " ".join(i["name"] or "" for i in trial["interventions"])).lower()
    if "GENETIC" in types or "gene therap" in text:
        return "gene therapy"
    if "vaccin" in text:
        return "vaccine"
    if "DEVICE" in types:
        return "device"
    if types & {"DRUG", "BIOLOGICAL", "COMBINATION_PRODUCT"}:
        return "drug"
    return model_type if model_type in ("drug", "vaccine", "device", "gene therapy") else "other"


# --- Rule 3: organisations ---
def _norm(name: str) -> str:
    """'Shanghai Jiatan Pharmatech Co., Ltd.' -> 'shanghai jiatan pharmatech co ltd'."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", name.lower())).strip()


ACADEMIC_RE = re.compile(r"universit|college|hospital|institut|school|academ|clinic|"
                         r"medical cent|research cent|consortium|faculty", re.I)
GOVERNMENT_RE = re.compile(r"ministry|centers for disease|national institutes? of|agency|"
                           r"department|world health|\bwho\b|foundation|organi[sz]ation|"
                           r"council|fund|government|public health|unicef|"
                           r"m[ée]decins|charit|trust\b|alliance|initiative|programme|"
                           r"\bnih\b|\bniaid\b|\bcdc\b", re.I)
COMPANY_RE = re.compile(r"\b(inc|ltd|llc|plc|gmbh|ag|sa|s\.a|co|corp|corporation|company|"
                        r"limited|pharma\w*|therapeutics|biotech\w*|biosciences|laboratories|"
                        r"labs)\b", re.I)


def _bucket(name: str, model_bucket: str) -> str:
    """Sort by keywords. Government/NGO first (so 'National Institute of...'
    is not academic), then companies, then academic. Fall back to the model."""
    if GOVERNMENT_RE.search(name):
        return "government_ngo"
    if COMPANY_RE.search(name):
        return "companies"
    if ACADEMIC_RE.search(name):
        return "academic_institutions"
    return model_bucket


def _clean_players(players: KeyPlayers, catalog: EvidenceCatalog,
                   report: list[str]) -> KeyPlayers:
    known = {_norm(n): n for n in catalog.names}  # normalised -> spelling in the tools
    all_text = _norm(catalog.search_text())
    out = {"companies": [], "academic_institutions": [], "government_ngo": []}
    seen, dropped = set(), []
    for bucket in out:
        for p in getattr(players, bucket):
            key = _norm(p.name)
            if not key or key in seen:
                continue
            if key in known:
                name = known[key]  # use the exact spelling from the tool result
            elif re.search(rf"\b{re.escape(key)}\b", all_text):
                name = p.name  # named in an article or web page
            else:
                dropped.append(p.name)
                continue
            if safety_problem(p.role):
                continue
            seen.add(key)
            out[_bucket(name, bucket)].append(Player(name=name, role=p.role))
    if dropped:
        report.append(f"Dropped {len(dropped)} organisation(s) not found in any tool result: "
                      f"{', '.join(dropped)}")
    return KeyPlayers(**out)


# --- The two entry points ---
SHEET_SECTIONS = ["what_it_is", "common_signs", "how_it_is_usually_treated",
                  "what_you_can_do_at_home", "warning_signs_get_help_now"]
SOC_SECTIONS = ["first_line", "second_line", "supportive_care", "when_to_refer"]


def ground_sheet(sheet: PatientSheet, catalog: EvidenceCatalog) -> tuple[PatientSheet, list[str]]:
    """Apply the rules to a Patient Info Sheet."""
    report: list[str] = []
    for name in SHEET_SECTIONS:
        setattr(sheet, name, _clean_points(getattr(sheet, name), catalog, name, report))
    sheet.questions_to_ask_your_health_worker = _clean_texts(
        sheet.questions_to_ask_your_health_worker, "questions", report)

    studies, dropped = [], []
    for s in sheet.research_studies_you_can_ask_about:
        trial = _trial(s.nct_id, catalog)
        if not trial or trial["status"] != "RECRUITING" or safety_problem(s.plain_summary):
            dropped.append(s.nct_id)
            continue
        studies.append(s.model_copy(update={
            "nct_id": trial["nct_id"], "title": trial["title"],
            "countries": trial["countries"], "status": trial["status"], "url": trial["url"]}))
    if dropped:
        report.append(f"Dropped {len(dropped)} study(ies) that are not real recruiting trials "
                      f"from ClinicalTrials.gov: {', '.join(dropped)}")
    sheet.research_studies_you_can_ask_about = studies

    sheet.sources = _sources_used([getattr(sheet, n) for n in SHEET_SECTIONS], catalog,
                                  [s.nct_id for s in studies])
    return sheet, report


def ground_briefing(briefing: Briefing, catalog: EvidenceCatalog) -> tuple[Briefing, list[str]]:
    """Apply the rules to a Care Team Briefing."""
    report: list[str] = []
    soc = briefing.standard_of_care
    for name in SOC_SECTIONS:
        setattr(soc, name, _clean_points(getattr(soc, name), catalog, name, report))
    briefing.overview = _clean_points(briefing.overview, catalog, "overview", report)
    if problem := safety_problem(briefing.field_relevance):
        report.append(f"Replaced field relevance: it {problem}")
        briefing.field_relevance = REMOVED
    briefing.field_relevance = briefing.field_relevance.strip() or NO_DATA

    treatments, dropped = [], []
    for t in briefing.emerging_treatments:
        trial = _trial(t.nct_id, catalog)
        if not trial:
            dropped.append(t.nct_id or t.name)
            continue
        treatments.append(t.model_copy(update={
            "name": _intervention_name(t.name, trial), "nct_id": trial["nct_id"],
            "type": _treatment_type(t.type, trial),
            "phase": trial["phase"], "status": trial["status"],
            "sponsor": trial["lead_sponsor"], "url": trial["url"]}))
    if dropped:
        report.append(f"Dropped {len(dropped)} treatment(s) without a real NCT id from "
                      f"ClinicalTrials.gov: {', '.join(dropped)}")
    briefing.emerging_treatments = treatments

    briefing.key_players = _clean_players(briefing.key_players, catalog, report)

    soc.sources = _sources_used([getattr(soc, n) for n in SOC_SECTIONS], catalog)
    briefing.sources = _sources_used([briefing.overview] + [getattr(soc, n) for n in SOC_SECTIONS],
                                     catalog, [t.nct_id for t in treatments])
    return briefing, report
