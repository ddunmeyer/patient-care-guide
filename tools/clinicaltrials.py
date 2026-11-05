"""Purpose of this program: search ClinicalTrials.gov for phase 2 and 3
trials of a condition that are recruiting, active or about to start.

Uses the public ClinicalTrials.gov API v2 (no key needed). Returns up to 20
trials with NCT ID, title, interventions, phase, status, lead sponsor,
collaborators, countries and a link. These NCT IDs are the ONLY ones the
agents may mention (guardrails/grounding.py drops any other).

Try it:  python -m tools.clinicaltrials malaria
"""

import sys

import httpx

from tools.cache import cached

API_URL = "https://clinicaltrials.gov/api/v2/studies"
STATUSES = "RECRUITING,ACTIVE_NOT_RECRUITING,NOT_YET_RECRUITING"
# Ask the API only for the fields we use, so the reply is small.
FIELDS = ("NCTId,BriefTitle,Phase,OverallStatus,LeadSponsorName,CollaboratorName,"
          "InterventionName,InterventionType,LocationCountry")


@cached("clinicaltrials_search")
def clinicaltrials_search(condition: str) -> dict:
    params = {
        "query.cond": condition,
        "filter.overallStatus": STATUSES,
        "query.term": "AREA[Phase](PHASE2 OR PHASE3)",  # phase 2 or phase 3 only
        "pageSize": 20,
        "fields": FIELDS,
    }
    try:
        response = httpx.get(API_URL, params=params, timeout=30)
        response.raise_for_status()
        studies = response.json().get("studies", [])
    except (httpx.HTTPError, ValueError) as e:
        return {"tool": "clinicaltrials_search", "query": condition, "results": [],
                "error": f"ClinicalTrials.gov lookup failed: {e}"}

    results = []
    for study in studies:
        p = study.get("protocolSection", {})  # every field lives under protocolSection
        nct_id = p.get("identificationModule", {}).get("nctId")
        sponsors = p.get("sponsorCollaboratorsModule", {})
        interventions = p.get("armsInterventionsModule", {}).get("interventions", [])
        countries = {loc.get("country") for loc in
                     p.get("contactsLocationsModule", {}).get("locations", [])}
        results.append({
            "nct_id": nct_id,
            "title": p.get("identificationModule", {}).get("briefTitle"),
            "interventions": [{"name": i.get("name"), "type": i.get("type")}
                              for i in interventions],
            "phase": "/".join(p.get("designModule", {}).get("phases", [])),
            "status": p.get("statusModule", {}).get("overallStatus"),
            "lead_sponsor": sponsors.get("leadSponsor", {}).get("name"),
            "collaborators": [c.get("name") for c in sponsors.get("collaborators", [])],
            "countries": sorted(c for c in countries if c),
            "url": f"https://clinicaltrials.gov/study/{nct_id}",
        })
    return {"tool": "clinicaltrials_search", "query": condition, "results": results,
            "error": None}


if __name__ == "__main__":
    out = clinicaltrials_search(" ".join(sys.argv[1:]) or "malaria")
    print(out["error"] or f"{len(out['results'])} trials (cached={out['cached']})")
    for t in out["results"][:5]:
        print(f"- {t['nct_id']} {t['phase']} {t['status']} | {t['lead_sponsor']} | {t['title'][:70]}")
