"""Purpose of this program: find US FDA-approved drug labels that list a
condition in their "indications and usage" section.

Uses the public openFDA drug label API (no key needed for light use). Only
labels with brand/generic names are kept, and each generic drug is listed
once. The link goes to the label on DailyMed. This shows which drugs are
approved for the condition; it is never used to pick a drug or a dose.

Try it:  python -m tools.openfda "type 2 diabetes"
"""

import re
import sys

import httpx

from tools.cache import cached

API_URL = "https://api.fda.gov/drug/label.json"
MAX_DRUGS = 15
# Some labels put the strength in the name ("Metformin Er 500 Mg"). Strip it:
# this app never shows doses.
DOSE_RE = re.compile(r"\s*\d+(\.\d+)?\s*(mg|mcg|g|ml|units?|%)\b.*$", re.IGNORECASE)


def _name(values: list[str]) -> str:
    return DOSE_RE.sub("", ", ".join(values)).strip().title()


@cached("openfda_drugs")
def openfda_drugs(condition: str) -> dict:
    # Search the indications text for the exact phrase; skip labels without names.
    search = f'indications_and_usage:"{condition}" AND _exists_:openfda.generic_name'
    try:
        response = httpx.get(API_URL, params={"search": search, "limit": 100}, timeout=30)
        if response.status_code == 404:  # openFDA says 404 when nothing matches
            return {"tool": "openfda_drugs", "query": condition, "results": [], "error": None}
        response.raise_for_status()
        labels = response.json().get("results", [])
    except (httpx.HTTPError, ValueError) as e:
        return {"tool": "openfda_drugs", "query": condition, "results": [],
                "error": f"openFDA lookup failed: {e}"}

    results, seen = [], set()
    for label in labels:
        names = label.get("openfda", {})
        generic = _name(names.get("generic_name", []))
        if not generic or generic.lower() in seen:
            continue  # one entry per generic drug
        seen.add(generic.lower())
        results.append({
            "generic_name": generic,
            "brand_name": _name(names.get("brand_name", [])),
            "manufacturer": ", ".join(names.get("manufacturer_name", [])),
            "url": f"https://dailymed.nlm.nih.gov/dailymed/lookup.cfm?setid={label.get('set_id')}",
        })
        if len(results) == MAX_DRUGS:
            break
    return {"tool": "openfda_drugs", "query": condition, "results": results, "error": None}


if __name__ == "__main__":
    out = openfda_drugs(" ".join(sys.argv[1:]) or "type 2 diabetes")
    print(out["error"] or f"{len(out['results'])} drugs (cached={out['cached']})")
    for d in out["results"]:
        print(f"- {d['generic_name']} ({d['brand_name']}) by {d['manufacturer']}")
