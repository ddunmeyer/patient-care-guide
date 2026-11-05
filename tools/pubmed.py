"""Purpose of this program: find recent reviews and guidelines about a
condition on PubMed.

Uses NCBI E-utilities (no key needed for light use): esearch finds article
ids, esummary gets their title, journal and year. Only reviews, systematic
reviews and guidelines from the last 3 years are searched, and the condition
must be in the title, so results are on topic.

Try it:  python -m tools.pubmed tuberculosis
"""

import sys
import time

import httpx

from tools.cache import cached

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
ARTICLE_TYPES = ("review[pt] OR systematic review[pt] OR guideline[pt] OR "
                 "practice guideline[pt]")
THREE_YEARS_IN_DAYS = 3 * 365


def _get(url: str, params: dict) -> dict:
    """GET from NCBI, trying once more after a pause if it says we're too fast.
    Without an API key NCBI allows 3 requests a second."""
    for attempt in range(2):
        response = httpx.get(url, params=params, timeout=30)
        busy = response.status_code == 429 or '"ERROR"' in response.text[:300]
        if not busy:
            break
        time.sleep(1.5)
    response.raise_for_status()
    data = response.json()
    if "ERROR" in data.get("esearchresult", {}):  # rate-limit replies come as 200 + ERROR
        raise ValueError(data["esearchresult"]["ERROR"])
    return data


@cached("pubmed_search")
def pubmed_search(condition: str) -> dict:
    common = {"db": "pubmed", "retmode": "json", "tool": "patient-care-guide"}
    try:
        # 1. esearch: the ids of matching articles, most relevant first.
        ids = _get(f"{EUTILS}/esearch.fcgi", {
            **common, "retmax": 10, "sort": "relevance",
            "datetype": "pdat", "reldate": THREE_YEARS_IN_DAYS,  # published in last 3 years
            "term": f'"{condition}"[Title] AND ({ARTICLE_TYPES})'})["esearchresult"]["idlist"]
        if not ids:
            return {"tool": "pubmed_search", "query": condition, "results": [], "error": None}

        # 2. esummary: title, journal and date for each id.
        data = _get(f"{EUTILS}/esummary.fcgi", {**common, "id": ",".join(ids)})["result"]
    except (httpx.HTTPError, ValueError, KeyError) as e:
        return {"tool": "pubmed_search", "query": condition, "results": [],
                "error": f"PubMed lookup failed: {e}"}

    results = []
    for pmid in ids:
        item = data.get(pmid, {})
        results.append({
            "pmid": pmid,
            "title": item.get("title", "").rstrip("."),
            "journal": item.get("fulljournalname") or item.get("source"),
            "year": (item.get("pubdate") or "")[:4],
            "types": item.get("pubtype", []),
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        })
    return {"tool": "pubmed_search", "query": condition, "results": results, "error": None}


if __name__ == "__main__":
    out = pubmed_search(" ".join(sys.argv[1:]) or "tuberculosis")
    print(out["error"] or f"{len(out['results'])} articles (cached={out['cached']})")
    for a in out["results"]:
        print(f"- {a['year']} {a['journal']}: {a['title'][:80]}")
