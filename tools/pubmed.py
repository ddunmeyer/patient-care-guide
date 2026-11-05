"""Purpose of this program: find recent reviews and guidelines about a
condition on PubMed.

Uses NCBI E-utilities (no key needed for light use): esearch finds article
ids, esummary gets their title, journal and year, and efetch gets the
abstracts (the text the agents cite). Only reviews, systematic
reviews and guidelines from the last 3 years are searched, and the condition
must be in the title, so results are on topic.

Try it:  python -m tools.pubmed tuberculosis
"""

import sys
import time
import xml.etree.ElementTree as ET

import httpx

from tools.cache import cached

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
ARTICLE_TYPES = ("review[pt] OR systematic review[pt] OR guideline[pt] OR "
                 "practice guideline[pt]")
THREE_YEARS_IN_DAYS = 3 * 365
ABSTRACT_CHARS = 1500  # enough for the key points, short enough for the prompt


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


def _abstracts(ids: list[str]) -> dict[str, str]:
    """efetch: the abstract text for each id ({pmid: text}). efetch only
    answers in XML, so we read the <AbstractText> parts of each article."""
    time.sleep(0.4)  # stay under NCBI's 3 requests a second
    response = httpx.get(f"{EUTILS}/efetch.fcgi", timeout=30, params={
        "db": "pubmed", "id": ",".join(ids), "retmode": "xml", "rettype": "abstract",
        "tool": "patient-care-guide"})
    response.raise_for_status()
    texts = {}
    for article in ET.fromstring(response.text).iter("PubmedArticle"):
        pmid = article.findtext(".//PMID")
        parts = ["".join(a.itertext()) for a in article.iter("AbstractText")]
        texts[pmid] = " ".join(parts)[:ABSTRACT_CHARS]
    return texts


@cached("pubmed_search_v2")  # v2: results now include abstracts
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

        # 3. efetch: the abstracts.
        abstracts = _abstracts(ids)
    except (httpx.HTTPError, ValueError, KeyError, ET.ParseError) as e:
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
            "abstract": abstracts.get(pmid, ""),
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        })
    return {"tool": "pubmed_search", "query": condition, "results": results, "error": None}


if __name__ == "__main__":
    out = pubmed_search(" ".join(sys.argv[1:]) or "tuberculosis")
    print(out["error"] or f"{len(out['results'])} articles (cached={out['cached']})")
    for a in out["results"]:
        print(f"- {a['year']} {a['journal']}: {a['title'][:80]}")
        print(f"    {a['abstract'][:120]}")
