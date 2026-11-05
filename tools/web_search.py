"""Purpose of this program: search the web for current treatment guidelines
with Tavily, limited to trusted health sites (WHO, CDC, NICE, NIH and
specialty societies).

Needs TAVILY_API_KEY in .env. Without a key it returns no results and a note
saying web search is off; everything else keeps working.

Try it:  python -m tools.web_search "hypertension treatment guidelines"
"""

import os
import sys

import httpx
from dotenv import load_dotenv

from tools.cache import cached

load_dotenv()
API_URL = "https://api.tavily.com/search"
# Only trusted guideline sources. Add a society's site here to include it.
TRUSTED_DOMAINS = [
    "who.int", "cdc.gov", "nice.org.uk", "nih.gov", "ncbi.nlm.nih.gov",
    "medlineplus.gov", "msf.org", "ahajournals.org", "acc.org", "diabetes.org",
    "idf.org", "stoptb.org", "malariaconsortium.org", "sicklecelldisease.org",
    "thoracic.org", "idsociety.org", "paho.org", "unicef.org", "afro.who.int",
]


@cached("web_search")
def web_search(query: str) -> dict:
    key = os.getenv("TAVILY_API_KEY")
    if not key:
        return {"tool": "web_search", "query": query, "results": [],
                "error": "Web search is off: add TAVILY_API_KEY to .env to turn it on."}
    try:
        response = httpx.post(API_URL, timeout=30,
                              headers={"Authorization": f"Bearer {key}"},
                              json={"query": query, "max_results": 6, "search_depth": "basic",
                                    "include_domains": TRUSTED_DOMAINS})
        response.raise_for_status()
        hits = response.json().get("results", [])
    except (httpx.HTTPError, ValueError) as e:
        return {"tool": "web_search", "query": query, "results": [],
                "error": f"Web search failed: {e}"}

    results = [{"title": h.get("title"), "url": h.get("url"),
                "snippet": (h.get("content") or "")[:800]} for h in hits]
    return {"tool": "web_search", "query": query, "results": results, "error": None}


if __name__ == "__main__":
    out = web_search(" ".join(sys.argv[1:]) or "hypertension treatment guidelines")
    print(out["error"] or f"{len(out['results'])} pages (cached={out['cached']})")
    for r in out["results"]:
        print(f"- {r['title']} {r['url']}")
