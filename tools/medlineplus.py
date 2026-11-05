"""Purpose of this program: get plain-language patient information about a
condition from MedlinePlus (US National Library of Medicine).

Uses the MedlinePlus health topics web service (no key needed). It returns
XML; we read the top topics' title, summary and link, and strip the HTML
tags from the summary. This is the main source for the patient info sheet.

Try it:  python -m tools.medlineplus "sickle cell disease"
"""

import html
import re
import sys
import xml.etree.ElementTree as ET

import httpx

from tools.cache import cached

API_URL = "https://wsearch.nlm.nih.gov/ws/query"


def _plain(text: str) -> str:
    """Turn '&lt;p&gt;Malaria is...' into 'Malaria is...'."""
    text = re.sub(r"<[^>]+>", " ", html.unescape(text or ""))
    return re.sub(r"\s+", " ", text).strip()


@cached("medlineplus_lookup")
def medlineplus_lookup(condition: str) -> dict:
    try:
        response = httpx.get(API_URL, timeout=30,
                             params={"db": "healthTopics", "term": condition, "retmax": 3})
        response.raise_for_status()
        root = ET.fromstring(response.text)
    except (httpx.HTTPError, ET.ParseError) as e:
        return {"tool": "medlineplus_lookup", "query": condition, "results": [],
                "error": f"MedlinePlus lookup failed: {e}"}

    results = []
    for doc in root.iter("document"):
        # Each <content name="..."> holds one field of the topic.
        fields = {c.get("name"): c.text for c in doc.findall("content")}
        results.append({
            "title": _plain(fields.get("title")),
            "summary": _plain(fields.get("FullSummary"))[:2500],
            "url": doc.get("url"),
        })

    # Keep topics whose title shares a word with the condition ("Sickle Cell
    # Disease" for "sickle cell disease", not "Ischemic Stroke"). If none do,
    # keep the top result, which MedlinePlus ranks as the best match.
    words = {w for w in re.findall(r"[a-z]+", condition.lower()) if len(w) > 3}
    on_topic = [r for r in results if words & set(re.findall(r"[a-z]+", r["title"].lower()))]
    results = on_topic or results[:1]
    return {"tool": "medlineplus_lookup", "query": condition, "results": results, "error": None}


if __name__ == "__main__":
    out = medlineplus_lookup(" ".join(sys.argv[1:]) or "sickle cell disease")
    print(out["error"] or f"{len(out['results'])} topics (cached={out['cached']})")
    for t in out["results"]:
        print(f"- {t['title']} {t['url']}\n  {t['summary'][:150]}...")
