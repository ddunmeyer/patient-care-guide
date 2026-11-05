"""Purpose of this program: turn the raw tool results into a numbered
evidence list that the agents read and cite.

Each item gets a short id:
    M1, M2   MedlinePlus health topics      (plain-language patient info)
    P1..P10  PubMed reviews and guidelines  (with abstracts)
    D1..     openFDA drug labels            (approved drugs, manufacturers)
    W1..     web pages from trusted sites   (WHO, CDC, NICE... via Tavily)
    NCT...   ClinicalTrials.gov trials      (their real NCT id is the id)

The agents see this list as text and must cite ids. The grounding guardrail
uses the same EvidenceCatalog to check those ids and to build the source list
with real titles and links.
"""

from dataclasses import dataclass, field

from agents.care_models import SourceRef


@dataclass
class EvidenceCatalog:
    sources: dict[str, SourceRef] = field(default_factory=dict)  # id -> title/url/type
    texts: dict[str, str] = field(default_factory=dict)  # id -> text shown to the model
    trials: dict[str, dict] = field(default_factory=dict)  # NCT id -> trial record
    names: set[str] = field(default_factory=set)  # every organisation named in a tool result
    errors: list[str] = field(default_factory=list)  # tools that failed or were off

    def prompt_text(self) -> str:
        """The evidence as the model sees it, one block per item."""
        return "\n\n".join(f"[{sid}] {text}" for sid, text in self.texts.items())

    def search_text(self) -> str:
        """All tool text in lower case, to check that a name came from the tools."""
        return "\n".join(self.texts.values()).lower()


def build_catalog(evidence: dict[str, dict]) -> EvidenceCatalog:
    """evidence is the dict from tools.evidence.gather_evidence()."""
    cat = EvidenceCatalog()
    for name, result in evidence.items():
        if result.get("error"):
            cat.errors.append(result["error"])

    def add(sid: str, title: str, url: str, kind: str, text: str) -> None:
        cat.sources[sid] = SourceRef(id=sid, title=title, url=url, type=kind)
        cat.texts[sid] = text

    for i, t in enumerate(evidence["medlineplus"]["results"], 1):
        add(f"M{i}", f"MedlinePlus: {t['title']}", t["url"], "MedlinePlus",
            f"MedlinePlus health topic '{t['title']}': {t['summary']}")

    for i, a in enumerate(evidence["pubmed"]["results"], 1):
        add(f"P{i}", f"{a['title']} ({a['journal']}, {a['year']})", a["url"], "PubMed",
            f"PubMed {', '.join(a['types'])} '{a['title']}' ({a['journal']}, {a['year']}). "
            f"Abstract: {a.get('abstract') or '(no abstract)'}")

    for i, d in enumerate(evidence["openfda"]["results"], 1):
        add(f"D{i}", f"FDA label: {d['generic_name']} ({d['brand_name']})", d["url"],
            "openFDA label",
            f"US FDA-approved drug label listing this condition: {d['generic_name']} "
            f"(brand {d['brand_name']}), labeler {d['manufacturer']}.")
        cat.names.add(d["manufacturer"])

    for i, w in enumerate(evidence["web"]["results"], 1):
        add(f"W{i}", w["title"], w["url"], "Web", f"Web page '{w['title']}' ({w['url']}): "
                                                  f"{w['snippet']}")

    for t in evidence["clinicaltrials"]["results"]:
        nct = t["nct_id"]
        cat.trials[nct] = t
        interventions = "; ".join(f"{i['name']} ({i['type']})" for i in t["interventions"])
        add(nct, f"{nct}: {t['title']}", t["url"], "ClinicalTrials.gov",
            f"Clinical trial {nct}, phase {t['phase']}, status {t['status']}. "
            f"Title: {t['title']}. Interventions: {interventions}. "
            f"Lead sponsor: {t['lead_sponsor']}. "
            f"Collaborators: {', '.join(t['collaborators']) or 'none'}. "
            f"Countries: {', '.join(t['countries']) or 'not listed'}.")
        cat.names.update([t["lead_sponsor"], *t["collaborators"]])

    cat.names.discard(None)
    return cat
