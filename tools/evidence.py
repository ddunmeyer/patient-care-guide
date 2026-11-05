"""Purpose of this program: run all the condition lookup tools at once.

Each tool takes a condition name and returns a dict:
    {"tool": name, "query": what was searched,
     "results": [ {..., "url": ...}, ... ],
     "error": None or a short message}
A tool never raises: if an API is down it returns an empty list and an error,
so one failed lookup doesn't stop the briefing.

gather_evidence() runs all five at once (in threads, because each one mostly
waits on the network) and returns them in one dict. The agents only ever see
this evidence, which is what keeps their answers grounded.
"""

from concurrent.futures import ThreadPoolExecutor

from tools.clinicaltrials import clinicaltrials_search
from tools.medlineplus import medlineplus_lookup
from tools.openfda import openfda_drugs
from tools.pubmed import pubmed_search
from tools.web_search import web_search

GUIDELINE_QUERY = "{} treatment guidelines WHO CDC NICE"


def gather_evidence(condition: str) -> dict[str, dict]:
    """Run every lookup tool for one condition. Results are cached 24 hours."""
    jobs = {
        "clinicaltrials": lambda: clinicaltrials_search(condition),
        "pubmed": lambda: pubmed_search(condition),
        "openfda": lambda: openfda_drugs(condition),
        "medlineplus": lambda: medlineplus_lookup(condition),
        "web": lambda: web_search(GUIDELINE_QUERY.format(condition)),
    }
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futures = {name: pool.submit(job) for name, job in jobs.items()}
        return {name: f.result() for name, f in futures.items()}
