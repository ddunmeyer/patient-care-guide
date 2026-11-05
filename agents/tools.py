"""Purpose of this program: the tools the specialist agents can call.

Defines three LangChain tools so agents look things up instead of guessing:
  - search_plan_documents: answers from the plan documents (rag/retriever.py)
  - lookup_member: reads a fake member record from data/members.json
  - draft_prior_auth_request: writes a prior auth draft; it never submits it

Each tool returns JSON text. The model reads it, and the coordinator parses
it back to collect sources and drafts.
"""

import json  # tools return JSON text; this builds it
import uuid  # makes a short unique id for each draft
from datetime import date  # today's date for the draft
from pathlib import Path

from langchain_core.tools import tool  # turns a function into an agent tool

from rag.retriever import answer  # the RAG answer function from the last step

MEMBERS_FILE = Path("data/members.json")  # mock member database


def _load_member(member_id: str) -> dict | None:
    """Return one member's record, or None if the id is not found."""
    members = json.loads(MEMBERS_FILE.read_text(encoding="utf-8"))
    return members.get(member_id.strip().upper())  # "m001 " -> "M001"


# @tool reads the function name, type hints and docstring and shows them to
# the model, so the docstring tells the model when to use the tool.
@tool
def search_plan_documents(query: str) -> str:
    """Search the Lakeshore Health PPO plan documents (benefits, costs,
    prior authorization rules, drug formulary) and return a cited answer."""
    result = answer(query)  # retrieve chunks and answer from them
    return json.dumps({
        "answer": result["answer"],
        # Keep just what is needed to cite each chunk, not its full text.
        "sources": [{"source": c["source"], "location": c["location"],
                     "score": c["score"]} for c in result["chunks"]],
        "low_confidence": result["low_confidence"],
    })


@tool
def lookup_member(member_id: str) -> str:
    """Look up a member by id (for example M001): their plan, coverage type,
    deductible met so far, out-of-pocket spending and recent claims."""
    member = _load_member(member_id)
    if member is None:
        return json.dumps({"error": f"No member found with id {member_id}."})
    return json.dumps({"member_id": member_id.strip().upper(), **member})


@tool
def draft_prior_auth_request(member_id: str, service: str, reason: str) -> str:
    """Draft a prior authorization request for a member. This does NOT
    submit anything: the draft waits for a staff member to approve it.
    reason is the medical reason the member gave for the service."""
    member = _load_member(member_id)
    if member is None:
        return json.dumps({"error": f"No member found with id {member_id}."})

    # Build the draft. Status "pending_approval" means a person must review
    # it before anything is sent; the agent can never submit on its own.
    draft = {
        "draft_id": f"PA-DRAFT-{uuid.uuid4().hex[:8].upper()}",
        "status": "pending_approval",
        "member_id": member_id.strip().upper(),
        "member_name": member["name"],
        "plan": member["plan"],
        "service": service,
        "reason": reason,
        "created": date.today().isoformat(),
        "note": "Not submitted. A staff member must review and approve this draft.",
    }
    return json.dumps(draft)
