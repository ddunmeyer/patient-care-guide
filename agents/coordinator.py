"""Purpose of this program: the coordinator agent (LangGraph).

A small LangGraph StateGraph:

    START -> router -> benefits      -> END
                    -> prior_auth    -> END
                    -> out_of_scope  -> END

The router uses structured output to pick a route and give a one-line reason.
The chosen specialist answers. ask() returns the answer, which agent was used,
the route reason, the sources, any pending draft and the tools that were called.

Try it from the terminal:
    python -m agents.coordinator "What's my copay for a specialist visit? I'm member M001"
"""

import json  # tool results are JSON text
import sys  # to read the question from the command line
from typing import Literal, TypedDict

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field  # describes the router's output shape

from agents.specialists import benefits_agent, llm, prior_auth_agent


# --- State: the data passed from node to node through the graph ---
class State(TypedDict, total=False):  # total=False: keys are filled in as we go
    question: str
    route: str
    route_reason: str
    agent_used: str
    answer: str
    sources: list[dict]
    pending_draft: dict | None
    tools_called: list[str]


# --- Router ---
class RouteDecision(BaseModel):
    """The router's decision. The model must reply in exactly this shape."""
    route: Literal["benefits", "prior_auth", "out_of_scope"] = Field(
        description="Which specialist should handle the question.")
    reason: str = Field(description="One short sentence explaining the choice.")


ROUTER_PROMPT = """You route questions for a Lakeshore Health PPO member help desk.
- benefits: coverage, copays, deductibles, costs, prescription drug costs, claims.
- prior_auth: whether a service needs prior authorization, or starting a
  prior authorization request. Use this if the question asks about prior auth
  at all, even if it also asks about coverage.
- out_of_scope: anything not about this health plan (investing, legal, general
  chat) or requests for medical advice."""

# with_structured_output makes the model return a RouteDecision object
# instead of free text, so the route is always one of the three values.
router_llm = llm.with_structured_output(RouteDecision)


def router(state: State) -> State:
    """Ask the model which route to take and why."""
    decision = router_llm.invoke([("system", ROUTER_PROMPT), ("human", state["question"])])
    return {"route": decision.route, "route_reason": decision.reason}


# --- Specialist nodes ---
def run_specialist(agent, name: str, state: State) -> State:
    """Run one specialist agent, then pull its answer, the tools it called,
    its sources and any draft out of the agent's message history."""
    result = agent.invoke({"messages": [("human", state["question"])]})
    messages = result["messages"]  # the full conversation, including tool calls

    tools_called, sources, draft = [], [], None
    for msg in messages:
        # The model's tool requests: record each tool name and its arguments.
        if isinstance(msg, AIMessage):
            for call in msg.tool_calls:
                tools_called.append(f"{call['name']}({call['args']})")
        # The tool results: read sources and drafts out of the JSON.
        if isinstance(msg, ToolMessage):
            data = json.loads(msg.content)
            if msg.name == "search_plan_documents":
                sources.extend(data["sources"])
            if msg.name == "draft_prior_auth_request" and "draft_id" in data:
                draft = data

    return {
        "agent_used": name,
        "answer": messages[-1].content,  # the last message is the final answer
        "sources": sources,
        "pending_draft": draft,
        "tools_called": tools_called,
    }


def benefits(state: State) -> State:
    return run_specialist(benefits_agent, "benefits", state)


def prior_auth(state: State) -> State:
    return run_specialist(prior_auth_agent, "prior_auth", state)


def out_of_scope(state: State) -> State:
    """Politely refuse without calling any tools or specialists."""
    return {
        "agent_used": "none",
        "answer": ("Sorry, I can only help with questions about your Lakeshore "
                   "Health PPO plan, such as coverage, costs and prior "
                   "authorization. For anything else, please ask a qualified "
                   "professional."),
        "sources": [],
        "pending_draft": None,
        "tools_called": [],
    }


# --- Build the graph ---
graph = StateGraph(State)
graph.add_node("router", router)
graph.add_node("benefits", benefits)
graph.add_node("prior_auth", prior_auth)
graph.add_node("out_of_scope", out_of_scope)

graph.add_edge(START, "router")  # every question starts at the router
# After the router, go to the node whose name matches state["route"].
graph.add_conditional_edges("router", lambda state: state["route"])
for node in ("benefits", "prior_auth", "out_of_scope"):
    graph.add_edge(node, END)  # each specialist finishes the run

app = graph.compile()  # turn the description into something we can run


def ask(question: str) -> dict:
    """Run one question through the graph and return the final state."""
    return app.invoke({"question": question})


# Command-line test: print the route, tools, answer, sources and any draft.
if __name__ == "__main__":
    questions = [" ".join(sys.argv[1:])] if sys.argv[1:] else [
        "What's my copay for a specialist visit? I'm member M001",
        "I need an MRI, does it need prior auth? Please start the request. M002",
        "What stocks should I buy?",
    ]
    for q in questions:
        r = ask(q)
        print("=" * 70)
        print(f"Question: {q}")
        print(f"Route:    {r['route']}  ({r['route_reason']})")
        print(f"Agent:    {r['agent_used']}")
        print("Tools:   ", "\n          ".join(r["tools_called"]) or "(none)")
        print(f"\nAnswer:\n{r['answer']}\n")
        print("Sources:", ", ".join(sorted({s["source"] for s in r["sources"]})) or "(none)")
        if r["pending_draft"]:
            print("Pending draft:", json.dumps(r["pending_draft"], indent=2))
