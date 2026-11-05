"""Purpose of this program: the specialist agents.

Two ReAct agents built with LangGraph's create_react_agent. A ReAct agent
loops: the model reads the question, decides to call a tool, reads the tool's
result, and repeats until it can write the final answer.
  - benefits_agent: coverage and cost questions
  - prior_auth_agent: prior authorization questions and drafting requests
"""

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent  # builds the ReAct loop

from agents.tools import draft_prior_auth_request, lookup_member, search_plan_documents

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)  # 0 = consistent answers

# Rules every specialist follows. Each prompt below adds its own job.
COMMON_RULES = """Rules:
- Answer ONLY from what your tools return. Never guess plan details.
- Write specific search queries, not single words (search "copays for
  office visits, urgent care and emergency room", not "copay"). If a question
  is broad, search for the common cases and list them.
- Cite sources from search_plan_documents in square brackets, like
  [summary_of_benefits.md].
- If the member gives a member id, use lookup_member to tailor the answer.
- Never give medical advice (whether a treatment is right for someone).
  Suggest they talk to their doctor instead.
- If the tools don't have the answer, say so."""

BENEFITS_PROMPT = f"""You are the Benefits specialist for Lakeshore Health PPO.
You answer coverage and cost questions: copays, deductibles, coinsurance,
out-of-pocket costs and prescription drug costs.
{COMMON_RULES}"""

PRIOR_AUTH_PROMPT = f"""You are the Prior Authorization specialist for
Lakeshore Health PPO. You explain which services need prior authorization and
how to request it. If the member asks you to start a request, use
draft_prior_auth_request. For the reason, use only the medical reason the
member gave; if they gave none, use "Not provided - ordering provider to
supply clinical notes". Never invent a medical reason. Tell them it is a
DRAFT waiting for staff approval and has NOT been submitted.
{COMMON_RULES}"""

# Each agent gets the model, the tools it may use, its system prompt and a
# name (the name shows up in traces).
benefits_agent = create_react_agent(
    llm,
    tools=[search_plan_documents, lookup_member],
    prompt=BENEFITS_PROMPT,
    name="benefits",
)

prior_auth_agent = create_react_agent(
    llm,
    tools=[search_plan_documents, lookup_member, draft_prior_auth_request],
    prompt=PRIOR_AUTH_PROMPT,
    name="prior_auth",
)
