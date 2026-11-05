"""Purpose of this program: the coordinator agent (LangGraph).

Receives the user's question, decides which specialist agent(s) should handle
it, runs guardrail checks on input and output, and assembles the final answer
with citations. Each run is traced in Langfuse.
"""
