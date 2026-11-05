"""Purpose of this program: measure answer quality.

Runs each question in evals/eval_set.json through the agent pipeline, compares
the answers and citations with the expected results, and reports the scores
(optionally logging them to Langfuse).
"""
