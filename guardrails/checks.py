"""Purpose of this program: guardrail checks on questions and answers.

Screens user input (off-topic requests, prompt injection, personal data) before
it reaches the agents, and checks answers before they are shown (for example,
that every claim is backed by a cited source).
"""
