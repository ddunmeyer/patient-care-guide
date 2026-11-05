"""Purpose of this program: find the document chunks relevant to a question.

Opens the ChromaDB store built by rag/ingest.py and returns the best-matching
chunks with their source file and page, for the agents to use and cite.
"""
