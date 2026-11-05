"""Purpose of this program: load source documents into the vector store.

Reads the PDFs in data/docs/, splits them into chunks, embeds the chunks with
OpenAI embeddings, and saves them in ChromaDB (chroma_db/) with source and page
metadata so answers can cite them.
"""
