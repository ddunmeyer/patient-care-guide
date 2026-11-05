"""Purpose of this program: find the document chunks relevant to a question
and answer the question using only those chunks.

Opens the ChromaDB store built by rag/ingest.py. retrieve() returns the
best-matching chunks with their source and a relevance score. answer() sends
those chunks to gpt-4o-mini, which answers only from them and cites sources.

Try it from the terminal:
    python -m rag.retriever "Is an MRI covered and does it need prior authorization?"
"""

import sys  # to read the question from the command line

from dotenv import load_dotenv  # reads OPENAI_API_KEY from the .env file
from langchain_chroma import Chroma  # LangChain wrapper around ChromaDB
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Reuse ingest's settings so we open the same database with the same model.
from rag.ingest import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL

load_dotenv()  # make OPENAI_API_KEY available before any OpenAI call

CHAT_MODEL = "gpt-4o-mini"  # the model that writes the answer
# Score cutoffs, picked by testing questions against these documents.
# Off-topic questions score about 0.1; strong matches score 0.6 to 0.7.
MIN_SCORE = 0.25  # chunks below this score are treated as not relevant
LOW_CONFIDENCE_SCORE = 0.5  # if the best chunk is below this, flag the answer
NOT_FOUND = "I don't have that in the plan documents."

# The rules the model must follow. {context} is filled in with the chunks.
SYSTEM_PROMPT = """You answer questions about the Lakeshore Health PPO plan.
Use ONLY the plan document excerpts below. Do not use outside knowledge.
After each fact, cite the file it came from in square brackets, like
[summary_of_benefits.md]. Cite every file you used.
If the excerpts do not contain the answer, reply exactly:
"{not_found}"

Plan document excerpts:
{context}"""

# Open the saved database once, when the module is first imported.
_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL),
    persist_directory=str(CHROMA_DIR),
)
_llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)  # 0 = stick to the facts


def retrieve(query: str, k: int = 4) -> list[dict]:
    """Return the k chunks most similar to the query, best first.

    Each result is a dict with the chunk text, its source file, where in the
    file it came from (section heading or page number) and a relevance score
    from 0 to 1, where higher means more relevant.
    """
    # Embeds the query, finds the k nearest chunk vectors, and returns
    # (Document, score) pairs. With cosine distance, score = 1 - distance.
    results = _store.similarity_search_with_relevance_scores(query, k=k)
    return [
        {
            "text": doc.page_content,
            "source": doc.metadata["source"],
            # Markdown chunks have a section, PDF chunks have a page.
            "location": doc.metadata.get("section") or f"page {doc.metadata.get('page')}",
            "score": round(score, 3),
        }
        for doc, score in results
    ]


def answer(query: str) -> dict:
    """Answer the query from the plan documents only.

    Returns a dict with the answer text, the chunks used, the best score and
    a low_confidence flag.
    """
    # Keep only chunks that are similar enough to count as relevant.
    chunks = [c for c in retrieve(query) if c["score"] >= MIN_SCORE]

    # Nothing relevant: say so without calling the model, so it can't guess.
    if not chunks:
        return {"answer": NOT_FOUND, "chunks": [], "best_score": None,
                "low_confidence": True}

    # Join the chunks into one block of text, each labelled with its source.
    context = "\n\n".join(f"[{c['source']}] ({c['location']})\n{c['text']}"
                          for c in chunks)

    # Send the rules plus the chunks as the system message, the question as
    # the user message, and get the model's reply.
    reply = _llm.invoke([
        ("system", SYSTEM_PROMPT.format(context=context, not_found=NOT_FOUND)),
        ("human", query),
    ])

    best = chunks[0]["score"]  # results are sorted, so the first is the best
    text = reply.content

    # If the model found no answer in the chunks, don't list them as sources.
    if NOT_FOUND in text:
        chunks = []

    return {
        "answer": text,
        "chunks": chunks,
        "best_score": best,
        "low_confidence": best < LOW_CONFIDENCE_SCORE,
    }


# Command-line test: print the answer, then each source and its score.
if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "Is an MRI covered and does it need prior authorization?"
    result = answer(question)

    print(f"Question: {question}\n")
    if result["low_confidence"]:
        print("[LOW CONFIDENCE] The best match is weak; check the sources.\n")
    print(f"Answer:\n{result['answer']}\n")
    print("Sources:" if result["chunks"] else "Sources: (none)")
    for c in result["chunks"]:
        print(f"  {c['score']:.3f}  {c['source']}  ({c['location']})")
