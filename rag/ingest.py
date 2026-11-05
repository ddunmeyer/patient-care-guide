"""Purpose of this program: load source documents into the vector store.

Reads the .md, .txt and .pdf files in data/docs/, splits them into chunks,
embeds the chunks with OpenAI embeddings, and saves them in ChromaDB
(chroma_db/) with source and section/page metadata so answers can cite them.

Run it with:  python -m rag.ingest
Each run deletes the old database and builds a fresh one.
"""

import shutil  # to delete the old database folder
from pathlib import Path  # for working with file paths

from dotenv import load_dotenv  # reads OPENAI_API_KEY from the .env file
from langchain_chroma import Chroma  # LangChain wrapper around ChromaDB
from langchain_core.documents import Document  # a piece of text plus metadata
from langchain_openai import OpenAIEmbeddings  # turns text into vectors
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader  # reads text out of PDF files

# --- Settings (other modules import these so everything stays in sync) ---
DOCS_DIR = Path("data/docs")  # where the source documents live
CHROMA_DIR = Path("chroma_db")  # where ChromaDB saves its files
COLLECTION_NAME = "plan_docs"  # name of the table-like collection in Chroma
EMBEDDING_MODEL = "text-embedding-3-small"  # OpenAI embedding model
CHUNK_SIZE = 800  # max characters per chunk
CHUNK_OVERLAP = 100  # characters shared between neighbouring chunks


def load_markdown(path: Path) -> list[Document]:
    """Split a markdown file into one Document per section heading."""
    sections = []  # finished (heading, text) pairs
    heading = "Introduction"  # name for any text before the first heading
    lines = []  # lines that belong to the current section

    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):  # a new heading starts a new section
            if lines:  # save the section we were building, if it has text
                sections.append((heading, "\n".join(lines)))
            heading = line.lstrip("#").strip()  # "## Deductible" -> "Deductible"
            lines = [line]  # keep the heading in the text; it helps search
        else:
            lines.append(line)
    if lines:  # save the last section
        sections.append((heading, "\n".join(lines)))

    # Turn each section into a Document that remembers its file and heading.
    return [
        Document(page_content=text, metadata={"source": path.name, "section": name})
        for name, text in sections
    ]


def load_text(path: Path) -> list[Document]:
    """Load a plain text file as one Document (it has no sections or pages)."""
    text = path.read_text(encoding="utf-8")
    return [Document(page_content=text, metadata={"source": path.name})]


def load_pdf(path: Path) -> list[Document]:
    """Load a PDF as one Document per page, numbering pages from 1."""
    reader = PdfReader(path)
    return [
        Document(page_content=page.extract_text() or "",
                 metadata={"source": path.name, "page": number})
        for number, page in enumerate(reader.pages, start=1)
    ]


# Maps each file extension to the function that loads it.
LOADERS = {".md": load_markdown, ".txt": load_text, ".pdf": load_pdf}


def load_documents() -> list[Document]:
    """Load every supported file in DOCS_DIR, in alphabetical order."""
    docs = []
    for path in sorted(DOCS_DIR.iterdir()):
        loader = LOADERS.get(path.suffix.lower())  # None for unsupported files
        if loader:
            docs.extend(loader(path))
    return docs


def main() -> None:
    load_dotenv()  # make OPENAI_API_KEY available to the OpenAI client

    # 1. Load: read the files into Documents.
    docs = load_documents()
    print(f"Loaded {len(docs)} sections/pages from {DOCS_DIR}")

    # 2. Split: cut long sections into chunks of at most CHUNK_SIZE characters.
    #    The splitter tries paragraph breaks first, then lines, then words, so
    #    chunks break at natural places. Each chunk keeps its parent's metadata.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks")

    # 3. Rebuild cleanly: delete the old database so no stale chunks remain.
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)

    # 4. Embed and store: Chroma calls the embedding model on every chunk and
    #    saves the vectors to disk. "cosine" makes the relevance score
    #    1 - cosine distance, so higher means more similar.
    Chroma.from_documents(
        documents=chunks,
        embedding=OpenAIEmbeddings(model=EMBEDDING_MODEL),
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )
    print(f"Saved {len(chunks)} chunks to {CHROMA_DIR}/")


# Only run main() when this file is run directly, not when it is imported.
if __name__ == "__main__":
    main()
