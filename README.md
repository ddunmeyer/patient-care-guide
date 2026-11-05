# Agentic RAG Starter Kit

A starter project for a question-answering assistant over your own PDF documents.
A coordinator agent passes each question to specialist agents. The specialists
use tools to search a ChromaDB vector store of your documents, and the answer
comes back with citations. Guardrails check questions and answers, and Langfuse
traces every run.

Built with LangChain, LangGraph, ChromaDB, OpenAI and Streamlit.

## Architecture

```mermaid
flowchart LR
    U[User] --> S[Streamlit app<br/>app.py]
    S --> G1{{Guardrails<br/>input checks}}
    G1 --> C[Coordinator agent<br/>agents/coordinator.py]
    C --> SP[Specialist agents<br/>agents/specialists.py]
    SP --> T[Tools<br/>agents/tools.py]
    T --> R[Retriever<br/>rag/retriever.py]
    R --> DB[(ChromaDB<br/>chroma_db/)]
    I[Ingest<br/>rag/ingest.py] --> DB
    P[PDFs<br/>data/docs/] --> I
    SP --> C
    C --> G2{{Guardrails<br/>output checks}}
    G2 --> A[Answer with citations]
    A --> S
    LF[Langfuse tracing] -.-> C
    LF -.-> SP
    LF -.-> T
```

## Project layout

| Path | Job |
| --- | --- |
| `app.py` | Streamlit chat UI (entry point) |
| `agents/coordinator.py` | Routes questions to specialists and assembles the answer |
| `agents/specialists.py` | Specialist agents that do the work |
| `agents/tools.py` | Tools the agents can call, such as document search |
| `rag/ingest.py` | Loads .md, .txt and .pdf files from `data/docs/` into ChromaDB |
| `rag/retriever.py` | Finds relevant chunks and answers from them with citations |
| `guardrails/checks.py` | PII redaction, prompt-injection blocking, medical-advice blocking, missing-source flag |
| `evals/eval_set.json` | Test questions with expected answers |
| `evals/run_eval.py` | Scores the pipeline against the test set |
| `data/members.json` | Fake member records used by the `lookup_member` tool |
| `data/docs/` | Source documents (sample: fictional Lakeshore Health PPO plan) |

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create a `.env` file with your keys:

```
OPENAI_API_KEY=...
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_HOST=https://us.cloud.langfuse.com
```

Build the vector store and ask a question from the terminal:

```powershell
python -m rag.ingest
python -m rag.retriever "Is an MRI covered and does it need prior authorization?"
python -m agents.coordinator   # runs the memory and guardrail tests
```

Then run `streamlit run app.py`.

> Status: the RAG layer, agents, conversation memory (in RAM, per thread_id), rule-based guardrails and the Streamlit app work. Prior auth drafts are only saved to `data/approved_requests.json` after a person clicks Approve. Evals and Langfuse tracing are still skeletons.
> Member data in `data/members.json` is fake.
