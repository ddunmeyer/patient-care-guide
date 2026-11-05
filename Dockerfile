# syntax=docker/dockerfile:1
# Purpose of this file: build a container image of the Patient Intake &
# Condition Guide (care_app.py; the original member assistant, app.py, is also
# included). It installs the Python packages, builds the ChromaDB vector store
# from data/docs/ (python -m rag.ingest), and starts the Streamlit app on 8501.
#
# No API keys are stored in the image. Ingest needs OPENAI_API_KEY to embed the
# documents, so the build reads it from a BuildKit secret that is mounted only
# for that one step. At runtime, pass the keys as environment variables.

FROM python:3.14-slim

# Don't write .pyc files, and print logs straight away.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# A Unicode font so the Patient Info Sheet PDF can show accents and other
# scripts (agents/care_export.py looks for DejaVu Sans).
RUN apt-get update && apt-get install -y --no-install-recommends fonts-dejavu-core     && rm -rf /var/lib/apt/lists/*

# Install packages first so this layer is cached when only code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app code and source documents (.dockerignore keeps secrets out).
COPY . .

# Build the vector store. The key is read from /run/secrets/ for this step only
# and never written to an image layer.
RUN --mount=type=secret,id=openai_api_key,required=true \
    OPENAI_API_KEY="$(cat /run/secrets/openai_api_key)" python -m rag.ingest

EXPOSE 8501

# Starts the Patient Intake & Condition Guide (this branch's app). For the
# original member assistant, add this to docker run:
#   streamlit run app.py --server.address 0.0.0.0

CMD ["streamlit", "run", "care_app.py", "--server.address", "0.0.0.0"]
