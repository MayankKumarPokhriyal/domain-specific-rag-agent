# Domain-Specific RAG Agent

Production-grade, domain-specific Retrieval-Augmented Generation (RAG) system built with FastAPI, FAISS, Sentence-Transformers, and Ollama for fully local inference. Upload PDFs in real time, ingest and index them into a FAISS store, and answer queries with retrieval-grounded responses to minimize hallucinations.

## Features
- Real-time PDF upload and ingestion
- PDF chunking with metadata preservation
- Embeddings via Sentence-Transformers
- FAISS vector index persisted on disk
- Agent-controlled retrieval and refusal logic
- Local LLM inference through Ollama (llama3)
- Fully offline / local-first architecture
- REST API with FastAPI
- Optional Streamlit UI for interaction

## Architecture Overview
- [api/](api): FastAPI endpoints for ingestion and querying
- [ingestion/](ingestion): PDF loading, chunking, embedding, FAISS indexing
- [retrieval/](retrieval): FAISS-based retriever with defensive guards
- [generation/](generation): Prompt construction and Ollama client
- [agent/](agent): Decision logic for retrieval vs refusal
- [config/](config): Centralized settings
- [utils/](utils): Logging and helpers
- [data/](data): Uploaded documents and persisted FAISS index
- [ui.py](ui.py): Streamlit frontend (optional)

## Tech Stack
- FastAPI, Uvicorn
- Sentence-Transformers
- FAISS
- Ollama (llama3) for local LLM inference
- Streamlit (optional UI)
- Python 3.10+ (tested on Python 3.13)

## Project Structure
```
requirements.txt
ui.py
agent/
  controller.py
api/
  main.py
config/
  settings.py
data/
  faiss_index/
generation/
  generator.py
  llm_client.py
ingestion/
  chunker.py
  indexer.py
  loader.py
retrieval/
  retriever.py
tests/
  test_agent.py
utils/
  logging.py
```

## Setup
1) Create and activate a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```
2) Install dependencies
```bash
pip install -r requirements.txt
```
3) Environment variables (all prefixed with `RAG_`, see [config/settings.py](config/settings.py) for defaults):
- `RAG_DATA_DIR` (default: `data`)
- `RAG_VECTOR_STORE_PATH` (default: `data/faiss_index`)
- `RAG_CHUNK_SIZE` (default: `800`)
- `RAG_CHUNK_OVERLAP` (default: `120`)
- `RAG_EMBEDDING_MODEL_NAME` (default: `sentence-transformers/all-MiniLM-L6-v2`)
- `RAG_OLLAMA_API_URL` (default: `http://localhost:11434`)
- `RAG_OLLAMA_MODEL` (default: `llama3`)
- `RAG_OLLAMA_TEMPERATURE` (default: `0.2`)
- `RAG_OLLAMA_MAX_TOKENS` (default: `512`)
- `RAG_RETRIEVER_TOP_K` (default: `4`)
- `RAG_RETRIEVER_SCORE_THRESHOLD` (default: `0.45`; available in settings, not applied by the current retriever)

Example exports:
```bash
export RAG_DATA_DIR=data
export RAG_VECTOR_STORE_PATH=data/faiss_index
export RAG_OLLAMA_API_URL=http://localhost:11434
export RAG_OLLAMA_MODEL=llama3
```

## Ollama Setup
Install Ollama (https://ollama.com) locally and pull the model:
```bash
ollama pull llama3
ollama serve   # if not already running
```
The client targets the local endpoint `http://localhost:11434/api/generate`.

## Running the FastAPI Server
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
API docs: http://localhost:8000/docs

## Ingesting Documents (PDF)
POST a PDF file to ingest and index:
```bash
curl -X POST "http://localhost:8000/ingest/upload" \
  -F "file=@/path/to/doc.pdf"
```
- The service chunks the PDF, generates embeddings, and updates the FAISS index on disk under [data/faiss_index/](data/faiss_index).
- To rebuild from an existing directory, POST JSON to `/ingest` with an optional `data_dir` overriding the default data directory.

## Querying the System
Submit a natural-language question; the agent retrieves relevant chunks and generates a retrieval-grounded answer, explicitly refusing when no supporting evidence exists:
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the key findings in the report?"}'
```
Responses include citations. If the agent determines retrieval is not applicable (e.g., small talk), it returns a refusal message; if no supporting evidence is found, the API responds with HTTP 404.

## Streamlit UI (Optional)
Run the UI for interactive upload and query:
```bash
streamlit run ui.py
```

## Configuration
- Centralized in [config/settings.py](config/settings.py).
- Override via environment variables (above) to adjust host/port, model names, and data/index paths.
- Defaults favor local/offline execution.

## Error Handling and Design Decisions
- Defensive retrieval: agent refuses non-grounded queries and returns 404 when no supporting evidence is found.
- Index persistence: FAISS index stored on disk; safe to restart without re-ingestion.
- Logging: structured logs configured in [utils/logging.py](utils/logging.py).
- Input validation on upload and query endpoints; PDF parsing errors return clear HTTP responses.

## Future Improvements / Roadmap
- Add authentication/authorization for ingestion and query endpoints.
- Support additional file types (DOCX, HTML).
- Streaming responses from the LLM.
- Background ingestion queue for large batches.
- Evaluation harness for retrieval quality and grounding.
- Containerized deployment with GPU-aware settings.

## Testing
```bash
pytest
```

## License
MIT License.
