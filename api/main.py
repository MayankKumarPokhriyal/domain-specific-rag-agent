"""FastAPI surface for ingestion and query."""
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agent.controller import AgentController
from config.settings import get_settings
from generation.generator import AnswerGenerator
from generation.llm_client import OllamaClient
from ingestion.indexer import build_and_persist_index
from retrieval.retriever import VectorRetriever
from utils.logging import get_logger

logger = get_logger(__name__)
app = FastAPI(title="Domain-Specific RAG Agent")


class IngestRequest(BaseModel):
    data_dir: Optional[Path] = None


class QueryRequest(BaseModel):
    query: str


settings = get_settings()
retriever = VectorRetriever()
agent = AgentController(retriever)
llm_client = OllamaClient(
    api_url=settings.ollama_api_url,
    model=settings.ollama_model,
    temperature=settings.ollama_temperature,
    max_tokens=settings.ollama_max_tokens,
)
generator = AnswerGenerator(llm_client)


@app.post("/ingest")
def ingest(payload: IngestRequest) -> Dict[str, Any]:
    """Trigger ingestion and index creation."""

    index_path = build_and_persist_index(payload.data_dir)
    return {"index_path": str(index_path)}


@app.post("/query")
def query(payload: QueryRequest) -> Dict[str, Any]:
    """Handle user queries with agentic control."""

    decision, retrieved = agent.retrieve(payload.query)
    if not decision.require_retrieval:
        return {
            "answer": "I am designed to answer grounded questions. Please ask about the documents.",
            "reason": decision.reason,
            "citations": [],
        }

    if not retrieved:
        raise HTTPException(
            status_code=404,
            detail="No supporting evidence found for this query.",
        )

    answer = generator.generate(payload.query, retrieved)
    citations: List[str] = [
        doc.metadata.get("source", "unknown") for doc, _score in retrieved
    ]
    return {"answer": answer, "citations": citations}
