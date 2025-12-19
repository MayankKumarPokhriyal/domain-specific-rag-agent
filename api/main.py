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
    """Request payload for ingestion."""

    data_dir: Optional[str] = None


class QueryRequest(BaseModel):
    """Request payload for querying."""

    query: str


def get_retriever() -> VectorRetriever:
    """Create a retriever (lazy init)."""
    return VectorRetriever()


def get_agent() -> AgentController:
    """Create the agent controller (lazy init)."""
    return AgentController(get_retriever())


def get_generator() -> AnswerGenerator:
    """Create the answer generator (lazy init)."""
    settings = get_settings()
    llm_client = OllamaClient(
        api_url=settings.ollama_api_url,
        model=settings.ollama_model,
        temperature=settings.ollama_temperature,
        max_tokens=settings.ollama_max_tokens,
    )
    return AnswerGenerator(llm_client)


@app.get("/")
def health() -> Dict[str, str]:
    """Simple health endpoint."""
    return {"status": "ok"}


@app.post("/ingest")
def ingest(payload: IngestRequest) -> Dict[str, Any]:
    """Trigger ingestion and index creation."""
    data_dir = Path(payload.data_dir) if payload.data_dir else None
    index_path = build_and_persist_index(data_dir)
    return {"index_path": str(index_path)}


@app.post("/query")
def query(payload: QueryRequest) -> Dict[str, Any]:
    """Handle user queries with agentic control."""
    agent = get_agent()
    generator = get_generator()

    decision, retrieved = agent.retrieve(payload.query)

    if not decision.require_retrieval:
        return {
            "answer": "I can only answer grounded questions based on the ingested documents.",
            "reason": decision.reason,
            "citations": [],
        }

    if not retrieved:
        raise HTTPException(
            status_code=404,
            detail="No supporting evidence found for this query. Ingest documents or refine the question.",
        )

    answer = generator.generate(payload.query, retrieved)
    citations: List[str] = [
        doc.metadata.get("source", "unknown") for doc, _score in retrieved
    ]
    return {"answer": answer, "citations": citations, "reason": decision.reason}