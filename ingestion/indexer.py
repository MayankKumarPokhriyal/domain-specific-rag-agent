"""End-to-end ingestion pipeline for FAISS index creation."""
from pathlib import Path
from typing import Optional

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from config.settings import get_settings
from ingestion.chunker import chunk_documents
from ingestion.loader import load_documents
from utils.logging import get_logger

logger = get_logger(__name__)


def _embedding_model(model_name: str, batch_size: int) -> HuggingFaceEmbeddings:
    """Instantiate the embedding model."""

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"batch_size": batch_size},
    )


def build_and_persist_index(data_dir: Optional[Path] = None) -> Path:
    """Load documents, chunk, embed, and persist FAISS index.

    Args:
        data_dir: Optional override for the data directory.
    Returns:
        Path to the persisted FAISS index directory.
    """

    settings = get_settings()
    source_dir = data_dir or settings.data_dir
    documents = load_documents(source_dir)
    if not documents:
        raise ValueError("No documents loaded; cannot build index.")

    chunked_docs = chunk_documents(
        documents,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    if not chunked_docs:
        raise ValueError("Chunking produced no documents; check splitter settings.")

    logger.info("Creating embeddings with %s", settings.embedding_model_name)
    embeddings = _embedding_model(settings.embedding_model_name, settings.embedding_batch_size)

    logger.info("Building FAISS index with %d chunks", len(chunked_docs))
    vector_store = FAISS.from_documents(chunked_docs, embeddings)

    index_path = settings.vector_store_path
    index_path.parent.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(index_path.as_posix())
    logger.info("Persisted FAISS index to %s", index_path)
    return index_path
