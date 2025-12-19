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


def create_embedding_model(
    model_name: str,
    batch_size: int,
) -> HuggingFaceEmbeddings:
    """Create a HuggingFace embedding model."""
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"batch_size": batch_size},
    )


def build_and_persist_index(
    data_dir: Optional[Path] = None,
) -> Path:
    """
    Load documents, chunk them, embed them, and persist a FAISS index.

    Args:
        data_dir: Optional directory containing documents to ingest.

    Returns:
        Path to the persisted FAISS index directory.
    """
    settings = get_settings()
    source_dir = data_dir or settings.data_dir

    logger.info("Starting ingestion from directory: %s", source_dir)

    documents = load_documents(source_dir)
    if not documents:
        raise ValueError(
            "No documents were loaded. "
            "Ensure the data directory exists and contains supported files."
        )

    chunked_docs = chunk_documents(
        documents,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    if not chunked_docs:
        raise ValueError("Document chunking produced no chunks.")

    logger.info(
        "Creating embeddings using model: %s",
        settings.embedding_model_name,
    )
    embeddings = create_embedding_model(
        settings.embedding_model_name,
        settings.embedding_batch_size,
    )

    logger.info(
        "Building FAISS index from %d chunks",
        len(chunked_docs),
    )
    vector_store = FAISS.from_documents(chunked_docs, embeddings)

    index_path = settings.vector_store_path
    index_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Saving FAISS index to %s", index_path)
    vector_store.save_local(index_path.as_posix())

    logger.info("Ingestion complete")
    return index_path