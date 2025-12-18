"""Document loading utilities for PDFs and text files."""
from pathlib import Path
from typing import List

from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader

from utils.logging import get_logger

logger = get_logger(__name__)


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def load_documents(data_dir: Path) -> List[Document]:
    """Load supported documents from a directory.

    Args:
        data_dir: Directory containing source documents.
    Returns:
        List of LangChain Document objects with metadata.
    """

    documents: List[Document] = []
    if not data_dir.exists():
        logger.warning("Data directory %s does not exist; nothing to ingest.", data_dir)
        return documents

    for file_path in data_dir.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            logger.debug("Skipping unsupported file %s", file_path)
            continue

        try:
            if file_path.suffix.lower() == ".pdf":
                loader = PyPDFLoader(file_path.as_posix())
                loaded_docs = loader.load()
            else:
                loader = TextLoader(file_path.as_posix(), encoding="utf-8")
                loaded_docs = loader.load()

            for doc in loaded_docs:
                doc.metadata.setdefault("source", file_path.name)
                doc.metadata.setdefault("path", str(file_path))
            documents.extend(loaded_docs)
            logger.info("Loaded %d documents from %s", len(loaded_docs), file_path.name)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load %s: %s", file_path, exc)

    return documents
