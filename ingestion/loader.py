"""Document loading utilities for PDFs and text files."""

from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader

from utils.logging import get_logger

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def load_documents(data_dir: Path) -> List[Document]:
    """
    Load supported documents from a directory.

    Supported formats: PDF, TXT, MD.
    """
    documents: List[Document] = []

    if not data_dir.exists():
        logger.warning(
            "Data directory %s does not exist. No documents loaded.",
            data_dir,
        )
        return documents

    for file_path in data_dir.rglob("*"):
        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            logger.debug("Skipping unsupported file: %s", file_path)
            continue

        try:
            if file_path.suffix.lower() == ".pdf":
                loader = PyPDFLoader(file_path.as_posix())
                loaded_docs = loader.load()
            else:
                loader = TextLoader(
                    file_path.as_posix(),
                    encoding="utf-8",
                    autodetect_encoding=True,
                )
                loaded_docs = loader.load()

            for doc in loaded_docs:
                if doc.metadata is None:
                    doc.metadata = {}
                doc.metadata.setdefault("source", file_path.name)
                doc.metadata.setdefault("path", str(file_path))

            documents.extend(loaded_docs)
            logger.info(
                "Loaded %d document(s) from %s",
                len(loaded_docs),
                file_path.name,
            )

        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Failed to load file %s due to error: %s",
                file_path,
                exc,
            )

    logger.info("Total documents loaded: %d", len(documents))
    return documents