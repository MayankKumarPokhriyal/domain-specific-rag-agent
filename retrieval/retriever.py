"""Retrieval layer using FAISS."""
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from config.settings import get_settings
from utils.logging import get_logger

logger = get_logger(__name__)


class VectorRetriever:
    """Wrapper around FAISS for defensive retrieval."""

    def __init__(
        self,
        index_path: Optional[Path] = None,
        *,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> None:
        settings = get_settings()
        self.index_path = index_path or settings.vector_store_path
        self.top_k = top_k or settings.retriever_top_k
        self.score_threshold = score_threshold or settings.retriever_score_threshold
        self._embeddings = self._embedding_model(
            settings.embedding_model_name, settings.embedding_batch_size
        )
        self._store: Optional[FAISS] = None

    @staticmethod
    def _embedding_model(model_name: str, batch_size: int) -> HuggingFaceEmbeddings:
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"batch_size": batch_size},
        )

    def _load_store(self) -> None:
        if self._store is not None:
            return
        if not self.index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {self.index_path}. Run ingestion first."
            )
        logger.info("Loading FAISS index from %s", self.index_path)
        self._store = FAISS.load_local(
            self.index_path.as_posix(),
            self._embeddings,
            allow_dangerous_deserialization=True,
        )

    def retrieve(self, query: str) -> List[Tuple[Document, float]]:
        """Retrieve relevant documents with similarity scores."""

        if not query.strip():
            logger.warning("Empty query received; returning no results.")
            return []

        self._load_store()
        assert self._store is not None
        results = self._store.similarity_search_with_score(query, k=self.top_k)
        filtered = [
            (doc, score)
            for doc, score in results
            if score is not None and score <= self.score_threshold
        ]
        logger.info(
            "Retrieved %d/%d documents under threshold %.2f",
            len(filtered),
            len(results),
            self.score_threshold,
        )
        return filtered

    def as_retriever(self) -> FAISS:
        """Expose the underlying store for advanced flows."""

        self._load_store()
        assert self._store is not None
        return self._store


def format_citations(docs: Sequence[Document]) -> str:
    """Format source citations for final answers."""

    citations = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        chunk_id = doc.metadata.get("chunk_id", "?")
        citations.append(f"[{source} - chunk {chunk_id}]")
    return " ".join(citations)
