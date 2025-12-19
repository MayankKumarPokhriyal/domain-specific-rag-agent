"""Retrieval layer using FAISS with defensive guards."""

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from config.settings import get_settings
from utils.logging import get_logger

logger = get_logger(__name__)


class VectorRetriever:
    """
    Wrapper around a FAISS vector store.

    IMPORTANT:
    FAISS returns distance scores (lower = more similar).
    """

    def __init__(
        self,
        index_path: Optional[Path] = None,
        *,
        top_k: Optional[int] = None,
        max_distance: Optional[float] = None,
    ) -> None:
        settings = get_settings()

        self.index_path = index_path or settings.vector_store_path
        self.top_k = top_k or settings.retriever_top_k
        self.max_distance = max_distance or settings.retriever_score_threshold

        self._embeddings = self._create_embeddings(
            settings.embedding_model_name,
            settings.embedding_batch_size,
        )
        self._store: Optional[FAISS] = None

    @staticmethod
    def _create_embeddings(
        model_name: str,
        batch_size: int,
    ) -> HuggingFaceEmbeddings:
        """Create embedding model instance."""
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"batch_size": batch_size},
        )

    def _load_store(self) -> None:
        """Lazy-load the FAISS index from disk."""
        if self._store is not None:
            return

        if not self.index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {self.index_path}. "
                "Run the /ingest endpoint first."
            )

        logger.info("Loading FAISS index from %s", self.index_path)
        self._store = FAISS.load_local(
            self.index_path.as_posix(),
            self._embeddings,
            allow_dangerous_deserialization=True,
        )

    def retrieve(self, query: str) -> List[Tuple[Document, float]]:
        """
        Retrieve relevant documents with distance scores.

        Returns:
            List of (Document, distance_score) tuples where
            distance_score <= max_distance.
        """
        if not query or not query.strip():
            logger.warning("Empty query received; skipping retrieval.")
            return []

        self._load_store()
        assert self._store is not None

        raw_results = self._store.similarity_search_with_score(
            query,
            k=self.top_k,
        )

        filtered_results: List[Tuple[Document, float]] = [
            (doc, score)
            for doc, score in raw_results
            if score is not None and score <= self.max_distance
        ]

        logger.info(
            "Retrieved %d/%d chunks under max distance %.3f",
            len(filtered_results),
            len(raw_results),
            self.max_distance,
        )

        return filtered_results

    def as_store(self) -> FAISS:
        """Expose the underlying FAISS store if needed."""
        self._load_store()
        assert self._store is not None
        return self._store


def format_citations(docs: Sequence[Document]) -> str:
    """Format citations for grounded answers."""
    citations: List[str] = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        chunk_id = doc.metadata.get("chunk_id", "?")
        citations.append(f"[{source} - chunk {chunk_id}]")
    return " ".join(citations)