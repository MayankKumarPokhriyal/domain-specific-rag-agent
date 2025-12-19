"""Agent-style routing and decision logic."""
import re
from dataclasses import dataclass
from typing import List, Tuple

from langchain.schema import Document

from retrieval.retriever import VectorRetriever
from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AgentDecision:
    """Decision about whether retrieval is needed."""

    require_retrieval: bool
    reason: str


class AgentController:
    """Lightweight agent that guards retrieval and refusals."""

    def __init__(self, retriever: VectorRetriever) -> None:
        self.retriever = retriever

    @staticmethod
    def _is_small_talk(query: str) -> bool:
        """
        Detect trivial conversational queries that do not require retrieval.
        """
        normalized = re.sub(r"[^\w\s]", "", query.lower()).strip()

        trivial_phrases = {
            "hello",
            "hi",
            "hey",
            "how are you",
            "who are you",
            "what is your name",
            "thank you",
            "thanks",
            "good morning",
            "good evening",
        }

        return normalized in trivial_phrases

    def decide(self, query: str) -> AgentDecision:
        """
        Decide whether the query requires document retrieval.
        """
        if not query or not query.strip():
            return AgentDecision(
                require_retrieval=False,
                reason="Empty or whitespace-only query",
            )

        if self._is_small_talk(query):
            return AgentDecision(
                require_retrieval=False,
                reason="Conversational or small-talk query",
            )

        return AgentDecision(
            require_retrieval=True,
            reason="Document-grounded information request",
        )

    def retrieve(self, query: str) -> Tuple[AgentDecision, List[Tuple[Document, float]]]:
        decision = self.decide(query)
        if not decision.require_retrieval:
            logger.info("Skipping retrieval: %s", decision.reason)
            return decision, []

        results = self.retriever.retrieve(query)
        return decision, results
