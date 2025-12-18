"""Lightweight tests for agent decision logic."""
from agent.controller import AgentController, AgentDecision
from retrieval.retriever import VectorRetriever


class DummyRetriever(VectorRetriever):
    def __init__(self) -> None:  # type: ignore[no-untyped-def]
        pass

    def retrieve(self, query: str):  # type: ignore[override]
        return []


def test_small_talk_detection():
    agent = AgentController(DummyRetriever())
    decision = agent.decide("hello there")
    assert decision == AgentDecision(require_retrieval=False, reason="Small talk")


def test_information_query():
    agent = AgentController(DummyRetriever())
    decision = agent.decide("Explain the theory of relativity")
    assert decision.require_retrieval is True
