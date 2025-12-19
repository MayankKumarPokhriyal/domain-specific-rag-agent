"""Lightweight tests for agent decision logic."""

from agent.controller import AgentController, AgentDecision


class DummyRetriever:
    """Minimal retriever stub for agent tests."""

    def retrieve(self, query: str):
        return []


def test_empty_query():
    agent = AgentController(DummyRetriever())
    decision = agent.decide("")
    assert decision == AgentDecision(
        require_retrieval=False,
        reason="Empty or whitespace-only query",
    )


def test_small_talk_detection():
    agent = AgentController(DummyRetriever())
    decision = agent.decide("hello")
    assert decision == AgentDecision(
        require_retrieval=False,
        reason="Conversational or small-talk query",
    )


def test_information_query():
    agent = AgentController(DummyRetriever())
    decision = agent.decide("Explain the theory of relativity")
    assert decision == AgentDecision(
        require_retrieval=True,
        reason="Document-grounded information request",
    )