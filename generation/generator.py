"""Answer generation with grounding and citations."""
from typing import Iterable, List, Tuple

from langchain.schema import Document

from generation.llm_client import OllamaClient
from retrieval.retriever import format_citations

SYSTEM_INSTRUCTIONS = (
    "You are a domain-specific assistant. Only answer using the provided context. "
    "If the context is insufficient, say you do not have enough information. "
    "Cite sources inline in square brackets."
)


def _build_prompt(query: str, docs: Iterable[Tuple[Document, float]]) -> str:
    context_blocks: List[str] = []
    for idx, (doc, _) in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        chunk_id = doc.metadata.get("chunk_id", "?")
        context_blocks.append(
            f"[Source {idx}: {source} chunk {chunk_id}]\n{doc.page_content}"
        )
    context_text = "\n\n".join(context_blocks)
    prompt = (
        f"{SYSTEM_INSTRUCTIONS}\n\n"
        f"Context:\n{context_text}\n\n"
        f"User question: {query}\n"
        "Answer:"
    )
    return prompt


class AnswerGenerator:
    """Generate grounded answers with citations."""

    def __init__(self, client: OllamaClient) -> None:
        self.client = client

    def generate(self, query: str, retrieved: List[Tuple[Document, float]]) -> str:
        if not retrieved:
            raise ValueError("No supporting evidence to generate an answer.")

        prompt = _build_prompt(query, retrieved)
        raw_answer = self.client.generate(prompt)
        citations = format_citations([doc for doc, _ in retrieved])
        confidence = f"Confidence: grounded with {len(retrieved)} source chunks."
        return f"{raw_answer}\n\nSources: {citations}\n{confidence}"
