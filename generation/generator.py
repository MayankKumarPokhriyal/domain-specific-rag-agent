"""Answer generation with explicit grounding and citations."""

from typing import Iterable, List, Tuple

from langchain.schema import Document

from generation.llm_client import OllamaClient
from retrieval.retriever import format_citations

SYSTEM_PROMPT = (
    "You are a domain-specific assistant.\n"
    "Answer the question using ONLY the provided context.\n"
    "If the context does not contain enough information, say so explicitly.\n"
    "Do NOT use prior knowledge.\n"
    "Cite sources inline using square brackets.\n"
)


def build_prompt(
    query: str,
    docs: Iterable[Tuple[Document, float]],
) -> str:
    """Construct a grounded prompt from retrieved documents."""

    context_blocks: List[str] = []
    for idx, (doc, _score) in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        chunk_id = doc.metadata.get("chunk_id", "?")
        context_blocks.append(
            f"[Source {idx}: {source}, chunk {chunk_id}]\n"
            f"{doc.page_content}"
        )

    context_text = "\n\n".join(context_blocks)

    prompt = (
        f"{SYSTEM_PROMPT}\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {query}\n"
        f"Answer:"
    )

    return prompt


class AnswerGenerator:
    """Generate grounded answers using retrieved document context."""

    def __init__(self, client: OllamaClient) -> None:
        self.client = client

    def generate(
        self,
        query: str,
        retrieved: List[Tuple[Document, float]],
    ) -> str:
        """
        Generate an answer strictly grounded in retrieved documents.
        """
        if not retrieved:
            raise ValueError("No retrieved context available for answer generation.")

        prompt = build_prompt(query, retrieved)
        answer_text = self.client.generate(prompt)

        citations = format_citations([doc for doc, _ in retrieved])
        confidence_note = (
            f"Confidence: grounded using {len(retrieved)} document chunk(s)."
        )

        return f"{answer_text}\n\nSources: {citations}\n{confidence_note}"