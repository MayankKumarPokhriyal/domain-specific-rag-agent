"""Chunking utilities for source documents."""

from typing import Iterable, List

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_documents(
    documents: Iterable[Document],
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> List[Document]:
    """
    Split documents into overlapping chunks while preserving metadata.

    Each chunk receives a unique `chunk_id` used later for citations.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )

    chunked_docs: List[Document] = []
    chunk_counter = 0

    for chunk in splitter.split_documents(list(documents)):
        # Ensure metadata exists
        if chunk.metadata is None:
            chunk.metadata = {}

        chunk.metadata["chunk_id"] = chunk_counter
        chunk_counter += 1

        chunked_docs.append(chunk)

    return chunked_docs