"""Chunking utilities for documents."""
from typing import Iterable, List

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


def chunk_documents(
    documents: Iterable[Document],
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> List[Document]:
    """Split documents into overlapping chunks with metadata preserved."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )
    chunked_docs: List[Document] = []
    for idx, chunk in enumerate(splitter.split_documents(list(documents))):
        chunk.metadata["chunk_id"] = idx
        chunked_docs.append(chunk)
    return chunked_docs
