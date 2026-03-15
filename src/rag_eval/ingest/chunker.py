"""Split documents into smaller chunks for embedding."""

from __future__ import annotations

import logging
from typing import Iterable

from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_eval.config import Settings

log = logging.getLogger(__name__)


def chunk_documents(
    docs: Iterable[dict],
    *,
    settings: Settings | None = None,
) -> list[dict]:
    """Split each doc's ``text`` into chunks, preserving id and metadata.

    Returns a flat list of ``{"id": "<orig_id>_<chunk_idx>", "text": ..., "metadata": ...}``
    """
    settings = settings or Settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " "],
    )

    chunks: list[dict] = []
    for doc in docs:
        parts = splitter.split_text(doc["text"])
        for i, part in enumerate(parts):
            chunks.append(
                {
                    "id": f"{doc['id']}_{i}",
                    "text": part,
                    "metadata": {**doc["metadata"], "parent_id": doc["id"]},
                }
            )

    log.info("Chunked into %d pieces (size=%d, overlap=%d)",
             len(chunks), settings.chunk_size, settings.chunk_overlap)
    return chunks
