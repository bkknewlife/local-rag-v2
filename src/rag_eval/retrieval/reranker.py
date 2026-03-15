"""Optional cross-encoder reranking of retrieved documents.

This module provides a placeholder reranker that can be swapped for a
real cross-encoder (e.g. sentence-transformers cross-encoder model)
when one is available.  By default it simply passes documents through
unchanged, sorted by their original retrieval score.
"""

from __future__ import annotations

import logging
from rag_eval.retrieval.store import RetrievedDoc

log = logging.getLogger(__name__)


def rerank(
    query: str,
    docs: list[RetrievedDoc],
    top_k: int | None = None,
) -> list[RetrievedDoc]:
    """Rerank *docs* for *query*.  Default: passthrough sorted by score."""
    ranked = sorted(docs, key=lambda d: d.score, reverse=True)
    if top_k:
        ranked = ranked[:top_k]
    return ranked
