"""ChromaDB retriever wrapper for the RAG pipeline.

Uses the same embedding backend as ingestion (via ``get_embedder``) and
resolves the backend-tagged collection name so queries always hit the
index that was built with matching vectors.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from rag_eval.config import Settings
from rag_eval.ingest.embedder import Embedder, get_embedder
from rag_eval.ingest.indexer import get_chroma_client, get_or_create_collection

log = logging.getLogger(__name__)


@dataclass
class RetrievedDoc:
    text: str
    metadata: dict
    score: float  # cosine similarity (1 - distance)


class ChromaRetriever:
    """Thin wrapper around ChromaDB that embeds a query and returns top-k docs.

    The embedder is created with ``embedding_fallback=False`` to prevent
    dimension mismatches.  If the collection was built with 768-dim Ollama
    vectors, querying with 384-dim HuggingFace vectors would always fail.
    The retriever must use the same backend that was used during ingestion.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = (settings or Settings()).model_copy(
            update={"embedding_fallback": False}
        )
        self._embedder: Embedder = get_embedder(self.settings)
        self._client = get_chroma_client(self.settings)
        self._collection = get_or_create_collection(
            self._client, self.settings, self._embedder
        )
        log.info("Retriever using collection %r (%d docs, fallback=off)",
                 self._collection.name, self._collection.count())

    @property
    def doc_count(self) -> int:
        return self._collection.count()

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedDoc]:
        top_k = top_k or self.settings.retrieval_top_k
        q_vec = self._embedder.embed_query(query)
        results = self._collection.query(
            query_embeddings=[q_vec],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        docs: list[RetrievedDoc] = []
        if not results["documents"] or not results["documents"][0]:
            return docs

        for text, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            docs.append(RetrievedDoc(
                text=text,
                metadata=meta or {},
                score=round(1.0 - dist, 4),
            ))
        return docs
