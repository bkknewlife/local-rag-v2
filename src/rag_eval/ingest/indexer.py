"""Manage ChromaDB collections and batch-upsert embedded chunks.

The collection name is derived from ``Settings.chroma_collection`` plus the
embedding backend tag, ensuring that vectors from different embedding models
are never mixed (they have different dimensions and are incompatible).
"""

from __future__ import annotations

import logging

import chromadb
from chromadb.api.models.Collection import Collection
from tqdm import tqdm

from rag_eval.config import Settings
from rag_eval.ingest.embedder import Embedder, get_embedder

log = logging.getLogger(__name__)


def get_chroma_client(settings: Settings | None = None) -> chromadb.ClientAPI:
    settings = settings or Settings()
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(settings.chroma_dir))


def collection_name_for(settings: Settings, embedder: Embedder | None = None) -> str:
    """Build a collection name that encodes the embedding backend.

    Example: ``"rag_eval__ollama-nomic-embed-text"`` or
             ``"rag_eval__hf-all-MiniLM-L6-v2"``
    """
    if embedder is None:
        embedder = get_embedder(settings)
    tag = embedder.backend_tag.replace("/", "-").replace(":", "-")
    return f"{settings.chroma_collection}__{tag}"


def get_or_create_collection(
    client: chromadb.ClientAPI | None = None,
    settings: Settings | None = None,
    embedder: Embedder | None = None,
) -> Collection:
    settings = settings or Settings()
    if client is None:
        client = get_chroma_client(settings)
    name = collection_name_for(settings, embedder)
    log.debug("ChromaDB collection: %s", name)
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": settings.chroma_distance},
    )


def index_chunks(
    chunks: list[dict],
    *,
    settings: Settings | None = None,
) -> int:
    """Embed and upsert *chunks* into ChromaDB. Returns number indexed."""
    settings = settings or Settings()
    embedder = get_embedder(settings)
    collection = get_or_create_collection(settings=settings, embedder=embedder)

    import time

    batch_size = settings.embed_batch_size
    total = 0
    n_batches = (len(chunks) + batch_size - 1) // batch_size

    log.info("Indexing %d chunks in %d batches (batch_size=%d)",
             len(chunks), n_batches, batch_size)

    for start in tqdm(
        range(0, len(chunks), batch_size),
        desc="Embedding & indexing",
        unit="batch",
    ):
        batch = chunks[start : start + batch_size]
        texts = [c["text"] for c in batch]
        ids = [c["id"] for c in batch]
        metas = [c["metadata"] for c in batch]

        t0 = time.perf_counter()
        vecs = embedder.embed_documents(texts)
        embed_s = time.perf_counter() - t0

        collection.upsert(
            documents=texts,
            embeddings=vecs,
            ids=ids,
            metadatas=metas,
        )
        total += len(batch)
        chunks_per_sec = len(batch) / max(embed_s, 0.001)
        log.debug("Batch %d: %d chunks in %.1fs (%.0f chunks/sec)",
                  start // batch_size + 1, len(batch), embed_s, chunks_per_sec)

    log.info("Indexed %d chunks into collection %r (total in collection: %d)",
             total, collection.name, collection.count())
    return total
