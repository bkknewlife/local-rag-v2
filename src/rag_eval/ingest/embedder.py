"""Embedding backend — supports Ollama (GPU) and HuggingFace sentence-transformers (CPU).

The backend is chosen via ``Settings.embedding_backend``:

  * ``"ollama"``      — GPU-accelerated via Ollama (default).  Uses ``nomic-embed-text``
                        (768-dim, 8192 max tokens).  Requires Ollama running.
  * ``"huggingface"`` — CPU-only via sentence-transformers.  Uses
                        ``all-MiniLM-L6-v2`` (384-dim, 256 max tokens).  No GPU needed.

When ``embedding_fallback=True`` (the default) and the primary backend is
``"ollama"``, a failed Ollama call will transparently fall back to the
HuggingFace model.

**Critical:** The two backends produce vectors of *different dimensions*
(768 vs 384).  You must use the **same** backend for ingestion and retrieval.
The collection name encodes the backend so they stay separated.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from rag_eval.config import Settings

log = logging.getLogger(__name__)


class Embedder(ABC):
    """Common interface for embedding backends."""

    @property
    @abstractmethod
    def backend_tag(self) -> str:
        """Short string identifying this backend (used in collection names)."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        ...


class OllamaBackend(Embedder):
    """GPU-accelerated embeddings via Ollama's native /api/embed endpoint.

    Sends the entire batch as a single HTTP request using Ollama's native
    multi-input ``"input": [...]`` support, avoiding the per-text overhead
    of the LangChain wrapper.

    On DGX Spark with ``OLLAMA_MAX_LOADED_MODELS=1``, Ollama must swap
    between the chat model and the embedding model.  The retry logic here
    accommodates swaps that can take 2-4 minutes by retrying with
    exponential back-off.
    """

    MAX_RETRIES = 5
    RETRY_BASE_DELAY = 15.0  # seconds

    def __init__(self, model: str, base_url: str) -> None:
        import httpx
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=httpx.Timeout(600.0, connect=60.0))
        log.info("Ollama embedding backend (direct API): model=%s url=%s", model, base_url)
        self._warmup()

    def _warmup(self) -> None:
        """Pre-load the embedding model in Ollama with an 8-hour keep_alive.

        On DGX Spark with OLLAMA_MAX_LOADED_MODELS=1, Ollama must unload
        the current chat model before loading the embedding model.  This
        can take 2-4 minutes.  We absorb that cost here so subsequent
        embed calls are fast.

        The ``keep_alive=8h`` matches the manual pre-warming curl commands
        documented in the README, preventing auto-eviction during long runs.
        """
        import time
        log.info("Warming up Ollama embedding model '%s' (this may take a few minutes "
                 "if Ollama needs to swap models)...", self._model)
        t0 = time.perf_counter()
        try:
            resp = self._client.post(
                f"{self._base_url}/api/embed",
                json={"model": self._model, "input": ["warmup"], "keep_alive": "8h"},
            )
            resp.raise_for_status()
            vecs = resp.json()["embeddings"]
            elapsed = time.perf_counter() - t0
            dim = len(vecs[0]) if vecs else "?"
            log.info("Ollama model '%s' ready — dim=%s, warmup took %.1fs",
                     self._model, dim, elapsed)
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            log.error("Ollama warmup FAILED after %.1fs: %s", elapsed, exc)
            raise

    @property
    def backend_tag(self) -> str:
        return f"ollama-{self._model}"

    def _call_embed_raw(self, inputs: list[str]) -> list[list[float]]:
        """Single attempt to call Ollama /api/embed (no retry)."""
        resp = self._client.post(
            f"{self._base_url}/api/embed",
            json={"model": self._model, "input": inputs},
        )
        resp.raise_for_status()
        data = resp.json()
        return data["embeddings"]

    def _call_embed(self, inputs: list[str]) -> list[list[float]]:
        """Call Ollama /api/embed with retry for model-swap delays."""
        import time
        last_exc: Exception | None = None
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                return self._call_embed_raw(inputs)
            except Exception as exc:
                last_exc = exc
                if attempt == self.MAX_RETRIES:
                    break
                delay = self.RETRY_BASE_DELAY * (2 ** (attempt - 1))
                log.warning(
                    "Ollama embed attempt %d/%d failed: %s  "
                    "— retrying in %.0fs (model swap may be in progress)",
                    attempt, self.MAX_RETRIES, exc, delay,
                )
                time.sleep(delay)
        raise RuntimeError(
            f"Ollama embed failed after {self.MAX_RETRIES} attempts: {last_exc}"
        ) from last_exc

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._call_embed(texts)

    def embed_query(self, text: str) -> list[float]:
        vecs = self._call_embed([text])
        return vecs[0]


class HuggingFaceBackend(Embedder):
    """CPU-based embeddings via sentence-transformers."""

    def __init__(self, model: str) -> None:
        from sentence_transformers import SentenceTransformer
        self._model_name = model
        self._st = SentenceTransformer(model)
        log.info("HuggingFace embedding backend: model=%s (dim=%d)",
                 model, self._st.get_sentence_embedding_dimension())

    @property
    def backend_tag(self) -> str:
        short = self._model_name.split("/")[-1]
        return f"hf-{short}"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vecs = self._st.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        return vecs.tolist()

    def embed_query(self, text: str) -> list[float]:
        vec = self._st.encode([text], normalize_embeddings=True)
        return vec[0].tolist()


class FallbackEmbedder(Embedder):
    """Wraps OllamaBackend and falls back to HuggingFaceBackend on failure.

    The HuggingFace fallback is **lazy** -- it is only loaded into memory
    the first time Ollama actually fails.  This avoids wasting startup time
    and RAM when Ollama is healthy.
    """

    def __init__(self, primary: OllamaBackend, fallback_model: str) -> None:
        self._primary = primary
        self._fallback_model = fallback_model
        self._fallback: HuggingFaceBackend | None = None
        self._using_fallback = False

    @property
    def backend_tag(self) -> str:
        return self._primary.backend_tag

    def _get_fallback(self) -> HuggingFaceBackend:
        if self._fallback is None:
            log.info("Lazy-loading HuggingFace fallback: %s", self._fallback_model)
            self._fallback = HuggingFaceBackend(self._fallback_model)
        return self._fallback

    def _handle_fallback(self, exc: Exception) -> None:
        if not self._using_fallback:
            log.warning(
                "Ollama embedding failed: %s. Switching to HuggingFace fallback (%s). "
                "NOTE: fallback produces 384-dim vectors — incompatible with an "
                "existing Ollama-embedded index. Only use for a fresh collection.",
                exc, self._fallback_model,
            )
            self._using_fallback = True

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self._using_fallback:
            return self._get_fallback().embed_documents(texts)
        try:
            result = self._primary.embed_documents(texts)
            return result
        except Exception as exc:
            self._handle_fallback(exc)
            return self._get_fallback().embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        if self._using_fallback:
            return self._get_fallback().embed_query(text)
        try:
            result = self._primary.embed_query(text)
            return result
        except Exception as exc:
            self._handle_fallback(exc)
            return self._get_fallback().embed_query(text)


def get_embedder(settings: Settings | None = None) -> Embedder:
    """Create the appropriate embedder based on configuration.

    The Ollama backend includes a warmup step that pre-loads the model.
    If warmup succeeds, Ollama is used for all subsequent calls (no
    fallback needed).  If warmup fails and ``embedding_fallback`` is
    enabled, the HuggingFace backend is used instead (with a warning).
    """
    settings = settings or Settings()

    if settings.embedding_backend == "huggingface":
        return HuggingFaceBackend(settings.embedding_model)

    try:
        primary = OllamaBackend(settings.embedding_model, settings.ollama_base_url)
        if settings.embedding_fallback:
            return FallbackEmbedder(primary, settings.embedding_fallback_model)
        return primary
    except Exception as exc:
        if settings.embedding_fallback:
            log.warning(
                "Ollama backend failed to initialize (%s). "
                "Using HuggingFace fallback: %s",
                exc, settings.embedding_fallback_model,
            )
            return HuggingFaceBackend(settings.embedding_fallback_model)
        raise


def embed_texts(
    texts: list[str],
    *,
    embedder: Embedder | None = None,
    settings: Settings | None = None,
) -> list[list[float]]:
    """Embed a list of texts, returning a list of float vectors."""
    if embedder is None:
        embedder = get_embedder(settings)
    return embedder.embed_documents(texts)
