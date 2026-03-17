"""Central configuration for the RAG evaluation platform."""

from __future__ import annotations

from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """All tunables for the RAG eval platform.

    Values are read from environment variables (prefixed ``RAG_``) with
    fallbacks to the defaults below.
    """

    model_config = {"env_prefix": "RAG_"}

    # -- Ollama --
    ollama_base_url: str = "http://localhost:11434"

    # -- Embeddings --
    # Backend: "ollama" (GPU via Ollama) or "huggingface" (CPU via sentence-transformers)
    embedding_backend: str = "ollama"
    # Model name — interpreted according to the backend:
    #   ollama:      "nomic-embed-text" (768-dim, 8192 tokens, GPU)
    #   huggingface: "sentence-transformers/all-MiniLM-L6-v2" (384-dim, 256 tokens, CPU)
    embedding_model: str = "nomic-embed-text"
    # Fallback: if True, automatically try huggingface when ollama embedding fails
    embedding_fallback: bool = True
    # HuggingFace fallback model (used when backend=ollama and fallback kicks in)
    embedding_fallback_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Models to evaluate during a sweep
    eval_models: list[str] = Field(
        default_factory=lambda: [
            "llama3",
            "mistral",
            "phi3",
        ]
    )
    # Larger model used as the LLM-as-judge
    judge_model: str = "qwen3-coder:30b"

    # -- ChromaDB --
    chroma_dir: Path = _PROJECT_ROOT / "data" / "chroma_db"
    chroma_collection: str = "rag_eval"
    chroma_distance: str = "cosine"

    # -- Ingestion --
    chunk_size: int = 512
    chunk_overlap: int = 64
    embed_batch_size: int = 32
    hf_cache_dir: str | None = None  # None → default HF cache

    # -- Retrieval --
    retrieval_top_k: int = 5

    # -- LangGraph --
    max_retries: int = 1

    # -- Evaluation --
    results_dir: Path = _PROJECT_ROOT / "data" / "results"

    # -- Web Search (SearXNG) --
    searxng_base_url: str = "http://localhost:8888"
    web_search_enabled: bool = False
    web_search_max_results: int = 3

    # -- API --
    api_host: str = "0.0.0.0"
    api_port: int = 8000


_settings_singleton: Settings | None = None


def get_settings() -> Settings:
    global _settings_singleton
    if _settings_singleton is None:
        _settings_singleton = Settings()
    return _settings_singleton
