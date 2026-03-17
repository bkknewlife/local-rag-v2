"""LangGraph state definition for the self-correcting RAG pipeline."""

from __future__ import annotations

from typing import TypedDict


class GraphState(TypedDict, total=False):
    """Shared state flowing through every node of the RAG graph.

    Fields
    ------
    question : str
        The current query (may be rewritten during retries).
    original_question : str
        The verbatim question as submitted — never overwritten.
    documents : list[dict]
        Retrieved context documents (``{"text": ..., "score": ..., "metadata": ...}``).
    generation : str
        The latest LLM answer.
    model_name : str
        Ollama model currently being evaluated.
    retries : int
        Number of query-rewrite / re-generation cycles executed so far.
    max_retries : int
        Upper bound to prevent infinite loops.
    web_search_enabled : bool
        Whether to augment ChromaDB retrieval with SearXNG web results.
    latency : dict
        Breakdown of timing per phase (``{"retrieve_s": ..., "generate_s": ..., ...}``).
    gpu_snapshot : dict
        Memory / utilisation snapshot captured during generation.
    """

    question: str
    original_question: str
    documents: list[dict]
    generation: str
    model_name: str
    retries: int
    max_retries: int
    web_search_enabled: bool
    latency: dict
    gpu_snapshot: dict
