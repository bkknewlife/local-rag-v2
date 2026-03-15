"""Conditional edge functions for the LangGraph RAG pipeline."""

from __future__ import annotations

from rag_eval.graph.state import GraphState


def route_after_grading(state: GraphState) -> str:
    """After grading documents, decide whether to generate or rewrite.

    If fewer than half the original docs survived grading AND we haven't
    exceeded retries, rewrite the query.
    """
    docs = state.get("documents", [])
    retries = state.get("retries", 0)
    max_retries = state.get("max_retries", 3)

    if len(docs) == 0 and retries < max_retries:
        return "rewrite"
    return "generate"


def route_after_hallucination(state: GraphState) -> str:
    """After hallucination check, decide whether the answer is grounded."""
    latency = state.get("latency", {})
    grounded = latency.get("hallucination_grounded", True)
    retries = state.get("retries", 0)
    max_retries = state.get("max_retries", 3)

    if not grounded and retries < max_retries:
        return "regenerate"
    return "check_useful"


def route_after_usefulness(state: GraphState) -> str:
    """After usefulness check, decide whether to accept or rewrite."""
    latency = state.get("latency", {})
    useful = latency.get("answer_useful", True)
    retries = state.get("retries", 0)
    max_retries = state.get("max_retries", 3)

    if not useful and retries < max_retries:
        return "rewrite"
    return "end"
