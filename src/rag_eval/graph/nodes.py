"""All LangGraph node functions for the self-correcting RAG pipeline."""

from __future__ import annotations

import logging
import time

import httpx

from rag_eval.config import Settings, get_settings
from rag_eval.graph.state import GraphState
from rag_eval.retrieval.store import ChromaRetriever

log = logging.getLogger(__name__)

_retriever: ChromaRetriever | None = None
_settings: Settings | None = None


def _get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = get_settings()
    return _settings


def _get_retriever() -> ChromaRetriever:
    global _retriever
    if _retriever is None:
        _retriever = ChromaRetriever(_get_settings())
    return _retriever


def _ollama_generate(model: str, prompt: str, system: str = "") -> dict:
    """Call Ollama chat endpoint and return the full response dict."""
    settings = _get_settings()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    resp = httpx.post(
        f"{settings.ollama_base_url}/api/chat",
        json={"model": model, "messages": messages, "stream": False},
        timeout=600.0,
    )
    resp.raise_for_status()
    return resp.json()


SYSTEM_RAG = (
    "You are a precise Q&A assistant. "
    "Answer ONLY using the provided context. "
    "If the answer is not in the context, say 'I don't know'. "
    "Do not add information from outside the context."
)


# ── Node: retrieve ──────────────────────────────────────────────────

def retrieve(state: GraphState) -> GraphState:
    """Embed the question and fetch top-k docs from ChromaDB.

    On DGX Spark with OLLAMA_MAX_LOADED_MODELS=1, a prior generate() call
    may have swapped in the chat model.  The OllamaBackend retry logic
    handles the swap back to the embedding model transparently.
    """
    t0 = time.perf_counter()
    retriever = _get_retriever()
    docs = retriever.retrieve(state["question"])
    elapsed = time.perf_counter() - t0

    scores = [f"{d.score:.3f}" for d in docs]
    log.info("[retrieve] %d docs in %.3fs, scores=[%s], question=%r",
             len(docs), elapsed, ", ".join(scores), state["question"][:80])

    latency = dict(state.get("latency") or {})
    latency["retrieve_s"] = round(elapsed, 4)

    return {
        **state,
        "documents": [
            {"text": d.text, "score": d.score, "metadata": d.metadata}
            for d in docs
        ],
        "latency": latency,
    }


# ── Node: web_search ────────────────────────────────────────────────

def web_search(state: GraphState) -> GraphState:
    """Query SearXNG and append web results to the existing documents.

    Only the question text is sent to SearXNG — no dataset content,
    embeddings, or model answers ever leave the machine.
    """
    settings = _get_settings()
    question = state["question"]
    max_results = settings.web_search_max_results

    t0 = time.perf_counter()
    web_docs: list[dict] = []
    try:
        resp = httpx.get(
            f"{settings.searxng_base_url}/search",
            params={
                "q": question,
                "format": "json",
                "categories": "general",
                "language": "en",
            },
            timeout=30.0,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])[:max_results]
        for r in results:
            content = r.get("content", "").strip()
            title = r.get("title", "").strip()
            if not content:
                continue
            text = f"{title}\n{content}" if title else content
            web_docs.append({
                "text": text,
                "score": 0.0,
                "metadata": {"source": "web", "url": r.get("url", "")},
            })
        log.info("SearXNG returned %d results for: %s", len(web_docs), question[:60])
    except Exception as exc:
        log.warning("SearXNG search failed (continuing without web results): %s", exc)

    elapsed = time.perf_counter() - t0
    latency = dict(state.get("latency") or {})
    latency["web_search_s"] = round(elapsed, 4)

    existing_docs = list(state.get("documents") or [])
    existing_docs.extend(web_docs)

    return {**state, "documents": existing_docs, "latency": latency}


# ── Node: grade_documents ──────────────────────────────────────────

def grade_documents(state: GraphState) -> GraphState:
    """Use the evaluated model to grade each retrieved doc as relevant or not."""
    model = state["model_name"]
    question = state["question"]
    graded: list[dict] = []

    for doc in state.get("documents", []):
        prompt = (
            f"Question: {question}\n\n"
            f"Document:\n{doc['text']}\n\n"
            "Is this document relevant to the question? "
            "Respond with ONLY the word 'yes' or 'no'."
        )
        try:
            data = _ollama_generate(model, prompt)
            answer = data["message"]["content"].strip().lower()
            if "yes" in answer:
                graded.append(doc)
        except Exception:
            graded.append(doc)  # keep on error

    if not graded:
        graded = state.get("documents", [])[:2]
        log.info("[grade_documents] No docs graded relevant, keeping top 2 as fallback")

    log.info("[grade_documents] %d/%d docs passed relevance filter",
             len(graded), len(state.get("documents", [])))
    return {**state, "documents": graded}


# ── Node: generate ──────────────────────────────────────────────────

def generate(state: GraphState) -> GraphState:
    """Generate an answer using the evaluated model + retrieved context."""
    model = state["model_name"]
    ctx = "\n\n---\n\n".join(d["text"] for d in state.get("documents", []))
    prompt = f"Context:\n{ctx}\n\nQuestion: {state['question']}\nAnswer:"

    t0 = time.perf_counter()
    data = _ollama_generate(model, prompt, system=SYSTEM_RAG)
    elapsed = time.perf_counter() - t0

    latency = dict(state.get("latency") or {})
    latency["generate_s"] = round(elapsed, 4)
    latency["prompt_tokens"] = data.get("prompt_eval_count", 0)
    latency["completion_tokens"] = data.get("eval_count", 0)
    latency["tokens_per_sec"] = round(
        data.get("eval_count", 0) / max(elapsed, 0.001), 1
    )

    answer = data["message"]["content"]
    log.info("[generate] model=%s, %.1fs, %d prompt_tok, %d compl_tok, %.1f tok/s, answer_len=%d",
             model, elapsed, data.get("prompt_eval_count", 0),
             data.get("eval_count", 0), latency["tokens_per_sec"], len(answer))
    log.debug("[generate] answer=%r", answer[:300])

    return {
        **state,
        "generation": answer,
        "latency": latency,
    }


# ── Node: check_hallucination ──────────────────────────────────────

def check_hallucination(state: GraphState) -> GraphState:
    """Ask the model whether its own answer is grounded in the context."""
    model = state["model_name"]
    ctx = "\n\n---\n\n".join(d["text"] for d in state.get("documents", []))
    prompt = (
        f"Context:\n{ctx}\n\n"
        f"Answer:\n{state['generation']}\n\n"
        "Is every claim in the answer fully supported by the context? "
        "Respond with ONLY the word 'yes' or 'no'."
    )
    try:
        data = _ollama_generate(model, prompt)
        verdict = data["message"]["content"].strip().lower()
        grounded = "yes" in verdict
    except Exception as exc:
        log.warning("[check_hallucination] model=%s failed: %s — assuming grounded", model, exc)
        grounded = True

    retries = state.get("retries", 0)
    if not grounded:
        retries += 1
    log.info("[check_hallucination] model=%s grounded=%s retries=%d", model, grounded, retries)

    latency = dict(state.get("latency") or {})
    latency["hallucination_grounded"] = grounded
    return {**state, "latency": latency, "retries": retries}


# ── Node: check_usefulness ─────────────────────────────────────────

def check_usefulness(state: GraphState) -> GraphState:
    """Ask the model whether the answer actually addresses the question."""
    model = state["model_name"]
    prompt = (
        f"Question: {state['question']}\n\n"
        f"Answer: {state['generation']}\n\n"
        "Does this answer address the question? "
        "Respond with ONLY the word 'yes' or 'no'."
    )
    try:
        data = _ollama_generate(model, prompt)
        verdict = data["message"]["content"].strip().lower()
        useful = "yes" in verdict
    except Exception as exc:
        log.warning("[check_usefulness] model=%s failed: %s — assuming useful", model, exc)
        useful = True

    retries = state.get("retries", 0)
    if not useful:
        retries += 1
    log.info("[check_usefulness] model=%s useful=%s retries=%d", model, useful, retries)

    latency = dict(state.get("latency") or {})
    latency["answer_useful"] = useful
    return {**state, "latency": latency, "retries": retries}


# ── Node: rewrite_query ────────────────────────────────────────────

def rewrite_query(state: GraphState) -> GraphState:
    """Rewrite the question to improve retrieval on the next attempt.

    Always references ``original_question`` (the verbatim user query)
    to prevent multi-hop drift where successive rewrites stray from
    the actual intent.
    """
    model = state["model_name"]
    original = state.get("original_question") or state["question"]
    prompt = (
        f"The following question did not get a good answer from document retrieval.\n"
        f"Original question: {original}\n\n"
        "Rewrite this question to be more specific and improve search results. "
        "Return ONLY the rewritten question, nothing else."
    )
    retries = state.get("retries", 0) + 1
    try:
        data = _ollama_generate(model, prompt)
        new_q = data["message"]["content"].strip()
        if new_q:
            log.info("[rewrite_query] model=%s retry=%d: %r -> %r",
                     model, retries, original[:60], new_q[:60])
            return {**state, "question": new_q, "retries": retries}
    except Exception as exc:
        log.warning("[rewrite_query] model=%s failed: %s", model, exc)
    return {**state, "retries": retries}
