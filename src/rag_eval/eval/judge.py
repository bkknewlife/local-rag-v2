"""LLM-as-judge: fully local quality scoring via Ollama.

Three metrics are evaluated by prompting the judge model:
  - Faithfulness  (answer grounded in context)
  - Answer Relevancy  (answer addresses the question)
  - Context Precision  (retrieved docs are relevant to the question)

Each returns a float score 0.0-1.0 plus a short reasoning string.
"""

from __future__ import annotations

import json
import logging
import re

import httpx

from rag_eval.config import Settings, get_settings

log = logging.getLogger(__name__)


def _ask_judge(model: str, prompt: str, settings: Settings, *, metric: str = "") -> dict:
    """Send a structured-scoring prompt to the judge model."""
    import time
    label = f"[judge:{metric}]" if metric else "[judge]"
    log.debug("%s Sending prompt to %s (%d chars)", label, model, len(prompt))

    t0 = time.perf_counter()
    resp = httpx.post(
        f"{settings.ollama_base_url}/api/chat",
        json={
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a strict evaluation judge. "
                        "Always respond with valid JSON matching: "
                        '{"score": <float 0.0-1.0>, "reasoning": "<brief explanation>"}'
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "format": "json",
        },
        timeout=600.0,
    )
    resp.raise_for_status()
    elapsed = time.perf_counter() - t0
    content = resp.json()["message"]["content"]
    log.debug("%s Raw response from %s (%.1fs): %s", label, model, elapsed, content[:500])

    try:
        parsed = json.loads(content)
        parsed["score"] = min(max(float(parsed.get("score", 0.5)), 0.0), 1.0)
        log.info("%s model=%s score=%.2f elapsed=%.1fs", label, model, parsed["score"], elapsed)
        return parsed
    except (json.JSONDecodeError, ValueError, TypeError) as parse_err:
        log.warning("%s Failed to parse JSON from %s: %s — raw: %s",
                    label, model, parse_err, content[:300])
        score_match = re.search(r"(\d+\.?\d*)", content)
        score = float(score_match.group(1)) if score_match else 0.5
        score = min(max(score, 0.0), 1.0)
        log.warning("%s Fallback score=%.2f (regex extraction)", label, score)
        return {"score": score, "reasoning": content[:200]}


def score_faithfulness(
    question: str,
    answer: str,
    contexts: list[str],
    *,
    settings: Settings | None = None,
) -> dict:
    settings = settings or get_settings()
    ctx = "\n\n---\n\n".join(contexts)
    prompt = (
        f"Question: {question}\n\n"
        f"Context:\n{ctx}\n\n"
        f"Answer:\n{answer}\n\n"
        "Evaluate faithfulness: Is every claim in the answer "
        "fully supported by the context? Score 0.0 (no support) to "
        "1.0 (perfectly grounded). Respond in JSON."
    )
    return _ask_judge(settings.judge_model, prompt, settings, metric="faithfulness")


def score_relevancy(
    question: str,
    answer: str,
    *,
    settings: Settings | None = None,
) -> dict:
    settings = settings or get_settings()
    prompt = (
        f"Question: {question}\n\n"
        f"Answer:\n{answer}\n\n"
        "Evaluate answer relevancy: Does this answer directly address "
        "the question asked? Score 0.0 (completely irrelevant) to "
        "1.0 (perfectly relevant). Respond in JSON."
    )
    return _ask_judge(settings.judge_model, prompt, settings, metric="relevancy")


def score_context_precision(
    question: str,
    contexts: list[str],
    *,
    settings: Settings | None = None,
) -> dict:
    settings = settings or get_settings()
    numbered = "\n".join(f"[{i+1}] {c[:300]}" for i, c in enumerate(contexts))
    prompt = (
        f"Question: {question}\n\n"
        f"Retrieved documents:\n{numbered}\n\n"
        "Evaluate context precision: What fraction of the retrieved "
        "documents are relevant to the question? Score 0.0 (none relevant) "
        "to 1.0 (all relevant). Respond in JSON."
    )
    return _ask_judge(settings.judge_model, prompt, settings, metric="context_precision")
