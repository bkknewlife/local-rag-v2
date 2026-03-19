"""Collect and aggregate evaluation metrics from a single RAG run."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class EvalResult:
    """One model x question evaluation result."""

    run_id: str
    model: str
    judge_model: str = ""
    question: str = ""
    answer: str = ""
    contexts: list[str] = field(default_factory=list)
    context_scores: list[float] = field(default_factory=list)

    # LangGraph execution metadata
    retries: int = 0

    # Latency
    retrieve_s: float = 0.0
    generate_s: float = 0.0
    web_search_used: bool = False
    web_search_s: float = 0.0
    total_s: float = 0.0

    # Token stats
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tokens_per_sec: float = 0.0

    # GPU snapshot during generation
    gpu_mem_used_gb: float = 0.0
    gpu_util_pct: int = -1
    gpu_temp_c: int = -1
    gpu_power_w: float = -1.0

    # Judge scores
    faithfulness: float = -1.0
    faithfulness_reasoning: str = ""
    relevancy: float = -1.0
    relevancy_reasoning: str = ""
    context_precision: float = -1.0
    context_precision_reasoning: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["contexts"] = "; ".join(c[:80] for c in self.contexts)
        d["context_scores"] = ", ".join(f"{s:.3f}" for s in self.context_scores)
        return d
