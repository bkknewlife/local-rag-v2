"""Multi-model evaluation harness — drives the LangGraph RAG pipeline
across models and questions, collecting metrics and judge scores.

Results are persisted incrementally after each question to CSV and JSON,
so partial results survive if the process is interrupted."""

from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path

import pandas as pd
from rich import print as rprint
from rich.table import Table
from rich.console import Console

from rag_eval.config import Settings, get_settings
from rag_eval.graph.builder import build_rag_graph
from rag_eval.eval import gpu_monitor, judge
from rag_eval.eval.metrics import EvalResult

log = logging.getLogger(__name__)

DEFAULT_QUESTIONS = [
    "What is the capital of France?",
    "Explain the main cause of World War I.",
    "What are the symptoms of diabetes?",
    "Who wrote the novel '1984'?",
    "What is photosynthesis?",
]


class _ResultStore:
    """Incrementally persists results to CSV and JSON after every evaluation."""

    def __init__(self, run_id: str, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self._csv_path = out_dir / f"{run_id}.csv"
        self._json_path = out_dir / f"{run_id}.json"
        self._results: list[EvalResult] = []
        self._run_id = run_id

    @property
    def results(self) -> list[EvalResult]:
        return self._results

    def append(self, result: EvalResult) -> None:
        self._results.append(result)
        self._flush()

    def _flush(self) -> None:
        dicts = [r.to_dict() for r in self._results]
        df = pd.DataFrame(dicts)
        df.to_csv(self._csv_path, index=False)
        self._json_path.write_text(json.dumps(dicts, indent=2))

    def print_paths(self) -> None:
        n = len(self._results)
        rprint(f"\n[bold green]Results saved ({n} evaluations):[/bold green]")
        rprint(f"  CSV : {self._csv_path}")
        rprint(f"  JSON: {self._json_path}")


def run_evaluation(
    *,
    questions: list[str] | None = None,
    run_id: str | None = None,
    settings: Settings | None = None,
) -> list[EvalResult]:
    """Execute a full evaluation sweep with incremental persistence."""
    settings = settings or get_settings()
    questions = questions or DEFAULT_QUESTIONS
    run_id = run_id or uuid.uuid4().hex[:12]
    console = Console()

    rprint(f"\n[bold cyan]Starting evaluation run: {run_id}[/bold cyan]")
    rprint(f"  Models : {settings.eval_models}")
    rprint(f"  Judge  : {settings.judge_model}")
    rprint(f"  Questions: {len(questions)}")
    rprint(f"  Results : {settings.results_dir / run_id}.*")

    graph = build_rag_graph()
    store = _ResultStore(run_id, settings.results_dir)

    total_evals = len(settings.eval_models) * len(questions)
    completed = 0

    for model in settings.eval_models:
        rprint(f"\n[bold magenta]▸ Model: {model}[/bold magenta]")

        for q_idx, question in enumerate(questions, 1):
            completed += 1
            rprint(f"  [{completed}/{total_evals}] {question[:70]}...")
            t_total = time.perf_counter()

            gpu_before = gpu_monitor.snapshot()

            initial_state = {
                "question": question,
                "original_question": question,
                "documents": [],
                "generation": "",
                "model_name": model,
                "retries": 0,
                "max_retries": settings.max_retries,
                "latency": {},
                "gpu_snapshot": {},
            }

            try:
                final_state = graph.invoke(initial_state)
            except Exception as exc:
                log.error("Graph failed for %s / %r: %s", model, question, exc)
                store.append(EvalResult(
                    run_id=run_id, model=model, question=question,
                    answer=f"ERROR: {exc}",
                ))
                continue

            total_s = time.perf_counter() - t_total
            gpu_after = gpu_monitor.snapshot()
            lat = final_state.get("latency", {})

            docs = final_state.get("documents", [])
            ctx_texts = [d["text"] for d in docs]
            ctx_scores = [d.get("score", 0.0) for d in docs]
            answer = final_state.get("generation", "")

            result = EvalResult(
                run_id=run_id,
                model=model,
                question=question,
                answer=answer,
                contexts=ctx_texts,
                context_scores=ctx_scores,
                retries=final_state.get("retries", 0),
                retrieve_s=lat.get("retrieve_s", 0.0),
                generate_s=lat.get("generate_s", 0.0),
                total_s=round(total_s, 3),
                prompt_tokens=lat.get("prompt_tokens", 0),
                completion_tokens=lat.get("completion_tokens", 0),
                tokens_per_sec=lat.get("tokens_per_sec", 0.0),
                gpu_mem_used_gb=gpu_after.get("mem_used_gb", 0.0),
                gpu_util_pct=gpu_after.get("gpu_util_pct", -1),
                gpu_temp_c=gpu_after.get("temp_c", -1),
                gpu_power_w=gpu_after.get("power_w", -1.0),
            )

            # Judge scoring (uses the judge model, not the evaluated model)
            try:
                f = judge.score_faithfulness(question, answer, ctx_texts, settings=settings)
                result.faithfulness = f.get("score", -1.0)
                result.faithfulness_reasoning = f.get("reasoning", "")
            except Exception as exc:
                log.warning("Judge faithfulness failed: %s", exc)

            try:
                r = judge.score_relevancy(question, answer, settings=settings)
                result.relevancy = r.get("score", -1.0)
                result.relevancy_reasoning = r.get("reasoning", "")
            except Exception as exc:
                log.warning("Judge relevancy failed: %s", exc)

            try:
                cp = judge.score_context_precision(question, ctx_texts, settings=settings)
                result.context_precision = cp.get("score", -1.0)
                result.context_precision_reasoning = cp.get("reasoning", "")
            except Exception as exc:
                log.warning("Judge context_precision failed: %s", exc)

            store.append(result)
            rprint(
                f"    [green]✓[/green] {result.total_s}s | "
                f"{float(result.tokens_per_sec):.0f} tok/s | "
                f"faith={float(result.faithfulness):.2f} "
                f"rel={float(result.relevancy):.2f} "
                f"ctx={float(result.context_precision):.2f}"
            )

    store.print_paths()
    _print_summary(store.results, console)
    return store.results


def _print_summary(results: list[EvalResult], console: Console) -> None:
    df = pd.DataFrame([r.to_dict() for r in results])
    if df.empty:
        return

    summary = df.groupby("model").agg(
        avg_latency=("total_s", "mean"),
        p95_latency=("total_s", lambda x: x.quantile(0.95)),
        avg_tok_s=("tokens_per_sec", "mean"),
        avg_faithfulness=("faithfulness", "mean"),
        avg_relevancy=("relevancy", "mean"),
        avg_ctx_precision=("context_precision", "mean"),
        avg_gpu_gb=("gpu_mem_used_gb", "mean"),
        queries=("question", "count"),
    ).reset_index()

    table = Table(title="Evaluation Summary", show_lines=True)
    table.add_column("Model", style="cyan bold")
    table.add_column("Queries", justify="right")
    table.add_column("Avg Latency", justify="right")
    table.add_column("P95 Latency", justify="right")
    table.add_column("Avg tok/s", justify="right")
    table.add_column("Faithfulness", justify="right")
    table.add_column("Relevancy", justify="right")
    table.add_column("Ctx Precision", justify="right")
    table.add_column("GPU GB", justify="right")

    for _, row in summary.iterrows():
        table.add_row(
            str(row["model"]),
            str(int(row["queries"])),
            f"{row['avg_latency']:.2f}s",
            f"{row['p95_latency']:.2f}s",
            f"{row['avg_tok_s']:.0f}",
            f"{row['avg_faithfulness']:.2f}",
            f"{row['avg_relevancy']:.2f}",
            f"{row['avg_ctx_precision']:.2f}",
            f"{row['avg_gpu_gb']:.1f}",
        )

    console.print()
    console.print(table)
