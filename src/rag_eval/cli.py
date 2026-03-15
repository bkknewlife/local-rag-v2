"""CLI entry points for ingest, evaluate, and serve."""

from __future__ import annotations

import argparse
import logging

from rich.logging import RichHandler


def _setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )


# ── Ingest ──────────────────────────────────────────────────────────

def cli_ingest() -> None:
    parser = argparse.ArgumentParser(description="Ingest a HuggingFace dataset into ChromaDB")
    parser.add_argument("dataset", help="HuggingFace dataset id, e.g. rajpurkar/squad_v2")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-field", default="context")
    parser.add_argument("--id-field", default="id")
    parser.add_argument("--title-field", default="title")
    parser.add_argument("--extra-meta-fields", nargs="*", default=None,
                        help="Additional columns to store in metadata (e.g. citation state issuer)")
    parser.add_argument("--subset", default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--streaming", action="store_true",
                        help="Stream from HF without downloading full dataset (recommended with --max-rows for large datasets)")
    parser.add_argument("--embedding-backend", choices=["ollama", "huggingface"],
                        default=None, help="Embedding backend (default: ollama)")
    parser.add_argument("--embedding-model", default=None,
                        help="Override embedding model name")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    _setup_logging(args.verbose)

    from rag_eval.config import get_settings
    from rag_eval.ingest.loader import load_hf_dataset
    from rag_eval.ingest.chunker import chunk_documents
    from rag_eval.ingest.indexer import index_chunks

    settings = get_settings()
    if args.embedding_backend:
        settings.embedding_backend = args.embedding_backend
        if args.embedding_backend == "huggingface" and not args.embedding_model:
            settings.embedding_model = settings.embedding_fallback_model
    if args.embedding_model:
        settings.embedding_model = args.embedding_model
    docs = load_hf_dataset(
        args.dataset,
        split=args.split,
        text_field=args.text_field,
        id_field=args.id_field,
        title_field=args.title_field,
        extra_meta_fields=args.extra_meta_fields,
        subset=args.subset,
        max_rows=args.max_rows,
        streaming=args.streaming,
        settings=settings,
    )
    chunks = chunk_documents(docs, settings=settings)
    n = index_chunks(chunks, settings=settings)
    print(f"\nDone — indexed {n} chunks.")


# ── Evaluate ────────────────────────────────────────────────────────

def cli_evaluate() -> None:
    parser = argparse.ArgumentParser(description="Run RAG evaluation sweep")
    parser.add_argument("--questions-file", default=None,
                        help="Path to a text file with one question per line")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Override models to evaluate")
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--embedding-backend", choices=["ollama", "huggingface"],
                        default=None, help="Embedding backend (must match ingestion)")
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    _setup_logging(args.verbose)

    from rag_eval.config import get_settings
    from rag_eval.eval.harness import run_evaluation

    settings = get_settings()
    if args.embedding_backend:
        settings.embedding_backend = args.embedding_backend
        if args.embedding_backend == "huggingface" and not args.embedding_model:
            settings.embedding_model = settings.embedding_fallback_model
    if args.embedding_model:
        settings.embedding_model = args.embedding_model
    if args.models:
        settings.eval_models = args.models
    if args.judge_model:
        settings.judge_model = args.judge_model

    questions: list[str] | None = None
    if args.questions_file:
        from pathlib import Path
        questions = [
            q.strip() for q in Path(args.questions_file).read_text().splitlines()
            if q.strip()
        ]

    run_evaluation(questions=questions, run_id=args.run_id, settings=settings)


# ── Serve ───────────────────────────────────────────────────────────

def cli_serve() -> None:
    parser = argparse.ArgumentParser(description="Start the RAG eval dashboard")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    _setup_logging(args.verbose)

    import uvicorn
    from rag_eval.config import get_settings

    settings = get_settings()
    uvicorn.run(
        "rag_eval.api.app:app",
        host=args.host or settings.api_host,
        port=args.port or settings.api_port,
        reload=False,
    )
