"""REST API routes for the RAG eval platform."""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

import httpx
from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from rag_eval.config import get_settings
from rag_eval.eval import gpu_monitor

log = logging.getLogger(__name__)
router = APIRouter()

_eval_lock = threading.Lock()
_eval_running = False


# ── Request / Response Models ───────────────────────────────────────

class IngestRequest(BaseModel):
    dataset: str = "rajpurkar/squad_v2"
    split: str = "train"
    text_field: str = "context"
    id_field: str = "id"
    title_field: str = "title"
    subset: str | None = None
    max_rows: int | None = None


class EvalRequest(BaseModel):
    models: list[str] | None = None
    judge_model: str | None = None
    questions: list[str] | None = None
    run_id: str | None = None


class StatusResponse(BaseModel):
    ollama_reachable: bool = False
    ollama_models: list[str] = Field(default_factory=list)
    chroma_doc_count: int = 0
    chroma_collection: str = ""
    embedding_backend: str = ""
    embedding_model: str = ""
    gpu: dict = Field(default_factory=dict)
    eval_running: bool = False


# ── Dashboard ───────────────────────────────────────────────────────

@router.get("/", response_class=HTMLResponse)
async def dashboard():
    html_path = Path(__file__).resolve().parent.parent / "dashboard" / "index.html"
    if not html_path.exists():
        return HTMLResponse("<h1>Dashboard not found</h1>", status_code=404)
    return HTMLResponse(html_path.read_text())


# ── Status ──────────────────────────────────────────────────────────

@router.get("/status", response_model=StatusResponse)
async def status():
    settings = get_settings()
    resp = StatusResponse()

    # Ollama
    try:
        r = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=5.0)
        r.raise_for_status()
        data = r.json()
        resp.ollama_reachable = True
        resp.ollama_models = [m["name"] for m in data.get("models", [])]
    except Exception:
        pass

    # Embedding info
    resp.embedding_backend = settings.embedding_backend
    resp.embedding_model = settings.embedding_model

    # ChromaDB
    try:
        from rag_eval.ingest.indexer import get_or_create_collection
        col = get_or_create_collection(settings=settings)
        resp.chroma_doc_count = col.count()
        resp.chroma_collection = col.name
    except Exception:
        pass

    resp.gpu = gpu_monitor.snapshot()
    resp.eval_running = _eval_running
    return resp


# ── Ingest ──────────────────────────────────────────────────────────

@router.post("/ingest")
async def ingest(req: IngestRequest, background: BackgroundTasks):
    def _run():
        from rag_eval.config import get_settings
        from rag_eval.ingest.loader import load_hf_dataset
        from rag_eval.ingest.chunker import chunk_documents
        from rag_eval.ingest.indexer import index_chunks

        s = get_settings()
        docs = load_hf_dataset(
            req.dataset, split=req.split,
            text_field=req.text_field, id_field=req.id_field,
            title_field=req.title_field, subset=req.subset,
            max_rows=req.max_rows, settings=s,
        )
        chunks = chunk_documents(docs, settings=s)
        index_chunks(chunks, settings=s)

    background.add_task(_run)
    return {"status": "ingestion_started", "dataset": req.dataset}


# ── Evaluate ────────────────────────────────────────────────────────

@router.post("/evaluate")
async def evaluate(req: EvalRequest, background: BackgroundTasks):
    global _eval_running
    if _eval_running:
        raise HTTPException(409, "An evaluation is already running")

    def _run():
        global _eval_running
        _eval_running = True
        try:
            from rag_eval.config import get_settings
            from rag_eval.eval.harness import run_evaluation

            s = get_settings()
            if req.models:
                s.eval_models = req.models
            if req.judge_model:
                s.judge_model = req.judge_model
            run_evaluation(questions=req.questions, run_id=req.run_id, settings=s)
        finally:
            _eval_running = False

    background.add_task(_run)
    return {"status": "evaluation_started", "models": req.models or get_settings().eval_models}


# ── Results ─────────────────────────────────────────────────────────

@router.get("/results")
async def list_results():
    settings = get_settings()
    results_dir = settings.results_dir
    if not results_dir.exists():
        return []
    files = sorted(results_dir.glob("*.json"), reverse=True)
    return [{"run_id": f.stem, "path": str(f)} for f in files]


@router.get("/results/{run_id}")
async def get_result(run_id: str):
    settings = get_settings()
    json_path = settings.results_dir / f"{run_id}.json"
    if not json_path.exists():
        raise HTTPException(404, f"Run {run_id} not found")
    return JSONResponse(json.loads(json_path.read_text()))
