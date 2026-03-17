"""REST API routes for the RAG eval platform."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import uuid
from pathlib import Path

import httpx
from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from rag_eval.config import get_settings

log = logging.getLogger(__name__)
router = APIRouter()

_eval_lock = threading.Lock()
_eval_running = False
_ingest_running = False

_progress_queues: dict[str, asyncio.Queue] = {}

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


# ── Request / Response Models ───────────────────────────────────────

class IngestRequest(BaseModel):
    dataset: str = "HFforLegal/case-law"
    split: str = "train"
    text_field: str = "document"
    id_field: str = "id"
    title_field: str = "title"
    extra_meta_fields: list[str] | None = None
    subset: str | None = None
    max_rows: int | None = None
    streaming: bool = False
    embedding_backend: str | None = None
    embedding_model: str | None = None


class EvalRequest(BaseModel):
    models: list[str] | None = None
    judge_model: str | None = None
    questions: list[str] | None = None
    questions_file: str | None = None
    run_id: str | None = None
    web_search: bool = False
    searxng_url: str | None = None
    embedding_backend: str | None = None
    embedding_model: str | None = None


class WarmRequest(BaseModel):
    model: str
    keep_alive: str = "8h"
    role: str = "chat"


class UnloadRequest(BaseModel):
    model: str


class SettingsPatch(BaseModel):
    retrieval_top_k: int | None = None
    max_retries: int | None = None
    embed_batch_size: int | None = None
    web_search_enabled: bool | None = None
    web_search_max_results: int | None = None
    searxng_base_url: str | None = None


class StatusResponse(BaseModel):
    ollama_reachable: bool = False
    ollama_models: list[str] = Field(default_factory=list)
    chroma_doc_count: int = 0
    chroma_collection: str = ""
    embedding_backend: str = ""
    embedding_model: str = ""
    gpu: dict = Field(default_factory=dict)
    eval_running: bool = False
    ingest_running: bool = False


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

    try:
        r = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=5.0)
        r.raise_for_status()
        data = r.json()
        resp.ollama_reachable = True
        resp.ollama_models = [m["name"] for m in data.get("models", [])]
    except Exception:
        pass

    resp.embedding_backend = settings.embedding_backend
    resp.embedding_model = settings.embedding_model

    try:
        from rag_eval.ingest.indexer import get_or_create_collection
        col = get_or_create_collection(settings=settings)
        resp.chroma_doc_count = col.count()
        resp.chroma_collection = col.name
    except Exception:
        pass

    from rag_eval.eval import gpu_monitor
    resp.gpu = gpu_monitor.snapshot()
    resp.eval_running = _eval_running
    resp.ingest_running = _ingest_running
    return resp


# ── Questions Files ─────────────────────────────────────────────────

@router.get("/questions-files")
async def list_questions_files():
    data_dir = _PROJECT_ROOT / "data"
    if not data_dir.exists():
        return []
    files = sorted(data_dir.glob("*.txt"))
    result = []
    for f in files:
        lines = [ln.strip() for ln in f.read_text().splitlines() if ln.strip()]
        result.append({"name": f.name, "path": str(f), "count": len(lines)})
    return result


# ── SSE Helper ──────────────────────────────────────────────────────

async def _sse_stream(queue_id: str):
    """Yield SSE events from a progress queue until a 'done' event."""
    q = _progress_queues.get(queue_id)
    if q is None:
        yield f"data: {json.dumps({'error': 'unknown run_id'})}\n\n"
        return
    while True:
        try:
            event = await asyncio.wait_for(q.get(), timeout=300)
        except asyncio.TimeoutError:
            yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
            continue
        yield f"data: {json.dumps(event)}\n\n"
        if event.get("type") == "done":
            _progress_queues.pop(queue_id, None)
            break


def _make_progress_callback(queue_id: str):
    """Create a thread-safe callback that pushes events to an asyncio queue."""
    q = _progress_queues[queue_id]
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.get_event_loop()

    def _callback(event: dict) -> None:
        loop.call_soon_threadsafe(q.put_nowait, event)

    return _callback


# ── Ingest ──────────────────────────────────────────────────────────

@router.post("/ingest")
async def ingest(req: IngestRequest, background: BackgroundTasks):
    global _ingest_running
    if _ingest_running:
        raise HTTPException(409, "An ingestion is already running")

    ingest_id = uuid.uuid4().hex[:12]
    _progress_queues[ingest_id] = asyncio.Queue()
    callback = _make_progress_callback(ingest_id)

    def _run():
        global _ingest_running
        _ingest_running = True
        try:
            from rag_eval.config import get_settings
            from rag_eval.ingest.loader import load_hf_dataset
            from rag_eval.ingest.chunker import chunk_documents
            from rag_eval.ingest.indexer import index_chunks

            s = get_settings()
            if req.embedding_backend:
                s.embedding_backend = req.embedding_backend
                if req.embedding_backend == "huggingface" and not req.embedding_model:
                    s.embedding_model = s.embedding_fallback_model
            if req.embedding_model:
                s.embedding_model = req.embedding_model

            callback({"phase": "loading", "chunks_done": 0, "chunks_total": 0})
            docs = load_hf_dataset(
                req.dataset, split=req.split,
                text_field=req.text_field, id_field=req.id_field,
                title_field=req.title_field,
                extra_meta_fields=req.extra_meta_fields,
                subset=req.subset, max_rows=req.max_rows,
                streaming=req.streaming, settings=s,
            )
            callback({"phase": "chunking", "chunks_done": 0, "chunks_total": 0})
            chunks = chunk_documents(docs, settings=s)
            total_chunks = len(chunks)
            callback({"phase": "indexing", "chunks_done": 0, "chunks_total": total_chunks})
            n = index_chunks(chunks, settings=s)
            callback({"type": "done", "chunks_indexed": n})
        except Exception as exc:
            log.error("Ingestion failed: %s", exc)
            callback({"type": "done", "error": str(exc), "chunks_indexed": 0})
        finally:
            _ingest_running = False

    background.add_task(_run)
    return {"status": "ingestion_started", "dataset": req.dataset, "ingest_id": ingest_id}


@router.get("/ingest/{ingest_id}/progress")
async def ingest_progress(ingest_id: str):
    if ingest_id not in _progress_queues:
        raise HTTPException(404, f"Ingest run {ingest_id} not found")
    return StreamingResponse(
        _sse_stream(ingest_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Evaluate ────────────────────────────────────────────────────────

@router.post("/evaluate")
async def evaluate(req: EvalRequest, background: BackgroundTasks):
    global _eval_running
    if _eval_running:
        raise HTTPException(409, "An evaluation is already running")

    run_id = req.run_id or uuid.uuid4().hex[:12]
    _progress_queues[run_id] = asyncio.Queue()
    callback = _make_progress_callback(run_id)

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
            if req.web_search:
                s.web_search_enabled = True
            if req.searxng_url:
                s.searxng_base_url = req.searxng_url
            if req.embedding_backend:
                s.embedding_backend = req.embedding_backend
                if req.embedding_backend == "huggingface" and not req.embedding_model:
                    s.embedding_model = s.embedding_fallback_model
            if req.embedding_model:
                s.embedding_model = req.embedding_model

            questions = req.questions
            if not questions and req.questions_file:
                qpath = Path(req.questions_file)
                if qpath.exists():
                    questions = [q.strip() for q in qpath.read_text().splitlines() if q.strip()]

            run_evaluation(
                questions=questions,
                run_id=run_id,
                settings=s,
                on_progress=callback,
            )
        except Exception as exc:
            log.error("Evaluation failed: %s", exc)
            callback({"type": "done", "run_id": run_id, "error": str(exc)})
        finally:
            _eval_running = False

    background.add_task(_run)
    return {
        "status": "evaluation_started",
        "run_id": run_id,
        "models": req.models or get_settings().eval_models,
    }


@router.get("/evaluate/{run_id}/progress")
async def eval_progress(run_id: str):
    if run_id not in _progress_queues:
        raise HTTPException(404, f"Eval run {run_id} not found")
    return StreamingResponse(
        _sse_stream(run_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


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


# ── Model Management ────────────────────────────────────────────────

@router.get("/models/running")
async def models_running():
    settings = get_settings()
    try:
        r = httpx.get(f"{settings.ollama_base_url}/api/ps", timeout=5.0)
        r.raise_for_status()
        data = r.json()
        models = []
        for m in data.get("models", []):
            models.append({
                "name": m.get("name", ""),
                "size_gb": round(m.get("size", 0) / 1e9, 2),
                "size_vram_gb": round(m.get("size_vram", m.get("size", 0)) / 1e9, 2),
                "expires_at": m.get("expires_at", ""),
            })
        return models
    except Exception as exc:
        raise HTTPException(502, f"Failed to reach Ollama: {exc}")


class WarmAllRequest(BaseModel):
    models: list[WarmRequest]


@router.post("/models/warm")
async def warm_model(req: WarmRequest, background: BackgroundTasks):
    settings = get_settings()
    warm_id = f"warm-{uuid.uuid4().hex[:8]}"
    _progress_queues[warm_id] = asyncio.Queue()
    callback = _make_progress_callback(warm_id)

    def _run():
        import time
        time.sleep(0.3)
        callback({"model": req.model, "status": "loading", "elapsed_s": 0})
        t0 = time.perf_counter()
        try:
            if req.role == "embed":
                r = httpx.post(
                    f"{settings.ollama_base_url}/api/embed",
                    json={"model": req.model, "input": ["warmup"], "keep_alive": req.keep_alive},
                    timeout=600.0,
                )
            else:
                r = httpx.post(
                    f"{settings.ollama_base_url}/api/chat",
                    json={
                        "model": req.model,
                        "messages": [{"role": "user", "content": "hi"}],
                        "stream": False,
                        "keep_alive": req.keep_alive,
                    },
                    timeout=600.0,
                )
            r.raise_for_status()
            elapsed = round(time.perf_counter() - t0, 1)
            log.info("Model '%s' warmed up in %.1fs (role=%s, keep_alive=%s)",
                     req.model, elapsed, req.role, req.keep_alive)
            callback({"model": req.model, "status": "ready", "elapsed_s": elapsed})
        except Exception as exc:
            elapsed = round(time.perf_counter() - t0, 1)
            log.error("Failed to warm model '%s': %s", req.model, exc)
            callback({"model": req.model, "status": "error", "error": str(exc), "elapsed_s": elapsed})
        callback({"type": "done"})

    background.add_task(_run)
    return {"status": "warming", "warm_id": warm_id, "model": req.model}


@router.post("/models/warm-all")
async def warm_all_models(req: WarmAllRequest, background: BackgroundTasks):
    warm_id = f"warm-{uuid.uuid4().hex[:8]}"
    _progress_queues[warm_id] = asyncio.Queue()
    callback = _make_progress_callback(warm_id)
    settings = get_settings()

    def _run():
        import time
        time.sleep(0.3)
        total = len(req.models)
        for idx, m in enumerate(req.models, 1):
            callback({"model": m.model, "role": m.role, "status": "loading",
                       "completed": idx - 1, "total": total, "elapsed_s": 0})
            t0 = time.perf_counter()
            try:
                if m.role == "embed":
                    r = httpx.post(
                        f"{settings.ollama_base_url}/api/embed",
                        json={"model": m.model, "input": ["warmup"], "keep_alive": m.keep_alive},
                        timeout=600.0,
                    )
                else:
                    r = httpx.post(
                        f"{settings.ollama_base_url}/api/chat",
                        json={"model": m.model,
                              "messages": [{"role": "user", "content": "hi"}],
                              "stream": False, "keep_alive": m.keep_alive},
                        timeout=600.0,
                    )
                r.raise_for_status()
                elapsed = round(time.perf_counter() - t0, 1)
                log.info("Model '%s' warmed in %.1fs", m.model, elapsed)
                callback({"model": m.model, "role": m.role, "status": "ready",
                           "completed": idx, "total": total, "elapsed_s": elapsed})
            except Exception as exc:
                elapsed = round(time.perf_counter() - t0, 1)
                log.error("Failed to warm '%s': %s", m.model, exc)
                callback({"model": m.model, "role": m.role, "status": "error",
                           "completed": idx, "total": total, "elapsed_s": elapsed,
                           "error": str(exc)})
        callback({"type": "done"})

    background.add_task(_run)
    return {"status": "warming", "warm_id": warm_id, "count": len(req.models)}


@router.get("/models/warm/{warm_id}/progress")
async def warm_progress(warm_id: str):
    if warm_id not in _progress_queues:
        raise HTTPException(404, f"Warm session {warm_id} not found")
    return StreamingResponse(
        _sse_stream(warm_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/models/unload")
async def unload_model(req: UnloadRequest):
    settings = get_settings()
    try:
        r = httpx.post(
            f"{settings.ollama_base_url}/api/generate",
            json={"model": req.model, "keep_alive": 0},
            timeout=30.0,
        )
        r.raise_for_status()
        return {"status": "unloaded", "model": req.model}
    except Exception as exc:
        raise HTTPException(502, f"Failed to unload model: {exc}")


# ── Settings ────────────────────────────────────────────────────────

@router.get("/settings")
async def get_runtime_settings():
    s = get_settings()
    return {
        "retrieval_top_k": s.retrieval_top_k,
        "max_retries": s.max_retries,
        "embed_batch_size": s.embed_batch_size,
        "web_search_enabled": s.web_search_enabled,
        "web_search_max_results": s.web_search_max_results,
        "searxng_base_url": s.searxng_base_url,
        "embedding_backend": s.embedding_backend,
        "embedding_model": s.embedding_model,
        "ollama_base_url": s.ollama_base_url,
    }


@router.patch("/settings")
async def patch_settings(patch: SettingsPatch):
    s = get_settings()
    updated = {}
    for field_name, value in patch.model_dump(exclude_none=True).items():
        setattr(s, field_name, value)
        updated[field_name] = value
    if not updated:
        raise HTTPException(400, "No fields to update")
    return {"status": "updated", "changes": updated}
