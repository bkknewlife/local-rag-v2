"""FastAPI application — dashboard + REST API for the RAG eval platform."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from rag_eval.api.routes import router
from rag_eval.logging_setup import setup_logging

setup_logging()

app = FastAPI(
    title="RAG Eval Platform",
    description="Local RAG evaluation with LangGraph on DGX Spark",
    version="0.1.0",
)

app.include_router(router)

_dashboard_dir = Path(__file__).resolve().parent.parent / "dashboard"
if _dashboard_dir.is_dir():
    app.mount("/static", StaticFiles(directory=str(_dashboard_dir)), name="static")
