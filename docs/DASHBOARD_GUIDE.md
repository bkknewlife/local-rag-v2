# RAG Eval Dashboard -- User Guide

> **URL**: `http://<host>:8000/` (default when running `python3 scripts/serve.py`)
>
> The dashboard is the browser-based control plane for the RAG Evaluation
> platform.  Every feature available via the CLI is also exposed here, plus
> real-time progress tracking via Server-Sent Events (SSE).

---

## Table of Contents

1. [Header & Status Badges](#1-header--status-badges)
2. [System Status Cards](#2-system-status-cards)
3. [Model Management](#3-model-management)
   - [Loaded Models Table](#31-loaded-models-table)
   - [Warm Up Model Controls](#32-warm-up-model-controls)
4. [Dataset Ingestion](#4-dataset-ingestion)
5. [Performance Settings](#5-performance-settings)
6. [Evaluation](#6-evaluation)
   - [Models to Evaluate](#61-models-to-evaluate)
   - [Judge Model](#62-judge-model)
   - [Questions Source](#63-questions-source)
   - [Run ID](#64-run-id)
   - [Web Search Toggle](#65-web-search-toggle)
   - [Start Evaluation & Live Progress](#66-start-evaluation--live-progress)
7. [Results Viewer](#7-results-viewer)
8. [Auto-Refresh Behavior](#8-auto-refresh-behavior)

---

## 1. Header & Status Badges

Three colored badges in the top-right corner provide at-a-glance health:

| Badge         | Green (OK)                       | Yellow (Warning)                  | Red (Error)                 |
|---------------|----------------------------------|-----------------------------------|-----------------------------|
| **OLLAMA**    | Ollama API is reachable          | Checking...                       | Ollama is unreachable       |
| **CHROMA**    | ChromaDB has indexed documents   | Collection exists but is empty    | --                          |
| **GPU**       | GPU detected, shows utilization% | GPU not available or unsupported  | --                          |

These badges refresh automatically every 10 seconds.

---

## 2. System Status Cards

Four summary cards below the header:

| Card              | What It Shows                                                                 |
|-------------------|-------------------------------------------------------------------------------|
| **Ollama Models** | Total number of models installed in Ollama (not just loaded -- all available) |
| **Indexed Chunks** | Total document chunks stored in ChromaDB across all collections              |
| **GPU Memory**    | Current GPU VRAM usage (e.g., `45.2 / 96.0 GB`)                             |
| **GPU Temp**      | Current GPU temperature in Celsius                                           |

---

## 3. Model Management

### 3.1 Loaded Models Table

Shows models **currently loaded in VRAM** (what Ollama's `/api/ps` reports). This is *not* a list of all installed models -- only those actively occupying GPU memory.

| Column              | Description                                                                                         |
|---------------------|-----------------------------------------------------------------------------------------------------|
| **Model**           | Model name with a green checkmark, indicating it is loaded                                          |
| **Size (GB)**       | Total model size on disk                                                                            |
| **VRAM (GB)**       | GPU memory the model is currently consuming                                                         |
| **Keep-Alive Expires** | Timestamp when Ollama will auto-unload this model if no new requests arrive                      |
| **Action**          | **Unload from VRAM** button -- evicts the model from GPU memory immediately. The model remains installed; it will simply need to be reloaded (warmed) before the next use. Use this to free VRAM for a different model. |

> **Note**: If you have `OLLAMA_MAX_LOADED_MODELS=3` in your Ollama Docker
> config, at most 3 models will appear here at any time.  When a 4th is
> loaded, Ollama automatically evicts the least-recently-used one.

The table scrolls vertically if more models are loaded than fit on screen.

---

### 3.2 Warm Up Model Controls

Pre-loading a model into VRAM before evaluation avoids cold-start latency on
the first query.

| Control            | Description                                                                                                      |
|--------------------|------------------------------------------------------------------------------------------------------------------|
| **Model dropdown** | Lists all installed Ollama models. Models already loaded show a `✓` prefix and `(loaded)` suffix.                |
| **Role dropdown**  | **Chat** -- warm for inference (answer generation, judge scoring). Use for all LLMs.<br>**Embedding** -- warm for embedding (vector generation). Use only for your embedding model (e.g., `nomic-embed-text`). |
| **Keep-Alive**     | Duration Ollama keeps the model in VRAM after the last request. Default `8h` (8 hours). Accepts: `5m`, `1h`, `8h`, `24h`, etc. Set long enough to cover your full evaluation run. |
| **Warm Up** button | Warms the single model selected in the dropdown with the chosen role and keep-alive.                             |
| **Warm All for Eval** button | One-click convenience: warms **all** of the following in sequence:<br>1. The embedding model (from Performance Settings)<br>2. Every model checked in "Models to Evaluate"<br>3. The selected judge model<br>All use the keep-alive value from the text field. |

#### Warm-Up Progress Panel

Appears below the controls when a warm-up is in progress:

- **Progress label**: Shows `Warming X / Y...` during warm-up, then `X / Y models ready` on completion.
- **Progress bar**: Fills proportionally as each model completes.
- **Log**: Lists each model with a ✓ (success) or ✗ (error) and elapsed time.
- The panel **auto-hides** 4 seconds after all models finish. If SSE events are lost, a 30-second safety timer auto-completes.

---

## 4. Dataset Ingestion

Downloads a HuggingFace dataset, splits documents into chunks, generates
embeddings, and indexes them into ChromaDB.

### Controls

| Field                | Description                                                                                                                                                  |
|----------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Preset**           | Quick-fill dropdown for known datasets:<br>- **HFforLegal/case-law** -- multi-jurisdiction legal case law<br>- **santoshtyss/us-court-cases** -- US court cases<br>- **Custom** -- blank fields for any HuggingFace dataset |
| **Dataset ID**       | Full HuggingFace dataset identifier (e.g., `HFforLegal/case-law`). Auto-filled by preset.                                                                   |
| **Text Field**       | Name of the dataset column containing the document text to embed. Default: `document`.                                                                       |
| **ID Field**         | Column used as the unique document identifier in ChromaDB. Default: `id`.                                                                                    |
| **Title Field**      | Column for the document title, stored as metadata. Default: `title`.                                                                                         |
| **Extra Meta Fields**| Comma-separated list of additional columns to store as metadata (e.g., `citation, jurisdiction`). Useful for filtering results later.                         |
| **Max Rows**         | Limit the number of rows loaded from the dataset. Leave blank to load all. Useful for quick testing (e.g., `500`).                                           |
| **Streaming**        | When checked, streams data from HuggingFace instead of downloading the full dataset first. Recommended for very large datasets or when using Max Rows, as it avoids downloading data you won't use. |

### Buttons

| Button               | Description                                                      |
|----------------------|------------------------------------------------------------------|
| **Start Ingestion**  | Begins the ingestion pipeline. Disabled while ingestion is active. |

### Progress Panel

Appears during ingestion:

- **Phase**: Shows the current stage -- `loading` (downloading data), `chunking` (splitting text), `indexing` (embedding + ChromaDB upsert), or `Done`.
- **Chunks**: Shows `chunks_done / chunks_total` during indexing.
- **Progress bar**: Fills proportionally during the indexing phase.

---

## 5. Performance Settings

Runtime-tunable parameters that affect retrieval and evaluation behavior.
Changes take effect immediately for the next evaluation run (no server restart
needed).

### Editable Fields

| Field                       | Description                                                                                                              | Default |
|-----------------------------|--------------------------------------------------------------------------------------------------------------------------|---------|
| **Retrieval Top-K**         | Number of document chunks retrieved from ChromaDB per query. Higher = more context but slower and noisier.               | `4`     |
| **Max Retries**             | Maximum number of self-correction loops in the LangGraph pipeline. If the LLM-as-judge detects a hallucination or irrelevant answer, the pipeline rewrites the query and retries. Set to `0` to disable self-correction. | `1`     |
| **Embed Batch Size**        | Number of text chunks sent per embedding API call. Larger batches are faster but use more VRAM. Reduce if Ollama runs out of memory during indexing. | `32`    |
| **Web Search Max Results**  | Maximum number of SearXNG search results to fetch when web search is enabled.                                            | `3`     |
| **SearXNG URL**             | Base URL of your local SearXNG instance (e.g., `http://localhost:8888`). Required only if web search is enabled.         | --      |

### Read-Only Fields

These are set at startup via environment variables or `config.py` and cannot be
changed at runtime:

| Field                 | Description                                                                        |
|-----------------------|------------------------------------------------------------------------------------|
| **Embedding Backend** | Which backend generates embeddings: `ollama` (GPU-accelerated, 768-dim) or `huggingface` (CPU, 384-dim). |
| **Embedding Model**   | The specific model used for embeddings (e.g., `nomic-embed-text`).                 |
| **Ollama URL**        | The Ollama API endpoint (e.g., `http://localhost:11434`).                          |

### Button

| Button              | Description                                                               |
|---------------------|---------------------------------------------------------------------------|
| **Apply Settings**  | Sends the editable field values to the server. A toast confirms which settings were changed. |

---

## 6. Evaluation

Runs the full RAG evaluation pipeline: for each question, retrieves context
from ChromaDB, generates an answer with each selected model, and scores the
answer using the judge model.

### 6.1 Models to Evaluate

A grid of **checkboxes** listing all installed Ollama models. Check one or more
models to include them in the evaluation sweep. Each model is evaluated against
every question independently.

- Models currently loaded in VRAM show a `✓` prefix and `(loaded)` suffix.
- You can select multiple models -- the evaluation runs them sequentially.

### 6.2 Judge Model

A **dropdown** to select which model acts as the LLM-as-judge. The judge scores
each answer on three dimensions:

| Metric                | What the Judge Evaluates                                                        | Scale |
|-----------------------|---------------------------------------------------------------------------------|-------|
| **Faithfulness**      | Is the answer grounded in the retrieved context? (no hallucinations)            | 0 - 1 |
| **Relevancy**         | Does the answer address the original question?                                  | 0 - 1 |
| **Context Precision** | Is the retrieved context relevant to the question?                              | 0 - 1 |

> **Tip**: The judge model should ideally be a capable model (e.g.,
> `qwen3:30b` or `mistral-small3.2`). It runs separately from the eval
> models, so it should also be warmed before starting.

### 6.3 Questions Source

A **dropdown** with three options:

| Source           | Description                                                                                                                |
|------------------|----------------------------------------------------------------------------------------------------------------------------|
| **Server File**  | Select a `.txt` file from the `data/` directory on the server. Each line is one question. This is the recommended approach. |
| **Custom (paste)** | Type or paste questions directly into a text area (one per line). Good for ad-hoc testing.                             |
| **Default (5 built-in)** | Uses 5 hardcoded sample questions built into the evaluation harness. Useful for a quick smoke test.              |

When **Server File** is selected, an additional dropdown appears listing all
`.txt` files found in the `data/` directory, along with their question count.

### 6.4 Run ID

A **text field** for a custom run identifier (e.g., `legal-multi-model-v2`).
If left blank, a timestamped ID is auto-generated. The run ID is used:

- As the filename for results (`data/results/<run_id>.csv` and `.json`)
- To select and view results later in the Results Viewer

> **Warning**: If you reuse a run ID, new results are **appended** to the
> existing files (incremental persistence), not overwritten.

### 6.5 Web Search Toggle

| Control                                  | Description                                                                                             |
|------------------------------------------|---------------------------------------------------------------------------------------------------------|
| **Augment with SearXNG web search**      | When checked, the LangGraph pipeline adds a web search step: if the initial ChromaDB retrieval scores poorly, the system queries your local SearXNG instance for supplementary context. **Only the question text is sent to SearXNG** -- no customer/dataset data leaves the machine. |

Requires a running SearXNG instance and a valid SearXNG URL in Performance
Settings.

### 6.6 Start Evaluation & Live Progress

| Button               | Description                                                                                               |
|----------------------|-----------------------------------------------------------------------------------------------------------|
| **Start Evaluation** | Launches the evaluation run. Disabled while a run is active. Validates that at least one model and a judge are selected. |

#### Live Progress Panel

Appears once evaluation starts:

- **Counter**: Shows `completed / total` questions across all models.
- **Current model**: Displays which model is currently being evaluated.
- **Progress bar**: Fills as questions complete.
- **Scrolling log**: Lists each completed question with:
  - ✓ (green) for successful evaluations
  - ✗ (red) for errors (e.g., embedding dimension mismatch, Ollama timeout)
  - Truncated question text and model name

Results are persisted incrementally -- if the evaluation is interrupted, all
completed questions are already saved.

---

## 7. Results Viewer

Browse and visualize completed evaluation runs.

### Controls

| Control            | Description                                                                        |
|--------------------|------------------------------------------------------------------------------------|
| **Run dropdown**   | Lists all completed runs found in `data/results/`. Select one to load its data.   |
| **Refresh button** | Re-scans the results directory for new runs.                                      |

### Summary Cards

Four cards showing aggregated metrics for the selected run:

| Card                | Value                                                   |
|---------------------|---------------------------------------------------------|
| **Models**          | Number of distinct models evaluated                     |
| **Questions**       | Number of distinct questions                            |
| **Avg Latency**     | Mean end-to-end latency per question (seconds)          |
| **Avg Faithfulness**| Mean faithfulness score across all successful evaluations |

### Charts

| Chart              | Type  | Description                                                            |
|--------------------|-------|------------------------------------------------------------------------|
| **Avg Latency**    | Bar   | Average latency per model (seconds). Lower is better.                  |
| **Quality Scores** | Radar | Per-model radar comparing Faithfulness, Relevancy, Context Precision. Larger area is better. |

### Results Table

Detailed per-question breakdown:

| Column            | Description                                                         |
|-------------------|---------------------------------------------------------------------|
| **Model**         | Which model generated the answer                                    |
| **Question**      | Truncated question text (hover for full answer in tooltip)          |
| **Latency**       | End-to-end time for this question (seconds)                         |
| **tok/s**         | Tokens per second during generation                                 |
| **Faith.**        | Faithfulness score (0-1), or `-` if evaluation failed               |
| **Relev.**        | Relevancy score (0-1)                                               |
| **Ctx Prec.**     | Context precision score (0-1)                                       |
| **Retries**       | Number of self-correction loops triggered                           |
| **Web**           | If web search was used: ✓ and lookup time; otherwise `--`           |
| **GPU GB**        | GPU memory delta during this question                               |

The table scrolls vertically for large result sets.

---

## 8. Auto-Refresh Behavior

The dashboard polls two endpoints every **10 seconds** automatically:

- `/status` -- updates badges, system status cards, model selectors, and
  button disabled states.
- `/models/running` -- updates the loaded models table and `✓ (loaded)`
  indicators across all dropdowns/checkboxes.

No manual refresh is needed to see models load/unload or system status changes.

---

## Quick Reference: Typical Workflow

1. **Check System Status** -- verify Ollama, ChromaDB, and GPU badges are green.
2. **Ingest a Dataset** -- select a preset or enter a custom HuggingFace dataset, click *Start Ingestion*, wait for completion.
3. **Warm Models** -- select eval models, judge model, then click *Warm All for Eval* with a long keep-alive (e.g., `8h`).
4. **Tune Settings** -- adjust Top-K, Max Retries if needed, click *Apply Settings*.
5. **Run Evaluation** -- check models, pick judge, choose question source, optionally enable web search, click *Start Evaluation*.
6. **View Results** -- select the run from the Results dropdown, analyze charts and table.
