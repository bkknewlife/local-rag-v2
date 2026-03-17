# RAG Evaluation Platform

**Local RAG evaluation with LangGraph on NVIDIA DGX Spark (GB10)**

End-to-end evaluation of retrieval-augmented generation across multiple local
LLM models served by Ollama, with ChromaDB vector indexing and LLM-as-judge
quality scoring — fully offline, no cloud dependencies.

## Architecture

```
HF Dataset → Chunk → Embed (Ollama/HuggingFace) → ChromaDB Index
                                                        ↓
Question Bank → Model Loop → LangGraph RAG Pipeline → Metrics + Judge Scores
                                   ↑ (opt-in)              ↓
                              SearXNG Web Search    FastAPI Dashboard
```

The LangGraph pipeline is **self-correcting**: it grades retrieved documents,
checks for hallucination, evaluates answer usefulness, and retries with
query-rewriting when quality checks fail. An `original_question` field is
preserved across rewrites to prevent query drift.

## Hardware Target

This platform is optimised for the **NVIDIA DGX Spark (GB10)** which has a
119 GiB **unified memory** pool shared between CPU and GPU. Key implications:

- Multiple models can be resident simultaneously when configured correctly
- Standard NVML memory queries (`nvmlDeviceGetMemoryInfo`) return "Not Supported";
  the GPU monitor falls back to `nvidia-smi` parsing
- Ollama runs in Docker with CDI-based GPU passthrough (`--device nvidia.com/gpu=all`)

## Prerequisites

- **Docker** with NVIDIA CDI support
- **Ollama** running in Docker at `localhost:11434` (see [Ollama Setup](#ollama-docker-setup))
- **SearXNG** (optional) — running in Docker at `localhost:8888` for [web search augmentation](#web-search-searxng)
- **Python 3.12+**
- Pull at least one embedding model and one chat model:

```bash
docker exec -it ollama ollama pull nomic-embed-text
docker exec -it ollama ollama pull mistral-small3.2
```

## Ollama Docker Setup

The Ollama container configuration is **critical** for performance. The two
most impactful settings are GPU access and `OLLAMA_MAX_LOADED_MODELS`.

### Creating the Container

```bash
docker run -d \
  --name ollama \
  --device nvidia.com/gpu=all \
  -v ollama:/root/.ollama \
  -p 11434:11434 \
  -e OLLAMA_HOST=0.0.0.0:11434 \
  -e OLLAMA_MAX_LOADED_MODELS=3 \
  -e OLLAMA_FLASH_ATTENTION=1 \
  -e OLLAMA_KV_CACHE_TYPE=q8_0 \
  -e OLLAMA_NUM_PARALLEL=1 \
  ollama/ollama:latest
```

### Key Environment Variables

| Variable | Recommended | Why |
|---|---|---|
| `OLLAMA_MAX_LOADED_MODELS` | `3` | Keeps embed + eval + judge models resident simultaneously. With `1` (default), every model transition triggers a 2-4 minute swap. |
| `OLLAMA_FLASH_ATTENTION` | `1` | Enables flash attention for faster inference |
| `OLLAMA_KV_CACHE_TYPE` | `q8_0` | Quantised KV cache reduces memory per loaded model |
| `OLLAMA_NUM_PARALLEL` | `1` | Single-request mode (sufficient for sequential eval) |

### GPU Access (DGX Spark)

The DGX Spark does **not** use the `nvidia` Docker runtime. GPU access is
provided via CDI (Container Device Interface):

```bash
# Correct (CDI):
docker run --device nvidia.com/gpu=all ...

# Also works:
docker run --gpus all ...

# WRONG (not available on DGX Spark):
docker run --runtime=nvidia ...   # "unknown runtime" error
```

Verify GPU access inside the container:

```bash
docker exec ollama nvidia-smi
```

### Why OLLAMA_MAX_LOADED_MODELS Matters

During evaluation, three models are used concurrently:

| Role | Example Model | Size |
|---|---|---|
| Embedding | `nomic-embed-text` | ~0.3 GB |
| Evaluation | `mistral-small3.2` | ~27 GB |
| Judge | `deepseek-r1-abliterated:32b-q8` | ~35 GB |

With `OLLAMA_MAX_LOADED_MODELS=1` (default), Ollama evicts the current model
before loading the next one. Each swap takes **2-4 minutes** on the GB10. A
single question requires 3-4 swaps (embed → eval → judge → embed), turning a
10-second evaluation into a 15-minute ordeal.

With `OLLAMA_MAX_LOADED_MODELS=3`, all three models stay resident (~62 GB total
out of 119 GB available), and model transitions are instant.

| Setting | Swaps/question | Time/question | 25 questions |
|---|---|---|---|
| `MAX_LOADED_MODELS=1` | 3-4 × 3 min | ~40 min | ~16 hours |
| `MAX_LOADED_MODELS=3` | 0 | ~10-30s | ~15-30 min |

## Pre-warming Models

Before running evaluation, pre-load all three models so Ollama has them
resident. This eliminates the cold-start penalty on the first question:

```bash
# 1. Load embedding model
curl -s http://localhost:11434/api/embed \
  -d '{"model":"nomic-embed-text","input":["warmup"],"keep_alive":"8h"}' \
  > /dev/null && echo "embed OK"

# 2. Load evaluation model
curl -s http://localhost:11434/api/chat \
  -d '{"model":"mistral-small3.2","messages":[{"role":"user","content":"hi"}],"stream":false,"keep_alive":"8h"}' \
  > /dev/null && echo "eval OK"

# 3. Load judge model
curl -s http://localhost:11434/api/chat \
  -d '{"model":"deepseek-r1-abliterated:32b-q8","messages":[{"role":"user","content":"hi"}],"stream":false,"keep_alive":"8h"}' \
  > /dev/null && echo "judge OK"

# 4. Verify all models are resident
curl -s http://localhost:11434/api/ps | python3 -c "
import sys,json
d=json.load(sys.stdin)
for m in d['models']:
    print(f'  {m[\"name\"]:<40} {m[\"size\"]/1e9:.1f} GB')
print(f'Total: {sum(m[\"size\"] for m in d[\"models\"])/1e9:.1f} GB')
"
```

The `keep_alive=8h` parameter tells Ollama to keep the model loaded for
8 hours, preventing automatic eviction during long evaluation runs.

## Quick Start

```bash
# Create a virtual environment and install
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

# 1. Ingest a legal dataset (streaming mode for large datasets)
python scripts/ingest.py HFforLegal/case-law \
  --text-field document --id-field id --title-field title \
  --extra-meta-fields citation jurisdiction \
  --max-rows 5000 --streaming -v

# 2. Run evaluation across models
python scripts/evaluate.py \
  --models mistral-small3.2 gemma3:27b qwen3:30b \
  --judge-model deepseek-r1-abliterated:32b-q8 \
  --questions-file data/questions_legal.txt \
  --run-id legal-eval \
  -v

# 3. Start the dashboard
python scripts/serve.py
# Open http://localhost:8000
```

## CLI Reference

### Ingest

```bash
python scripts/ingest.py <dataset> [options]

Options:
  --split              HF split name (default: train)
  --text-field         Column with text to index (default: context)
  --id-field           Column for document IDs (default: id)
  --title-field        Column for titles in metadata (default: title)
  --extra-meta-fields  Additional columns to store in metadata
  --subset             HF dataset config/subset
  --max-rows           Limit number of rows to ingest
  --streaming          Stream from HF without downloading full dataset
  --embedding-backend  "ollama" (default, GPU) or "huggingface" (CPU)
  --embedding-model    Override embedding model name
  -v, --verbose        Debug logging
```

### Evaluate

```bash
python scripts/evaluate.py [options]

Options:
  --questions-file     Path to a text file with one question per line
  --models             Space-separated model names to evaluate
  --judge-model        Model to use as the LLM-as-judge
  --embedding-backend  Must match the backend used during ingestion
  --embedding-model    Override embedding model name
  --web-search         Augment retrieval with SearXNG web results (off by default)
  --searxng-url        SearXNG base URL (default: http://localhost:8888)
  --run-id             Custom run identifier
  -v, --verbose        Debug logging
```

### Dashboard

```bash
python scripts/serve.py [--host 0.0.0.0] [--port 8000]
```

## Embedding Backends

Two embedding backends are supported. The backend used for **ingestion** must
match the backend used for **retrieval** — vectors of different dimensions are
incompatible.

| Backend | Model | Dimensions | Speed | Notes |
|---|---|---|---|---|
| `ollama` (default) | `nomic-embed-text` | 768 | GPU-accelerated | Requires Ollama running |
| `huggingface` | `all-MiniLM-L6-v2` | 384 | CPU-only | No external dependencies |

The ChromaDB collection name encodes the backend
(e.g. `rag_eval__ollama-nomic-embed-text`) so collections from different
backends are automatically separated.

### Embedding Fallback

When `embedding_fallback=True` (the default) and the Ollama backend is
selected, a failed Ollama embed call will fall back to HuggingFace. However,
the **retriever disables fallback** (`embedding_fallback=False`) to prevent
dimension mismatches — querying a 768-dim collection with 384-dim vectors
always fails.

### Embedding Retry Logic

The `OllamaBackend` includes retry with exponential back-off (5 attempts,
starting at 15s delay) to handle model-swap delays when Ollama needs to
reload the embedding model after a chat model was used.

## LangGraph Pipeline

The evaluation graph follows this topology:

```
retrieve ──(web_search on?)──→ web_search → grade_documents
           └─(off)──────────→ grade_documents

grade_documents ──(relevant)──→ generate
                └─(empty)───→ rewrite → retrieve

generate → check_hallucination ──(grounded)──→ check_usefulness
                                └─(not)──────→ generate (retries++)

check_usefulness ──(useful)──→ END (record result)
                  └─(not)───→ rewrite → retrieve (retries++)
```

When `--web-search` is enabled, the `web_search` node queries SearXNG and
**appends** web results to the ChromaDB documents before grading. See
[Web Search (SearXNG)](#web-search-searxng) for details.

Each node operates on a shared `GraphState` TypedDict that accumulates
question, documents, generation, latency breakdowns, and GPU snapshots.

### Self-Correction Safeguards

- **Retry counter**: `check_hallucination` and `check_usefulness` both
  increment the retry counter on negative verdicts, preventing infinite loops
- **Original question preservation**: The `original_question` field is stored
  in graph state and used by `rewrite_query` to prevent multi-hop query drift
  where successive rewrites stray from the actual intent
- **Default `max_retries=1`**: One retry is enough to catch genuinely bad
  answers without over-correcting good ones. Higher values can cause quality
  degradation as rewrites produce increasingly divergent queries

## Evaluation Metrics

**Quality** (scored by LLM-as-judge, 0.0-1.0):
- **Faithfulness** — Is every claim in the answer supported by context?
- **Answer Relevancy** — Does the answer address the question?
- **Context Precision** — Are retrieved documents relevant to the question?

The judge model returns JSON scores which are coerced to float and clamped
to `[0.0, 1.0]` to handle models that return scores as strings.

**Performance**:
- End-to-end latency (seconds)
- Token throughput (tokens/sec)
- Prompt and completion token counts
- Retry count per question

**GPU Monitoring**:
- GPU utilisation, temperature, power draw
- Memory usage (via `nvidia-smi` fallback on GB10 unified memory)
- Uses `nvidia-ml-py` (not the deprecated `pynvml` shim package)

**Persistence**: Results are saved incrementally after every question to both
CSV and JSON in `data/results/`, so partial results survive interruptions.

## Dashboard

The web dashboard (`python scripts/serve.py`) provides a full GUI alternative
to the CLI, with the same capabilities plus live progress tracking. Start it
with:

```bash
source .venv/bin/activate
python scripts/serve.py
# Open http://localhost:8000
```

### Dashboard Panels

The dashboard is organized into seven sections, from top to bottom:

1. **System Status** -- Ollama connectivity, indexed chunk count, GPU memory
   and temperature (auto-refreshes every 10 seconds)

2. **Model Management** -- Table of models currently loaded in VRAM with size
   and expiry, warm-up controls (model picker, chat/embed role selector,
   keep-alive duration), unload buttons, and a "Warm All for Eval" button
   that pre-loads embedding + eval + judge models in one click

3. **Dataset Ingestion** -- Full ingestion form with preset selector
   (HFforLegal/case-law, santoshtyss/us-court-cases, Custom), field mappings,
   max rows, streaming toggle. Live progress bar via SSE shows
   loading/chunking/indexing phases

4. **Performance Settings** -- Runtime-tunable fields (retrieval_top_k,
   max_retries, embed_batch_size, web_search_max_results, SearXNG URL) with
   an Apply button. Read-only display of embedding backend/model and Ollama URL

5. **Evaluation Configuration** -- Model checkboxes, judge model dropdown,
   questions source (server file / custom paste / default), web search toggle,
   run ID input

6. **Live Progress** -- Progress bar, current model/question display, and
   scrolling event log connected via SSE. Appears automatically when
   evaluation starts

7. **Results Viewer** -- Run selector dropdown, summary cards, latency bar
   chart, quality radar chart, and full results table with web search columns

### Dashboard vs CLI

Every feature available via CLI flags is accessible from the dashboard:

| CLI Flag | Dashboard Equivalent |
|---|---|
| `--models` | Model checkboxes |
| `--judge-model` | Judge dropdown |
| `--questions-file` | Questions source: Server file |
| `--web-search` | Web search checkbox |
| `--searxng-url` | Performance Settings panel |
| `--run-id` | Run ID text input |
| `--embedding-backend` | Ingestion form |
| `--streaming` | Ingestion streaming toggle |
| `--max-rows` | Ingestion max rows |

## REST API

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Dashboard HTML |
| GET | `/status` | System status (Ollama, ChromaDB, GPU) |
| GET | `/questions-files` | List available question files in data/ |
| GET | `/settings` | Current tunable settings |
| PATCH | `/settings` | Update runtime settings (in-memory) |
| GET | `/models/running` | Models currently loaded in VRAM |
| POST | `/models/warm` | Pre-load a model with keep_alive |
| POST | `/models/unload` | Evict a model from VRAM |
| POST | `/ingest` | Trigger dataset ingestion |
| GET | `/ingest/{id}/progress` | SSE stream for ingestion progress |
| POST | `/evaluate` | Trigger multi-model eval sweep |
| GET | `/evaluate/{run_id}/progress` | SSE stream for evaluation progress |
| GET | `/results` | List all evaluation runs |
| GET | `/results/{run_id}` | Get results for a specific run |

## Configuration

All settings can be overridden via environment variables prefixed with `RAG_`:

| Variable | Default | Description |
|---|---|---|
| `RAG_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API endpoint |
| `RAG_EMBEDDING_BACKEND` | `ollama` | `"ollama"` (GPU) or `"huggingface"` (CPU) |
| `RAG_EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model name |
| `RAG_EMBEDDING_FALLBACK` | `True` | Fall back to HuggingFace on Ollama failure |
| `RAG_EVAL_MODELS` | `llama3,mistral,phi3` | Models to evaluate |
| `RAG_JUDGE_MODEL` | `qwen3-coder:30b` | Judge model for quality scoring |
| `RAG_CHUNK_SIZE` | `512` | Text chunk size for ingestion |
| `RAG_CHUNK_OVERLAP` | `64` | Chunk overlap |
| `RAG_EMBED_BATCH_SIZE` | `32` | Embedding batch size |
| `RAG_RETRIEVAL_TOP_K` | `5` | Number of docs to retrieve |
| `RAG_MAX_RETRIES` | `1` | Max LangGraph self-correction loops |
| `RAG_WEB_SEARCH_ENABLED` | `False` | Enable SearXNG web search augmentation |
| `RAG_SEARXNG_BASE_URL` | `http://localhost:8888` | SearXNG API endpoint |
| `RAG_WEB_SEARCH_MAX_RESULTS` | `3` | Max web results to append per query |

## Web Search (SearXNG)

An optional **web search augmentation** can enrich ChromaDB retrieval with
live web results via a local SearXNG instance. This is useful when the
local dataset lacks coverage for a particular question.

### Data Privacy Guarantee

- **Outbound**: Only the question text is sent to SearXNG as a search query
- **Never sent**: No ChromaDB documents, no dataset content, no embeddings,
  no model answers
- **SearXNG runs locally** in Docker on the same machine, so even the
  question text stays on-premises

### How It Works

1. The `retrieve` node fetches top-k docs from ChromaDB (unchanged)
2. If `--web-search` is passed, a conditional edge routes to the `web_search` node
3. `web_search` sends `GET /search?q={question}&format=json` to SearXNG
4. Results are converted to the same `{"text", "score", "metadata"}` format
   and **appended** to the existing ChromaDB documents
5. `grade_documents` grades all documents (local + web) the same way —
   irrelevant web hits get filtered out
6. The `generate` node sees web results as additional context (system prompt
   still says "Answer ONLY using the provided context")

### Usage

```bash
# Default: local retrieval only
python scripts/evaluate.py --models mistral-small3.2 \
  --questions-file data/questions_legal.txt --run-id legal-v1

# With web search augmentation
python scripts/evaluate.py --models mistral-small3.2 \
  --questions-file data/questions_legal.txt --run-id legal-v1-web \
  --web-search

# Custom SearXNG URL
python scripts/evaluate.py --models mistral-small3.2 \
  --web-search --searxng-url http://localhost:9090 \
  --questions-file data/questions_legal.txt --run-id legal-v1-web
```

### Metrics Tracking

When web search is enabled, two additional fields appear in the results:

| Field | Description |
|---|---|
| `web_search_used` | `True` if web results survived grading and were used |
| `web_search_s` | Latency of the SearXNG API call (seconds) |

## Performance Tuning Checklist

1. **Set `OLLAMA_MAX_LOADED_MODELS=3`** in the Ollama Docker container
   (biggest single improvement — eliminates model swap overhead)
2. **Pre-warm all models** before evaluation using `curl` with `keep_alive`,
   or use the dashboard's "Warm All for Eval" button
3. **Verify GPU access** inside Docker (`docker exec ollama nvidia-smi`)
4. **Use `--streaming`** with `--max-rows` for large HuggingFace datasets
   to avoid downloading the entire dataset
5. **Keep `max_retries` low** (default 1) — aggressive self-correction
   degrades answer quality through query drift
6. **Uninstall `pynvml` shim** if you see a deprecation warning:
   `pip uninstall pynvml` (keep `nvidia-ml-py`)
7. **Use direct Ollama API** for embeddings (not the LangChain wrapper) —
   native batching via `/api/embed` with `"input": [...]` is 10-50x faster
8. **Set generous timeouts** (600s) for Ollama API calls to handle
   model loading delays on first request

## Project Structure

```
src/rag_eval/
  config.py          Central settings (Pydantic)
  cli.py             CLI entry points
  ingest/
    loader.py        HuggingFace dataset loading with schema normalisation
    chunker.py       Recursive text splitting
    embedder.py      Ollama/HuggingFace backends with retry and fallback
    indexer.py        ChromaDB upsert with backend-tagged collections
  retrieval/
    store.py         ChromaDB retriever (fallback disabled for safety)
  graph/
    state.py         GraphState TypedDict (includes original_question, web_search_enabled)
    nodes.py         LangGraph nodes (retrieve, web_search, grade, generate, checks)
    edges.py         Conditional routing with retry-aware logic + web search routing
    builder.py       Graph assembly
  eval/
    harness.py       Multi-model sweep with incremental persistence
    judge.py         LLM-as-judge scoring (faithfulness, relevancy, precision)
    gpu_monitor.py   NVML + nvidia-smi fallback for unified memory
    metrics.py       EvalResult dataclass
  api/
    app.py           FastAPI application
    routes.py        REST endpoints
  dashboard/
    index.html       Single-page dashboard with Chart.js
scripts/
  ingest.py          CLI wrapper for ingestion
  evaluate.py        CLI wrapper for evaluation
  serve.py           CLI wrapper for dashboard
data/
  questions_legal.txt  25 legal evaluation questions
```
