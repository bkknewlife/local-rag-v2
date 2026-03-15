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
                                                        ↓
                                          FastAPI Dashboard (tables + charts)
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
- **Python 3.12+**
- Pull at least one embedding model and one chat model:

```bash
ollama pull nomic-embed-text
ollama pull mistral-small3.2
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
retrieve → grade_documents ──(relevant)──→ generate
                            └─(empty)───→ rewrite → retrieve

generate → check_hallucination ──(grounded)──→ check_usefulness
                                └─(not)──────→ generate (retries++)

check_usefulness ──(useful)──→ END (record result)
                  └─(not)───→ rewrite → retrieve (retries++)
```

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

## REST API

| Method | Endpoint           | Description                          |
|--------|--------------------|--------------------------------------|
| GET    | `/`                | Dashboard HTML                       |
| GET    | `/status`          | System status (Ollama, ChromaDB, GPU)|
| POST   | `/ingest`          | Trigger dataset ingestion            |
| POST   | `/evaluate`        | Trigger multi-model eval sweep       |
| GET    | `/results`         | List all evaluation runs             |
| GET    | `/results/{run_id}`| Get results for a specific run       |

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

## Performance Tuning Checklist

1. **Set `OLLAMA_MAX_LOADED_MODELS=3`** in the Ollama Docker container
   (biggest single improvement — eliminates model swap overhead)
2. **Pre-warm all models** before evaluation using `curl` with `keep_alive`
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
    state.py         GraphState TypedDict (includes original_question)
    nodes.py         LangGraph nodes (retrieve, grade, generate, checks)
    edges.py         Conditional routing with retry-aware logic
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
