# RAG Evaluation Platform

**Local RAG evaluation with LangGraph on NVIDIA DGX Spark (GB10)**

End-to-end evaluation of retrieval-augmented generation across multiple local
LLM models served by Ollama, with ChromaDB vector indexing and LLM-as-judge
quality scoring — fully offline, no cloud dependencies.

## Architecture

```
HF Dataset → Chunk → Embed (Ollama) → ChromaDB Index
                                            ↓
Question Bank → Model Loop → LangGraph RAG Pipeline → Metrics + Judge Scores
                                            ↓
                              FastAPI Dashboard (tables + charts)
```

The LangGraph pipeline is **self-correcting**: it grades retrieved documents,
checks for hallucination, evaluates answer usefulness, and retries with
query-rewriting when quality checks fail.

## Prerequisites

- **Ollama** running at `localhost:11434` (Docker or native)
- **Python 3.12+**
- Pull at least one embedding model and one chat model:

```bash
ollama pull nomic-embed-text
ollama pull llama3
ollama pull mistral
```

## Quick Start

```bash
# Create a virtual environment and install
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

# 1. Ingest a dataset
python scripts/ingest.py rajpurkar/squad_v2 --max-rows 500

# 2. Run evaluation across models
python scripts/evaluate.py --models llama3 mistral phi3

# 3. Start the dashboard
python scripts/serve.py
# Open http://localhost:8000
```

## CLI Reference

### Ingest

```bash
python scripts/ingest.py <dataset> [options]

Options:
  --split           HF split name (default: train)
  --text-field      Column with text to index (default: context)
  --id-field        Column for document IDs (default: id)
  --title-field     Column for titles in metadata (default: title)
  --subset          HF dataset config/subset
  --max-rows        Limit number of rows to ingest
  -v, --verbose     Debug logging
```

### Evaluate

```bash
python scripts/evaluate.py [options]

Options:
  --questions-file  Path to a text file with one question per line
  --models          Space-separated model names to evaluate
  --judge-model     Model to use as the LLM-as-judge
  --run-id          Custom run identifier
  -v, --verbose     Debug logging
```

### Dashboard

```bash
python scripts/serve.py [--host 0.0.0.0] [--port 8000]
```

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

| Variable                | Default                  | Description                     |
|-------------------------|--------------------------|---------------------------------|
| `RAG_OLLAMA_BASE_URL`   | `http://localhost:11434` | Ollama API endpoint             |
| `RAG_EMBEDDING_MODEL`   | `nomic-embed-text`       | Embedding model                 |
| `RAG_EVAL_MODELS`       | `llama3,mistral,phi3`    | Models to evaluate              |
| `RAG_JUDGE_MODEL`       | `qwen3-coder:30b`        | Judge model for scoring         |
| `RAG_CHUNK_SIZE`        | `512`                    | Text chunk size                 |
| `RAG_CHUNK_OVERLAP`     | `64`                     | Chunk overlap                   |
| `RAG_RETRIEVAL_TOP_K`   | `5`                      | Number of docs to retrieve      |
| `RAG_MAX_RETRIES`       | `3`                      | Max LangGraph retry loops       |

## LangGraph Pipeline

The evaluation graph follows this topology:

```
retrieve → grade_documents ──(relevant)──→ generate
                            └─(empty)───→ rewrite → retrieve

generate → check_hallucination ──(grounded)──→ check_usefulness
                                └─(not)──────→ generate

check_usefulness ──(useful)──→ END (record result)
                  └─(not)───→ rewrite → retrieve
```

Each node operates on a shared `GraphState` TypedDict that accumulates
question, documents, generation, latency breakdowns, and GPU snapshots.

## Evaluation Metrics

**Quality** (scored by LLM-as-judge):
- **Faithfulness** — Is every claim in the answer supported by context?
- **Answer Relevancy** — Does the answer address the question?
- **Context Precision** — Are retrieved documents relevant to the question?

**Performance**:
- End-to-end latency (seconds)
- Token throughput (tokens/sec)
- Prompt and completion token counts
- GPU unified memory usage (GB)
- GPU utilisation, temperature, power

Results are saved as CSV and JSON in `data/results/`.

## Project Structure

```
src/rag_eval/
  config.py          Central settings (Pydantic)
  cli.py             CLI entry points
  ingest/            Dataset loading, chunking, embedding, ChromaDB indexing
  retrieval/         ChromaDB retriever wrapper, optional reranker
  graph/             LangGraph state, nodes, edges, graph builder
  eval/              Harness, LLM-as-judge, GPU monitor, metrics dataclass
  api/               FastAPI app and REST routes
  dashboard/         Single-page HTML dashboard with Chart.js
```
