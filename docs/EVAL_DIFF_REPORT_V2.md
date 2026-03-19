# Evaluation Diff Report v2: Three-Run Comparison

> **Run A** (baseline, no web): `legal-multi-model` -- 5 models, ChromaDB only
> **Run B** (web search v1): `14b8585d1ffb` -- 2 models, SearXNG enabled
> **Run C** (web search v2): `81d96ab51cba` -- 2 models, SearXNG enabled, with logging & embedding filter fix
> **Judge Model**: `deepseek-r1-abliterated:32b-q8` (all runs)
> **Questions**: 25 legal questions from `data/questions_legal.txt`
> **Compared Models**: `llama3.3:70b` and `mistral-small3.2`

---

## 1. Aggregate Comparison (3 Runs)

### llama3.3:70b

| Metric                | A (no web) | B (web v1) | C (web v2) | A→C Delta | Verdict              |
|-----------------------|:----------:|:----------:|:----------:|:---------:|:---------------------|
| **Faithfulness**      | 0.792      | 0.787      | **0.802**  | +0.010    | Slight improvement   |
| **Relevancy**         | 0.590      | 0.767      | 0.618      | +0.028    | Improved from A; regressed from B |
| **Context Precision** | 0.587      | 0.715      | 0.687      | +0.100    | Improved from A; slight dip from B |
| Avg Latency (s)       | 35.0       | 42.0       | **34.3**   | -0.7      | Faster than both     |
| Avg tok/s             | 3.8        | 3.8        | 3.7        | -0.1      | Same                 |
| Total Retries         | 24         | 15         | 16         | -8        | Fewer loops          |
| "I don't know" count  | 8/25       | 0/25       | **1/25**   | -7        | Near-eliminated      |
| Web search used       | 0/25       | 25/25      | 25/25      | --        | All questions        |

### mistral-small3.2

| Metric                | A (no web) | B (web v1) | C (web v2) | A→C Delta | Verdict              |
|-----------------------|:----------:|:----------:|:----------:|:---------:|:---------------------|
| **Faithfulness**      | 0.804      | 0.814      | **0.830**  | +0.026    | Best of all 3 runs   |
| **Relevancy**         | 0.674      | 0.706      | 0.648      | -0.026    | Regressed from A & B |
| **Context Precision** | 0.600      | 0.717      | 0.712      | +0.112    | Strong improvement   |
| Avg Latency (s)       | 12.4       | 14.3       | 21.3       | +8.9      | Slower (see below)   |
| Avg tok/s             | 9.5        | 11.3       | 11.5       | +2.0      | Faster generation    |
| Total Retries         | 22         | 10         | 9          | -13       | Fewest retries       |
| "I don't know" count  | 6/25       | 1/25       | **2/25**   | -4        | Near-eliminated      |
| Web search used       | 0/25       | 24/25      | 23/25      | --        | Most questions       |

---

## 2. Key Finding: Faithfulness Improved, Relevancy Regressed

### Faithfulness is now the best across all three runs

- **llama3.3:70b**: 0.792 → 0.787 → **0.802** (up from both prior runs)
- **mistral-small3.2**: 0.804 → 0.814 → **0.830** (best yet)

The -1.0 anomaly from Run B (judge scoring failure on "probable cause") was eliminated in Run C -- that question now scores 1.0 faithfulness. This alone accounts for much of the aggregate improvement.

The 0.5 judge defaults are still present but slightly less frequent:
- llama: 6 questions at 0.5 (down from 10 in Run B)
- mistral: 4 questions at 0.5 (down from 5 in Run B)

### Relevancy dropped compared to Run B

This is the notable regression in Run C:

| Model            | B (web v1) | C (web v2) | Delta  |
|------------------|:----------:|:----------:|:------:|
| llama3.3:70b     | 0.767      | 0.618      | -0.149 |
| mistral-small3.2 | 0.706      | 0.648      | -0.058 |

The judge appears to be scoring relevancy more conservatively in Run C. Many questions that scored 0.85-1.0 in Run B now score 0.5 in Run C. This may be due to:

1. **Judge model state**: DeepSeek R1 is a reasoning model that can produce inconsistent scores across runs, especially when reasoning tokens vary
2. **Context ordering**: SearXNG results may have returned in a different order, changing the context presentation to the judge
3. **The 0.5 default fallback**: Some relevancy scores of exactly 0.5 with empty reasoning suggest parse failures

---

## 3. llama3.3:70b -- Per-Question Faithfulness Changes (B → C)

### Significant Improvements (> +0.1)

| Question (truncated)                               | B     | C     | Delta  | Analysis |
|----------------------------------------------------|:-----:|:-----:|:------:|:---------|
| Probable cause for search warrant                  | -1.0  | **1.0** | +2.0 | Fixed: judge failure in B now scores perfectly |
| Qualified immunity                                 | 0.50  | **1.0** | +0.50 | Was 0.5 default; now properly scored |
| Double jeopardy attach                             | 0.50  | **1.0** | +0.50 | Same fix pattern |
| Negligence vs gross negligence                     | 0.50  | **0.95**| +0.45 | Was 0.5 default; now accurate |
| Defamation elements                                | 0.50  | **0.90**| +0.40 | Was 0.5 default; now accurate |
| Standing in federal court                          | 0.50  | **0.85**| +0.35 | Same pattern |
| Summary judgment standard                          | 0.50  | **0.80**| +0.30 | Was 0.5 default; now scored |
| Standard of review (appellate)                     | 0.50  | **0.90**| +0.40 | Was 0.5 default; now accurate |
| Parol evidence rule                                | 0.80  | **1.0** | +0.20 | Minor improvement |

### Significant Regressions (< -0.1)

| Question (truncated)                               | B     | C     | Delta  | Analysis |
|----------------------------------------------------|:-----:|:-----:|:------:|:---------|
| Fair use (copyright)                               | 1.0   | **0.50**| -0.50 | Model answered "I don't know" -- only IDK in Run C |
| Motion to dismiss vs summary judgment              | 1.0   | **0.50**| -0.50 | Correct answer but judge defaulted to 0.5 |
| Actual malice vs negligence                        | 0.90  | **0.50**| -0.40 | Same 0.5 default issue |
| Res judicata                                       | 1.0   | **0.65**| -0.35 | Answer correct; judge penalized mixed-source synthesis |
| Duress (contract voided)                           | 1.0   | **0.70**| -0.30 | Similar: web context diluted grounding |
| Class action Rule 23                               | 1.0   | **0.75**| -0.25 | Answer shortened; missing some details from B |
| Adverse possession elements                        | 0.90  | **0.50**| -0.40 | Correct answer but judge defaulted to 0.5 |
| Hearsay evidence                                   | 1.0   | **0.80**| -0.20 | Minor; answer was more verbose in C |
| Exclusionary rule                                  | 1.0   | **0.85**| -0.15 | Minor rounding |

### The Fair Use "I Don't Know" (Run C Only)

```
answer: "I don't know. The context provides general information about fair use,
but it does not explicitly state what constitutes fair use under copyright law."
web_search_used: true
context_scores: 0.000, 0.000, 0.000
```

All 3 retrieved documents have score 0.000 (from SearXNG), meaning the grader rejected all web results as irrelevant. The model saw only ungraded web snippets and correctly refused to answer. In Run B, the same web snippets were accepted (different SearXNG results or grading variance), allowing the model to compose an answer.

---

## 4. mistral-small3.2 -- Per-Question Faithfulness Changes (B → C)

### Significant Improvements (> +0.1)

| Question (truncated)                               | B     | C     | Delta  | Analysis |
|----------------------------------------------------|:-----:|:-----:|:------:|:---------|
| Motion to dismiss vs summary judgment              | 0.50  | **1.0** | +0.50 | Was 0.5 default; now "I don't know" gets full faith score |
| Actual malice vs negligence                        | 0.50  | **0.95**| +0.45 | Was 0.5 default; now properly graded |
| Due process (14th amendment)                       | 0.50  | **0.90**| +0.40 | Was 0.5 default; now scored |
| Standard of review (appellate)                     | 0.50  | **0.90**| +0.40 | Same pattern |
| Summary judgment standard                          | 0.50  | **0.80**| +0.30 | Improved from 0.5 default |
| Defamation elements                                | 0.70  | **1.0** | +0.30 | Fully grounded answer |

### Significant Regressions (< -0.1)

| Question (truncated)                               | B     | C     | Delta  | Analysis |
|----------------------------------------------------|:-----:|:-----:|:------:|:---------|
| Probable cause for search warrant                  | 1.0   | **0.50**| -0.50 | Judge defaulted to 0.5 with empty reasoning |
| Adverse possession elements                        | 1.0   | **0.50**| -0.50 | Same pattern |
| Fair use (copyright)                               | 1.0   | **0.60**| -0.40 | Shorter answer; judge penalized brevity |
| Stare decisis                                      | 0.85  | **0.50**| -0.35 | 0.5 default |
| Corporate veil                                     | 1.0   | **0.75**| -0.25 | Minor grounding penalty |
| Vagueness doctrine                                 | 1.0   | **0.80**| -0.20 | Minor |

### Mistral's Persistent "Summary Judgment" Failure

Both Run B and Run C show mistral answering "I don't know" to this question:

| Run | web_search_used | context_scores      | Retries |
|-----|:---------------:|:--------------------|:-------:|
| B   | false           | 0.835, 0.817        | 2       |
| C   | false           | 0.835, 0.817        | 2       |

Web search was **not triggered** because the ChromaDB retrieval scores (0.835, 0.817) were above the quality threshold. The route_after_retrieve edge decided the context was "good enough." But the context only covers state-court rules (Vermont), not federal court.

The same question with llama3.3:70b uses web search and gets the correct federal Rule 56 standard.

---

## 5. Latency Analysis

| Model            | A (no web) | B (web v1) | C (web v2) | Notes |
|------------------|:----------:|:----------:|:----------:|:------|
| llama3.3:70b     | 35.0s      | 42.0s      | **34.3s**  | Fastest! Model pre-warmed and fewer retries |
| mistral-small3.2 | 12.4s      | 14.3s      | **21.3s**  | Slower due to longer generation times |

**llama latency improved** in Run C despite using web search. Contributing factors:
- The -1.0 anomaly question in B took 212s; in C it completed in 23s
- Model pre-warming was consistent
- Average retry count comparable (15 vs 16)

**mistral latency increased** by ~7s. The generation times in Run C are longer (e.g., parol evidence: 23s vs 9.5s in B). This may be due to:
- GPU thermal throttling (temps 69-85°C in C vs 69-84°C in B)
- Longer answers in some cases (parol evidence: 297 tokens vs 122 in B)
- VRAM contention from concurrent models

---

## 6. Judge Scoring Reliability: The 0.5 Default Problem Persists

Across Run C, many scores land at exactly 0.5 with empty reasoning. This is the fallback when the judge model fails to produce parseable JSON:

| Model            | Faith=0.5 | Rel=0.5 | CtxP=0.5 |
|------------------|:---------:|:-------:|:--------:|
| llama3.3:70b     | 6/25      | 15/25   | 7/25     |
| mistral-small3.2 | 4/25      | 14/25   | 4/25     |

**Relevancy is the most affected metric** -- 58% of all scores are exactly 0.5. This explains why the aggregate relevancy appears to regress from Run B: it's not that answers are less relevant, but that the judge is failing to score them properly.

### Why relevancy is hardest for the judge

The relevancy prompt asks: "Is this answer useful for the question?" The judge must evaluate the complete answer against the question, which is a more subjective judgment than faithfulness (context grounding check). With mixed ChromaDB + web contexts, the DeepSeek R1 model appears to spend too many reasoning tokens on its internal chain-of-thought and then fails to emit a clean JSON response.

---

## 7. Run-Over-Run Trend Summary

| Dimension                | A → B             | B → C              | Net A → C          |
|--------------------------|:------------------|:-------------------|:-------------------|
| **Faithfulness**         | Flat (0.5 defaults) | Improved (+0.02)   | **Improved**       |
| **Relevancy**            | Strong +0.18       | Regressed -0.15    | Mixed (judge noise)|
| **Context Precision**    | Strong +0.13       | Slight dip -0.03   | **Improved**       |
| **IDK Answers**          | Eliminated (8→0)   | 1 new IDK          | **Major improvement** |
| **Latency (llama)**      | +7s                | **-8s**            | **Improved**       |
| **Latency (mistral)**    | +2s                | +7s                | Slower             |
| **Retries**              | -9                 | +1                 | **Fewer overall**  |
| **Judge reliability**    | 0.5 defaults present | Slightly better  | Still an issue     |

---

## 8. Recommendations

### Immediate

1. **Re-run with `llama3.3:70b` as judge model** -- Its 70B parameter count handles long mixed contexts better than the 32B DeepSeek R1 distillation, which struggles with relevancy scoring
2. **Fix the "fair use" IDK regression** -- The document grader rejected all 3 SearXNG results (score 0.000). Consider lowering the grading threshold for web documents or bypassing grading for web results
3. **Fix the summary-judgment routing for mistral** -- Force web search when `web_search_enabled=True` regardless of retrieval scores, or add keyword matching as a secondary check

### Long-term

4. **Separate faithfulness scoring** for local (ChromaDB) and web sources to isolate which context type the model is grounding on
5. **Log raw judge responses** (now enabled via the logging implementation) to analyze exactly when and why the judge defaults to 0.5
6. **Consider running judge scoring in a separate pass** with a longer timeout and no other models competing for VRAM

---

## 9. Appendix: Run Metadata

| Property          | Run A               | Run B               | Run C               |
|-------------------|:--------------------|:--------------------|:--------------------|
| Run ID            | `legal-multi-model` | `14b8585d1ffb`      | `81d96ab51cba`      |
| Judge             | deepseek-r1-abliterated:32b-q8 | deepseek-r1-abliterated:32b-q8 | deepseek-r1-abliterated:32b-q8 |
| Web Search        | Off                 | SearXNG On          | SearXNG On          |
| Embedding         | nomic-embed-text (768d) | nomic-embed-text (768d) | nomic-embed-text (768d) |
| Logging           | No                  | No                  | **Yes (per-run .log)** |
| Embed filter fix  | No                  | No                  | **Yes**              |
| judge_model in JSON | No                | No                  | **Yes**              |
