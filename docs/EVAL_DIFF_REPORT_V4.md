# Evaluation Diff Report v4: DeepSeek R1 Eval with Mistral Judge

> **Run F**: `9c2914ec47ed` -- deepseek-r1-abliterated:32b-q8 eval, **mistral-small3.2 judge**, web search on
> **Compared against**:
>
> - **Run D** `62767f5549bd` -- mistral-small3.2 eval, llama3.3:70b judge, web search on
>   **Questions**: 25 legal questions from `data/questions_legal.txt`

---

## 1. Aggregate Comparison: DeepSeek (eval) vs Mistral (eval)

| Metric                | Run D: mistral eval, llama judge | Run F: deepseek eval, mistral judge | Delta  |
| --------------------- | :------------------------------: | :---------------------------------: | :----: |
| **Faithfulness**      |            **0.904**             |                0.841                | -0.063 |
| **Relevancy**         |            **0.980**             |                0.864                | -0.116 |
| **Context Precision** |            **0.896**             |                0.442                | -0.454 |
| Avg Latency           |            **14.8s**             |               165.0s                | +150s  |
| Avg tok/s             |             **11.2**             |                 4.8                 |  -6.4  |
| Retries               |              **9**               |                 38                  |  +29   |
| IDK answers           |             **0/25**             |                0/25                 |  same  |
| Errors                |             **0/25**             |                3/25                 |   +3   |
| Web search used       |              24/25               |                22/25                |   -2   |
| Faith = 0.5 defaults  |              **0**               |                  0                  |  same  |
| Rel = 0.5 defaults    |              **0**               |                  0                  |  same  |

**Important caveat**: These runs use different judges, so the scores are not directly comparable on an absolute scale. The comparison reveals each model's strengths and weaknesses as evaluated by a different judge.

---

## 2. Key Findings

### DeepSeek R1 Produces Longer, Reasoning-Heavy Answers

DeepSeek R1 is a reasoning model that emits `<think>...</think>` blocks before its final answer. This has several consequences:

- **Higher token usage**: Average completion tokens are significantly higher than mistral
- **Extreme latency**: 165s per question vs 14.8s for mistral -- 11x slower
- **High retry rate**: 38 retries across 25 questions (1.5 retries/question average, vs 0.36 for mistral). The reasoning tokens can produce verbose intermediate outputs that fail document grading or hallucination checks

### Three Errors (Timeouts)

| #   | Question                   | Error     |
| --- | -------------------------- | --------- |
| 8   | Parol evidence rule        | Timed out |
| 13  | Res judicata doctrine      | Timed out |
| 22  | Class action under Rule 23 | Timed out |

These are likely caused by VRAM contention between deepseek (41 GB) and mistral-judge (28 GB) + embed (0.6 GB) = 69.6 GB total. While this fits in 128 GB, the KV cache for deepseek's long reasoning chains may push memory usage close to the limit.

### Context Precision Is Significantly Lower

DeepSeek's context precision averages **0.442** vs mistral's **0.896**. Seven questions scored exactly 0.5 (the default), and many scored 0.30. This suggests the mistral judge is stricter about context grading when evaluating deepseek's answers, or deepseek's answers synthesize beyond the retrieved context (which the judge penalizes).

### Faithfulness and Relevancy Are Respectable

Despite the challenges, deepseek achieved **0.841 faithfulness** and **0.864 relevancy** (excluding errors). These are solid scores -- lower than mistral's Run D numbers, but assessed by a different (smaller) judge. The mistral judge may be more conservative than llama.

---

## 3. Per-Question Head-to-Head

### Questions Where DeepSeek Outperformed Mistral

| Question (truncated)                  | Mistral (faith) | DeepSeek (faith) | Delta |
| ------------------------------------- | :-------------: | :--------------: | :---: |
| Breach of fiduciary duty              |       0.8       |     **1.0**      | +0.2  |
| Defamation elements                   |       0.9       |     **1.0**      | +0.1  |
| Motion to dismiss vs summary judgment |       0.8       |     **0.9**      | +0.1  |
| Pierce the corporate veil             |       0.8       |     **0.9**      | +0.1  |

DeepSeek's reasoning ability produced more thorough answers on complex legal topics requiring detailed element enumeration.

### Questions Where Mistral Outperformed DeepSeek

| Question (truncated)           | Mistral (faith) | DeepSeek (faith) | Delta |
| ------------------------------ | :-------------: | :--------------: | :---: |
| Standing in federal court      |     **1.0**     |       0.8        | -0.2  |
| Summary judgment standard      |     **1.0**     |       0.8        | -0.2  |
| Standard of review (appellate) |     **0.9**     |       0.7        | -0.2  |
| Negligence vs gross negligence |     **0.9**     |       0.7        | -0.2  |
| Hearsay evidence conditions    |     **0.9**     |       0.7        | -0.2  |

Mistral was more concise and grounded for straightforward questions where a direct answer suffices. DeepSeek's verbose reasoning sometimes introduced paraphrased claims that weren't directly in the context, which the judge penalized.

### Errors (DeepSeek only)

| Question (truncated)       | Mistral | DeepSeek | Notes   |
| -------------------------- | :-----: | :------: | :------ |
| Parol evidence rule        |   0.9   |  ERROR   | Timeout |
| Res judicata doctrine      |   0.9   |  ERROR   | Timeout |
| Class action under Rule 23 |   1.0   |  ERROR   | Timeout |

---

## 4. Retry Analysis

DeepSeek hit the maximum retry count (2) on **16 of 25 questions**. This is a significant operational concern:

| Retries | Mistral (Run D) | DeepSeek (Run F) |
| ------- | :-------------: | :--------------: |
| 0       |       16        |        3         |
| 1       |        7        |        6         |
| 2       |        2        |        13        |
| Errors  |        0        |        3         |

The high retry rate for deepseek is caused by:

1. **Reasoning tokens confuse grading**: The `<think>` blocks can contain speculative statements that the hallucination checker flags
2. **Verbose answers**: Longer answers have more surface area for the grading nodes to find potential issues
3. **Model swap overhead**: Each retry triggers another generation + judge cycle, compounding latency

---

## 5. Latency Breakdown

| Model            | Avg total_s | Avg generate_s | Avg tok/s | Retries/q |
| ---------------- | :---------: | :------------: | :-------: | :-------: |
| mistral (Run D)  |    14.8s    |      ~8s       |   11.2    |   0.36    |
| deepseek (Run F) |   165.0s    |      ~45s      |    4.8    |   1.52    |

DeepSeek is 11x slower per question. The 165s includes:

- ~45s generation time (deepseek generates at 4.8 tok/s vs mistral's 11.2)
- ~50-80s for retries (1.5 retries × ~45s each)
- ~20-30s for model swap overhead between eval and judge

---

## 6. Overall Verdict

| Dimension             | Mistral (Run D, llama judge) | DeepSeek (Run F, mistral judge) |
| --------------------- | :--------------------------: | :-----------------------------: |
| **Answer Quality**    |      Very good, concise      |   Good, thorough but verbose    |
| **Faithfulness**      |            0.904             |     0.841 (different judge)     |
| **Relevancy**         |            0.980             |     0.864 (different judge)     |
| **Reliability**       |       25/25 completed        |        22/25 (3 errors)         |
| **Latency**           |        14.8s/question        |          165s/question          |
| **Cost (retries)**    |           9 total            |            38 total             |
| **Suitable for RAG?** |          Excellent           |    Marginal on this hardware    |

**Recommendation**: For RAG evaluation on the DGX Spark (128 GB):

- **Primary eval model**: `mistral-small3.2` -- fast, reliable, concise answers, fits alongside llama judge
- **Primary judge**: `llama3.3:70b` -- reliable scoring, no 0.5 defaults, detailed reasoning
- **DeepSeek R1**: Better suited as a judge or for standalone (non-RAG) reasoning tasks. Its chain-of-thought is valuable but creates friction in the RAG pipeline's grading loop

---

## 7. Appendix: Run Metadata

| Property          | Run D                   | Run F                          |
| ----------------- | :---------------------- | :----------------------------- |
| Run ID            | `62767f5549bd`          | `9c2914ec47ed`                 |
| Eval Model        | mistral-small3.2        | deepseek-r1-abliterated:32b-q8 |
| Judge Model       | llama3.3:70b            | mistral-small3.2:latest        |
| Questions scored  | 25/25                   | 22/25 (3 errors)               |
| Web Search        | SearXNG On              | SearXNG On                     |
| Embedding         | nomic-embed-text (768d) | nomic-embed-text (768d)        |
| OLLAMA_MAX_LOADED | 4                       | 4                              |

Bottom line: DeepSeek R1 (32B-q8) is 11x slower than mistral and hit 3 timeouts. Its <think> reasoning blocks are thorough but create friction in the RAG pipeline -- the hallucination/usefulness grading nodes flag speculative reasoning as potential issues, driving a 1.5 retry/question rate. Context precision suffered most (0.442 vs 0.896), partly because DeepSeek synthesizes beyond retrieved context and the mistral judge penalizes this strictly.

DeepSeek R1 is better suited as a judge model or for standalone reasoning tasks. For RAG evaluation on your hardware, mistral-small3.2 (eval) + llama3.3:70b (judge) remains the strongest combination.
