# Evaluation Diff Report v5: Qwen3:30b with LLaMA Judge

> **Run G**: `ec8be86cacc1` -- qwen3:30b eval, **llama3.3:70b judge**, web search on
> **Compared against**:
>   - **Run D** `62767f5549bd` -- mistral-small3.2 eval, llama3.3:70b judge, web search on
>   - **Run F** `9c2914ec47ed` -- deepseek-r1-abliterated:32b-q8 eval, mistral-small3.2 judge, web search on
>   - **Run B** `legal-multi-model` (backup/5 models) -- qwen3:30b eval, deepseek-r1 judge, **no web search**
> **Questions**: 25 legal questions from `data/questions_legal.txt`

---

## 1. Qwen3 Before vs After: The Impact of Web Search + Better Judge

This is the most valuable comparison: same eval model (qwen3:30b), different judge and web search availability.

| Metric                | Run B: qwen3, deepseek judge, no web | Run G: qwen3, llama judge, web search | Delta |
|-----------------------|:------------------------------------:|:-------------------------------------:|:-----:|
| **Faithfulness**      | 0.707                                | **0.808**                             | **+0.101** |
| **Relevancy**         | 0.642                                | **0.840**                             | **+0.198** |
| **Context Precision** | 0.573                                | **0.863**                             | **+0.290** |
| Avg Latency           | **62.5s**                            | 64.2s                                 | +1.7s |
| Avg tok/s             | 50.7                                 | **70.4**                              | +19.7 |
| Retries               | 17                                   | **11**                                | -6    |
| IDK answers           | 7/25                                 | **3/25**                              | -4    |
| Errors                | 0/25                                 | 0/25                                  | same  |
| 0.5 defaults (faith)  | 13                                  | **0**                                 | **-13** |

### Key Takeaway

The dramatic improvement comes from two factors:
1. **LLaMA 70b judge** produces real scores instead of 0.5 defaults (13 -> 0 defaults)
2. **Web search** eliminated 4 of the 7 "I don't know" answers by providing supplementary context when ChromaDB retrieval was insufficient

---

## 2. Three-Way Aggregate Comparison (Same Judge: LLaMA)

Runs D and G both use llama3.3:70b as judge and have web search enabled, making them directly comparable.

| Metric                | Run D: mistral eval | Run G: qwen3 eval | Delta |
|-----------------------|:-------------------:|:------------------:|:-----:|
| **Faithfulness**      | **0.904**           | 0.808              | -0.096|
| **Relevancy**         | **0.980**           | 0.840              | -0.140|
| **Context Precision** | 0.896               | **0.863**          | -0.033|
| Avg Latency           | **14.8s**           | 64.2s              | +49.4s|
| Avg tok/s             | 11.2                | **70.4**           | +59.2 |
| Retries               | **9**               | 11                 | +2    |
| IDK answers           | **0/25**            | 3/25               | +3    |
| Errors                | **0/25**            | 0/25               | same  |
| Web search used       | 24/25               | 21/25              | -3    |

### Observations

- Mistral still leads on faithfulness and relevancy by a clear margin
- Qwen3 generates at **70.4 tok/s** (6.3x faster than mistral's 11.2) but total latency is 4.3x higher due to much longer outputs (qwen3 emits `<think>` reasoning blocks)
- Context precision is nearly identical -- both models retrieve and use context well
- Qwen3 has 3 IDK answers where mistral answered all 25

---

## 3. Full Model Leaderboard (All Runs with Web Search)

| Rank | Model | Judge | Faith | Relev | CtxPrec | Latency | IDK | Errors | Retries |
|:----:|-------|-------|:-----:|:-----:|:-------:|:-------:|:---:|:------:|:-------:|
| 1 | **mistral-small3.2** | llama3.3:70b | **0.904** | **0.980** | **0.896** | **14.8s** | 0 | 0 | 9 |
| 2 | **qwen3:30b** | llama3.3:70b | 0.808 | 0.840 | 0.863 | 64.2s | 3 | 0 | 11 |
| 3 | deepseek-r1:32b-q8 | mistral-small3.2 | 0.841 | 0.864 | 0.442 | 165.0s | 0 | 3 | 38 |

*Note: DeepSeek (Run F) used a different judge, so its scores are not directly comparable to the other two.*

---

## 4. Qwen3 "I Don't Know" Analysis

Three questions where qwen3 refused to answer:

| # | Question | Qwen3 (G) | Mistral (D) | Notes |
|---|----------|:----------:|:-----------:|:------|
| 14 | Actual malice vs negligence (defamation) | **IDK** f=0.0 r=0.0 | f=0.9 r=1.0 | Web search was used but qwen3 still refused. The model's thinking likely decided it couldn't be sufficiently certain |
| 22 | Class action under Rule 23 | **IDK** f=0.0 r=0.0 | f=1.0 r=0.8 | Web search was used. Despite good context (c=1.0), qwen3 refused to commit |
| 24 | Exclusionary rule and exceptions | **IDK** f=0.0 r=0.0 | f=0.8 r=0.9 | Web search was NOT used. ChromaDB retrieval may have been insufficient and no web fallback triggered |

**Pattern**: Qwen3's `<think>` reasoning can lead to over-caution. When the model's internal chain-of-thought identifies uncertainty, it defaults to "I don't know" rather than providing a partial answer. This is arguably more honest but scores poorly in the evaluation framework. Question 24 is notable -- it's one of only 4 questions where web search was *not* used, and the local retrieval context alone was insufficient.

---

## 5. Per-Question Head-to-Head: Qwen3 (G) vs Mistral (D)

### Questions Where Qwen3 Matched or Outperformed Mistral

| Question (truncated)                        | Mistral (D) faith | Qwen3 (G) faith | Delta |
|--------------------------------------------|:-----------------:|:----------------:|:-----:|
| Summary judgment standard                  | 1.0               | **1.0**          | 0.0   |
| Probable cause for search warrant          | 0.9               | **1.0**          | +0.1  |
| Elements of defamation                     | 0.9               | **1.0**          | +0.1  |
| Standing in federal court                  | 1.0               | **1.0**          | 0.0   |
| Res judicata doctrine                      | 0.9               | **1.0**          | +0.1  |
| Fair use under copyright                   | 1.0               | **1.0**          | 0.0   |
| Due process (14th Amendment)               | 0.9               | **1.0**          | +0.1  |
| Adverse possession elements                | 1.0               | **1.0**          | 0.0   |
| Pierce the corporate veil                  | 0.8               | **0.9**          | +0.1  |
| Contract voided for duress                 | 0.8               | **0.9**          | +0.1  |

Qwen3 excels at comprehensive, well-structured legal analysis. On 10 questions it matched or beat mistral on faithfulness.

### Questions Where Mistral Clearly Won

| Question (truncated)                        | Mistral (D) faith | Qwen3 (G) faith | Delta |
|--------------------------------------------|:-----------------:|:----------------:|:-----:|
| Actual malice vs negligence                | **0.9**           | 0.0 (IDK)       | -0.9  |
| Class action under Rule 23                 | **1.0**           | 0.0 (IDK)       | -1.0  |
| Exclusionary rule exceptions               | **0.8**           | 0.0 (IDK)       | -0.8  |
| Double jeopardy                            | **1.0**           | 0.8              | -0.2  |

The 3 IDK answers are the primary source of qwen3's aggregate score gap.

---

## 6. Latency Profile

| Model | Fastest Q | Slowest Q | Median | P95 |
|-------|:---------:|:---------:|:------:|:---:|
| mistral (D) | 6.4s | 44.2s | 12.3s | ~40s |
| qwen3 (G) | 49.5s | 112.3s | 58.7s | ~85s |
| deepseek (F) | ~40s | ~400s | ~130s | ~350s |

Qwen3's latency is fairly consistent (49-85s range) due to its reasoning overhead. The 112s outlier on Q1 (summary judgment) was likely caused by a long `<think>` block exploring multiple legal standards.

---

## 7. Qwen3's Strengths and Weaknesses for RAG

### Strengths
- **Highest throughput**: 70.4 tok/s (fast generation once started)
- **Strong on complex questions**: Outperformed mistral on questions requiring multi-element legal analysis
- **Zero errors**: All 25 questions completed (no timeouts)
- **Good context precision**: 0.863 (only 0.033 behind mistral), suggesting good retrieval alignment

### Weaknesses
- **Over-cautious**: 3 IDK answers where it had sufficient context but chose not to answer
- **Higher latency**: 64s average due to `<think>` reasoning tokens before final answer
- **Slightly lower faithfulness**: 0.808 vs 0.904 -- the reasoning process occasionally paraphrases beyond retrieved context

---

## 8. Updated Recommendations

| Use Case | Best Model | Judge | Why |
|----------|-----------|-------|-----|
| **Production RAG** | mistral-small3.2 | llama3.3:70b | Fastest, most reliable, highest scores |
| **Deep legal analysis** | qwen3:30b | llama3.3:70b | More thorough reasoning, but slower and sometimes refuses |
| **Avoid for RAG eval** | deepseek-r1:32b-q8 | any | Too slow, timeout-prone, excessive retries |
| **Best judge** | -- | llama3.3:70b | Eliminates 0.5 defaults, consistent scoring |

### Potential Improvements for Qwen3
1. **Tune the system prompt** to be less conservative about uncertainty
2. **Increase `max_retries` to 2** specifically for qwen3 to give it more chances to commit to an answer
3. **Strip `<think>` blocks** before passing answers to the judge to avoid penalizing reasoning tokens

---

## 9. Appendix: Run Metadata

| Property          | Run B (old)            | Run D                  | Run F                  | Run G (new)            |
|-------------------|:-----------------------|:-----------------------|:-----------------------|:-----------------------|
| Run ID            | `legal-multi-model`    | `62767f5549bd`         | `9c2914ec47ed`         | `ec8be86cacc1`         |
| Eval Model        | qwen3:30b              | mistral-small3.2       | deepseek-r1:32b-q8     | qwen3:30b              |
| Judge Model       | deepseek-r1 (unknown)  | llama3.3:70b           | mistral-small3.2       | llama3.3:70b           |
| Web Search        | Off                    | SearXNG On             | SearXNG On             | SearXNG On             |
| Questions scored  | 25/25                  | 25/25                  | 22/25 (3 errors)       | 25/25                  |
| Embedding         | nomic-embed-text (768d)| nomic-embed-text (768d)| nomic-embed-text (768d)| nomic-embed-text (768d)|
| OLLAMA_MAX_LOADED | 3                      | 4                      | 4                      | 4                      |
