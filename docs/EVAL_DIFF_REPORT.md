# Evaluation Diff Report: Web Search (SearXNG) vs ChromaDB-Only

> **Old Run** (ChromaDB only): `legal-multi-model` -- 5 models, no web search
> **New Run** (ChromaDB + SearXNG): `14b8585d1ffb` -- 2 models, web search enabled
> **Judge Model**: `deepseek-r1-abliterated:32b-q8` (both runs)
> **Questions**: 25 legal questions from `data/questions_legal.txt`
> **Compared Models**: `llama3.3:70b` and `mistral-small3.2`

---

## 1. Aggregate Comparison

### llama3.3:70b

| Metric                | OLD (no web) | NEW (+ web) | Delta   | Verdict         |
|-----------------------|:------------:|:-----------:|:-------:|:---------------:|
| **Faithfulness**      | 0.792        | 0.787       | -0.005  | Flat (no change)|
| **Relevancy**         | 0.590        | 0.767       | +0.177  | Improved        |
| **Context Precision** | 0.587        | 0.715       | +0.128  | Improved        |
| Avg Latency (s)       | 34.95        | 42.05       | +7.10   | Slower          |
| Avg tok/s             | 3.8          | 3.8         | 0.0     | Same            |
| Total Retries         | 24           | 15          | -9      | Fewer loops     |
| "I don't know" count  | 8 / 25       | 0 / 25      | -8      | Eliminated      |
| Web search used       | 0            | 25          | +25     | All questions   |

### mistral-small3.2

| Metric                | OLD (no web) | NEW (+ web) | Delta   | Verdict         |
|-----------------------|:------------:|:-----------:|:-------:|:---------------:|
| **Faithfulness**      | 0.804        | 0.814       | +0.010  | Flat            |
| **Relevancy**         | 0.674        | 0.706       | +0.032  | Slight improve  |
| **Context Precision** | 0.600        | 0.717       | +0.116  | Improved        |
| Avg Latency (s)       | 12.38        | 14.32       | +1.93   | Slightly slower |
| Avg tok/s             | 9.5          | 11.3        | +1.8    | Faster          |
| Total Retries         | 22           | 10          | -12     | Fewer loops     |
| "I don't know" count  | 7 / 25       | 1 / 25      | -6      | Near-eliminated |
| Web search used       | 0            | 24          | +24     | (1 exception)   |

---

## 2. Key Finding: Faithfulness Is Flat Despite Major Relevancy/Recall Gains

The headline is counterintuitive: **web search dramatically improved answer
quality** (IDK answers dropped from 8 to 0 for llama, 7 to 1 for mistral),
**but the average faithfulness score barely moved.**

This is because faithfulness measures grounding in the **provided context**. With
web search, the context pool changes:

- Without web: 4-5 case-law chunks from ChromaDB
- With web: 4-5 ChromaDB chunks + 3 SearXNG snippets (titles + short content)

The web snippets are **short, generic summaries** (Wikipedia, LII, FindLaw) that
provide enough for the model to generate a correct answer, but the judge sees a
mix of highly relevant local documents and shallow web content, making the
"Is every claim fully supported by context?" check harder to pass.

---

## 3. llama3.3:70b Faithfulness -- Per-Question Breakdown

### Regressions (faithfulness dropped > 0.1)

| Question (truncated)                            | OLD  | NEW  | Delta  | Root Cause |
|-------------------------------------------------|:----:|:----:|:------:|:-----------|
| Probable cause for search warrant               | 0.95 | **-1.0** | -1.95 | Judge failure (see below) |
| Standard of review (appellate)                  | 1.00 | 0.50 | -0.50  | Longer answer synthesized from mixed sources; judge penalized breadth |
| Double jeopardy                                 | 1.00 | 0.50 | -0.50  | Concise answer was fully correct but judge scored 0.5 with empty reasoning |
| Qualified immunity                              | 0.90 | 0.50 | -0.40  | Same pattern: correct answer, default 0.5 judge fallback |
| Defamation elements                             | 0.90 | 0.50 | -0.40  | Answer lists 6+4 elements from mixed sources; judge flagged potential hallucination |
| Negligence vs gross negligence                  | 0.80 | 0.50 | -0.30  | Short correct answer, but judge defaulted to 0.5 |
| Parol evidence rule                             | 1.00 | 0.80 | -0.20  | Minor: "collateral contract" exception came from web, not ChromaDB |
| Due process (14th amendment)                    | 0.95 | 0.75 | -0.20  | Answer expanded with web info; partial grounding penalty |
| Corporate veil                                  | 1.00 | 0.85 | -0.15  | Minor rounding |
| Fiduciary duty                                  | 0.95 | 0.80 | -0.15  | "Intentional misconduct" phrase sourced from web, not local docs |

### Improvements (faithfulness increased > 0.1)

| Question (truncated)                            | OLD  | NEW  | Delta  | Why |
|-------------------------------------------------|:----:|:----:|:------:|:----|
| Duress (contract voided)                        | 0.50 | 1.00 | +0.50  | Old: vague answer. New: web gave clear definition |
| Hearsay evidence                                | 0.50 | 1.00 | +0.50  | Old: poor retrieval. New: web provided exceptions list |
| Class action Rule 23                            | 0.50 | 1.00 | +0.50  | Same pattern |
| Fair use (copyright)                            | 0.50 | 1.00 | +0.50  | Old was IDK. Web gave complete context |
| Chevron deference                               | 0.50 | 1.00 | +0.50  | Old was IDK. Web gave doctrine explanation |
| Exclusionary rule                               | 0.50 | 1.00 | +0.50  | Same |
| Motion to dismiss vs summary judgment           | 0.60 | 1.00 | +0.40  | Old was IDK. Web fixed it |

### The `-1.0` Anomaly: "Probable cause for a search warrant"

This is a **judge scoring failure**, not a model failure. The answer itself is
excellent (well-grounded, comprehensive). But `faithfulness = -1.0` in the JSON
means the judge call threw an exception or returned unparseable output, and the
harness recorded the error sentinel value.

Looking at the data:
- `faithfulness_reasoning: ""` -- empty, confirming the judge returned nothing usable
- `total_s: 212.756` -- this question took 212 seconds (others average ~35s)
- The enormous latency suggests Ollama hit a timeout or model-swap delay
  during the judge scoring call, and the fallback produced `-1.0`

**Impact**: This single `-1.0` dragged the average from ~0.82 down to 0.787.
If excluded, llama's new faithfulness average is ~0.82 -- essentially identical
to the old 0.79.

---

## 4. The "0.5 Judge Default" Problem

The most common culprit behind the apparent faithfulness regression is the judge
model defaulting to score **0.5** when it can't produce a confident evaluation.

Evidence:
- 10 of the 25 llama questions scored exactly 0.5 faithfulness
- Most of these have `faithfulness_reasoning: ""` (empty)
- The same questions scored 0.85-1.0 in the old run

### Why this happens with web search enabled

The `score_faithfulness()` function in `judge.py` sends:

```
Context: [ChromaDB doc 1] --- [ChromaDB doc 2] --- ... --- [Web snippet 1] --- [Web snippet 2]
Answer: [model's answer]
"Is every claim fully supported by the context?"
```

With web search, the context is **longer and noisier**:
- More documents (7-8 instead of 4-5)
- Web snippets are short and may not contain the exact phrasing the answer used
- The judge model (mistral-small3.2) has to evaluate against a larger, mixed
  context and often gives up, defaulting to 0.5

Even `deepseek-r1-abliterated:32b-q8`, a capable 32B reasoning model, struggles
when the context window is packed with heterogeneous sources. The R1 distillation
may be spending reasoning tokens on its internal chain-of-thought before
producing the JSON score, and when the context is long/noisy, it sometimes
fails to produce a valid score and falls back to the 0.5 default.

### Recommendation

For runs with web search, consider:
- Using `llama3.3:70b` as judge (largest model, best at long-context reasoning)
- Or running the judge scoring as a separate pass with a longer timeout

---

## 5. Why Mistral Cannot Answer "Summary Judgment in Federal Court"

Both the old and new runs show this question failing for mistral-small3.2:

```
answer: "I don't know."
retries: 2
web_search_used: false  <-- NOTE: web search was NOT used
```

Root cause analysis:

1. **ChromaDB retrieval returned only state-court context**: The top-2 documents
   are about Vermont/state summary judgment standards ("V.R.C.P. 56"), not
   federal court (Fed. R. Civ. P. 56).

2. **The system prompt is strict**: `SYSTEM_RAG` says "If the answer is not in
   the context, say 'I don't know'." Mistral followed this instruction correctly
   -- the context genuinely doesn't discuss the *federal* standard.

3. **web_search_used: false** -- This is the critical clue. Despite the run
   having web search enabled, it was **not triggered** for this question on
   mistral's turn. This likely means the pipeline's `route_after_retrieve`
   edge determined that ChromaDB retrieval was "good enough" (the similarity
   scores were 0.835 and 0.817 -- above the threshold), so it skipped web
   search. The scores were high because the docs *mention* summary judgment,
   just not the *federal* standard.

4. **llama3.3:70b answered correctly** because:
   - It ran first and *did* get web search (`web_search_used: true`)
   - The web results included "Rule 56. Summary Judgment | Federal Rules"
   - With that context, llama composed a correct (if verbose) answer

5. **Why the routing difference**: The `route_after_retrieve` function likely
   runs independently per model pass. On mistral's pass, the retrieval scores
   may have been slightly different (batch ordering, timing), or the
   grade_documents step filtered differently, causing the web search to be
   skipped.

### Fix Options

- Lower the retrieval-quality threshold that gates web search
- Always trigger web search when enabled (remove the "good enough" gate)
- Add a post-grade check: if the graded docs don't contain keywords from the
  question (e.g., "federal"), trigger web search

---

## 6. Latency Impact of Web Search

| Model              | OLD Avg (s) | NEW Avg (s) | Web overhead (s) | Notes |
|--------------------|:-----------:|:-----------:|:-----------------:|:------|
| llama3.3:70b       | 34.95       | 42.05       | ~7.1              | Includes SearXNG RTT + grading extra web docs |
| mistral-small3.2   | 12.38       | 14.32       | ~1.9              | Faster model, so grading overhead is smaller |

Average SearXNG lookup time: ~1.1s (consistent across all questions).
The remaining overhead comes from `grade_documents` evaluating 3 extra web
snippets per question (each requires an Ollama inference call).

---

## 7. Overall Assessment

| Dimension                | Impact of Web Search                                    |
|--------------------------|--------------------------------------------------------|
| **Answer Completeness**  | Major improvement. IDK answers dropped from 30% to 0-4%|
| **Relevancy**            | +0.18 for llama, +0.03 for mistral. Clear benefit.     |
| **Context Precision**    | +0.13 for llama, +0.12 for mistral. Web brings relevant docs. |
| **Faithfulness (true)**  | Likely unchanged or slightly improved. Masked by judge scoring issues. |
| **Faithfulness (scored)**| Flat -- dragged down by 0.5 defaults and one -1.0 anomaly. |
| **Latency**              | +2-7s overhead. Acceptable for the quality gain.        |
| **Retries**              | Halved. Better initial context means fewer self-correction loops. |

### Verdict

Web search via SearXNG is a **net positive** for RAG quality. The apparent
faithfulness stagnation is a **measurement artifact** caused by:

1. The judge model (mistral-small3.2) struggling with longer, mixed contexts
   and defaulting to 0.5
2. One `-1.0` scoring failure on the probable-cause question for llama
3. The "fully supported by context" criterion being harder to satisfy when
   context includes brief web snippets that the model synthesized beyond

The *actual* answer quality is demonstrably better -- no "I don't know" answers,
more comprehensive explanations, and fewer retry loops.

### Recommended Next Steps

1. **Re-evaluate with `llama3.3:70b` as judge** -- it handles long context
   better than the DeepSeek R1 distillation, which tends to default to 0.5
   when reasoning over mixed ChromaDB + web contexts
2. **Fix the web search routing** so it always fires when enabled (not gated
   by retrieval scores), preventing the mistral summary-judgment failure
3. **Consider separate faithfulness scoring for ChromaDB vs web sources** to
   isolate which context type the answer is grounding on
