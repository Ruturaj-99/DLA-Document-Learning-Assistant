# Multi-document benchmark (Stage 5G)

A deterministic baseline over **five real PDFs** and **41 verified questions**.
It answers a simple question: when the app reads a different document, does it
still find the page that holds the answer, cite it correctly, and refuse
questions the document cannot answer?

## Files

| File | Purpose |
|---|---|
| `evals/multidoc_questions.py` | Document registry + verified questions and gold snippets |
| `evals/run_multidoc.py` | Per-document pipeline runner and report |
| `tests/test_multidoc_benchmark.py` | Dataset verification + runner tests |

The PDFs are user documents and are **not committed**. By default the runner
looks in the current user's `Downloads` folder; set `DLA_BENCHMARK_DIR` to
point somewhere else. Missing PDFs are reported as skipped, never scored as
failures.

## Run

```powershell
.venv\Scripts\python.exe -m evals.run_multidoc
.venv\Scripts\python.exe -m evals.run_multidoc --document odams
.venv\Scripts\python.exe -m evals.run_multidoc --json baseline.json
```

The LLM is replaced by the same scripted oracle used by the synthetic
evaluation, so no Ollama server is needed. The headline numbers are therefore
**retrieval-grounded accuracy**: an answer counts only when the retrieved
context really contains the verified gold snippet.

## Dataset rules

- Every answerable question has a 1-based `gold_pages` entry and a
  `gold_snippets` entry copied verbatim from that page's extracted text.
- `tests/test_multidoc_benchmark.py` re-extracts each available PDF and fails
  if a snippet no longer appears on its page, so the dataset cannot drift.
- Categories: `fact`, `paraphrase`, `toc`, `unanswerable`.
- An `unanswerable` question deliberately has no gold page; it tests the score
  gate and the honest "not found" path.

## Results: fixed-window baseline vs adaptive chunking vs hybrid retrieval vs boilerplate demotion vs answer generation (Stage 5L)

Five runs of the same 41 verified questions:

| Document | hit@1 | hit@3 | hit@5 | MRR |
|---|---:|---:|---:|---:|
| PDF TESTING FOR LLM.pdf | 0.143 → 0.286 → 0.429 → 0.429 → 0.429 | 0.143 → 0.571 → 0.429 → 0.429 → 0.429 | 0.286 → 0.571 → 0.571 → 0.571 → 0.571 | 0.179 → 0.405 → 0.464 → 0.464 → 0.464 |
| ANP IoT Data Gateway datasheet | 0.667 → 0.667 → 0.667 → 0.667 → 0.667 | 1.000 → 0.833 → 0.833 → 0.833 → 0.833 | 1.000 → 1.000 → 1.000 → 1.000 → 1.000 | 0.833 → 0.792 → 0.792 → 0.792 → 0.792 |
| Teltonika TSW114 datasheet | 0.375 → 0.250 → 0.250 → 0.375 → 0.375 | 0.375 → 0.375 → 0.500 → 0.625 → 0.625 | 0.625 → 0.500 → 0.500 → 0.750 → 0.750 | 0.431 → 0.338 → 0.354 → 0.504 → 0.504 |
| ODAMS user manual | 0.500 → 0.375 → 0.750 → 0.750 → 0.750 | 0.875 → 0.875 → 0.875 → 0.875 → **1.000** | 0.875 → 0.875 → 0.875 → 0.875 → **1.000** | 0.688 → 0.625 → 0.812 → 0.812 → **0.854** |
| SW-245 4G LTE router datasheet | 0.571 → 0.429 → 0.429 → 0.429 → 0.429 | 1.000 → 1.000 → 1.000 → 1.000 → 1.000 | 1.000 → 1.000 → 1.000 → 1.000 → 1.000 | 0.786 → 0.690 → 0.690 → 0.690 → 0.690 |
| **Overall** | **0.444 → 0.389 → 0.500 → 0.528 → 0.528** | **0.667 → 0.722 → 0.722 → 0.750 → 0.778** | **0.750 → 0.778 → 0.778 → 0.833 → 0.861** | **0.575 → 0.559 → 0.616 → 0.649 → 0.658** |

Runs: (1) fixed 1200-char windows + pure vector search, (2) adaptive
semantic chunking (Stage 5G) + pure vector search, (3) hybrid retrieval
with weighted RRF (Stage 5J), (4) boilerplate demotion (Stage 5K),
(5) answer generation & abstention (Stage 5L).
Chunk counts in runs 3-5 are identical to run 2 (chunking untouched
by Stages 5J/5K/5L).

Every unanswerable question (5/5) was rejected correctly in all runs.
The synthetic 100-question evaluation with Stage 5L:
hit@1 0.920, hit@3 1.000, hit@5 1.000, MRR 0.958,
context_coverage 1.000, answer_supported 1.000, citation_valid 1.000,
rejection 1.0 (Stage 5K had answer_supported 0.960; the two
`low_relevance` false rejections are gone).

## Stage 5L: answer generation & abstention

Fixes the cases where retrieval finds the gold passage but the answer
path still refuses. Six mechanisms, all evidence-based - the 0.25
cosine gate itself is untouched:

1. **Three-tier answerability** (`src/qa.py`, `_is_answerable_evidence`).
   Tier 1: cosine >= 0.25 (unchanged). Tier 2: lexical evidence - the
   IDF-weighted share of the question's rare terms contained in the best
   passage must reach 0.5 (measured separation: no unanswerable question
   exceeds 0.36 over both suites). Tier 3: a passage within 0.10 below
   the gate is sent to the model under the strict passages-only prompt,
   and the MODEL validates answerability - the same treatment passages
   above the gate already get; its refusal is honoured and its answer
   must pass grounding. This rescued the two synthetic false rejections
   (gold inside the top-1 chunk at cosine 0.21-0.23, lexically too thin
   for any safe threshold).
2. **Refusal-variant detection** (`looks_like_refusal`). The local model
   does not always reply with the exact marker (observed:
   "NO ANSWER_IN_DOCUMENTS"); hedged wordings ("cannot find", "does not
   contain the answer", "insufficient information") are caught too, so a
   refusal can no longer leak through as a real answer with citations.
3. **Wrong-refusal retry.** When the model refuses but the lexical tier
   says the passages DO contain the question's rare terms, it is asked
   once more, explicitly - the refusal marker stays available, so the
   model is never forced. Still refusing → honest "not found".
   (Stage 5M widened this to a neutral retry after ANY refusal; see below.)
4. **Grounding check** (`_ungrounded_numbers`). Every number in the
   answer must appear in the retrieved passages (boundary-aware, comma
   and point decimals both accepted, spelled-out document values like
   "zero point two bar" digitised on both sides). An answer that invents
   a value is discarded (`ungrounded_answer`) with empty sources.
5. **Source prioritization** (`prioritize_sources`). Passages containing
   the answer's key values are listed first, generic catalogue-style
   passages after; order-only, nothing is dropped.
6. **Dotted-leader contents boost** (`src/retrieval.py`,
   `is_dotted_contents`). Contents pages formatted as
   "Step 2: Adding Stations ............ 9" were invisible to the strict
   TOC detector, so TOC questions never received the Stage 5F intent
   boost and the gate rejected them (`odams-07`). The detector is used
   ONLY for the boost, never for filtering: hiding the chunks would
   shift BM25 statistics and reorder unrelated questions (measured:
   `odams-01` regressed 1 → 2 under filtering).

## Stage 5M: evidence-to-answer reliability

The Stage 5L benchmarks above measure **retrieval + grounding**, so they were
re-run unchanged after Stage 5M: synthetic hit@1 0.920 / hit@3 1.000 /
hit@5 1.000 / MRR 0.958 / context_coverage 1.000 / answer_supported 1.000 /
citation_valid 1.000 / rejection 1.0, and multidoc overall
hit@1 0.528 / hit@3 0.778 / hit@5 0.861 / MRR 0.658 / rejection 1.000.
No metric moved: retrieval embeddings, chunking, ranking, the model, and the
0.25 cosine gate are untouched.

Real-PDF diagnostics (29-page pressure datasheet, printed 13-31 pages of the
valve sequence document) exposed gaps that those scripted benchmarks cannot
see:

| Symptom | Measured cause | Fix (all outside retrieval ranking) |
|---|---|---|
| "calibration range of tag GQB01CP001" refused although the Annexure-IV row exists | The exact row was vector rank 59 / BM25 rank 32; hybrid RRF buried it under prose about the same subject | `select_evidence` adds chunks containing rare exact literals (letters+digits identifiers and rare 2-3 word phrases) that occur in <= 3 chunks; at most 2 added, prepended, real cosine kept |
| Correct remote-seal explanation refused | A refusal was final unless IDF lexical coverage >= 0.5; explanatory questions legitimately share few rare terms with the passage | After ANY refusal, re-read the SAME evidence once with a neutral prompt; a second refusal is still honoured |
| Same question refused on one run and answered on the next | Ollama's default temperature 0.8 sampled the extraction task | `EXTRACTION_OPTIONS`/`EXTRACTION_TEMPERATURE` set temperature 0 for local and BYOK providers |
| Accuracy/turndown answer dumped a table; multi-part answers lost a part | Prompt only asked for "brief" text | Explicit answer rules: exact values/units, match the row/column, answer every part, one or two short sentences, never copy a passage or cite from the model |
| A serial-number question was answered "1" from the row's quantity column; a row count question was answered "4" from 14 rows | Grounding only checked whether a number appears SOMEWHERE, not whether it answers the asked attribute | Narrow deterministic guards discard a 1-2 digit reply to an identifier question and counts not stated with a count label in the passages; both abstain honestly |
| UI said "answers require at least 0.25" even when the model refused after retrieval | One generic caption for all refusal reasons | Reason-specific UI explanation, including `unsupported_attribute` |

Live pressure-datasheet before -> after (top_k 4, same PDF, llama3.2:3b at
temperature 0): calibration range "could not find" -> "0 - 3.55 kg/cm2";
remote-seal explanation "could not find" -> correct diaphragm/capillary answer;
accuracy/turndown correct but table-dump -> "± 0.06 % ... 60:1"; output signal
and supply voltage "4-20 mA, 18-30 V DC" (the 18-30 value is NOT in the PDF) ->
"4 to 20 mA ... 10.5 to 42 V DC" (the value printed on page 1); flange BOM
(image-only drawing page) still abstains; price still abstains; serial-number
question now abstains instead of answering "1".

Live valve-sequence-document validation (19 pages, top_k 4) confirmed the
same mechanisms on a different document type: CEB2 pumps/valves and the
"NO OF CYCLES : 10 appx." count answered correctly; an implicit row count,
an unstated compressor model/manufacturer, and an unstated pump-pressure
setpoint all abstain. The count guard first caused three benchmark
regressions (an identifier phrased "part number of", an inline count "5 x
eth ports", and the worded answer "Dual SIM"); it was narrowed to skip
identifier questions and non-numeric answers and to accept counts stated
beside the counted noun. Both benchmarks then returned to their Stage 5L
baselines exactly: synthetic answer_supported/citation_valid 1.000,
multidoc overall supp/cite 0.861, with every other retrieval metric
unchanged.

The PDFs used for these checks are user documents and are not committed;
`tests/test_qa_stage5m.py` pins each mechanism with a fake model so the
regressions stay catchable without Ollama.

## Remaining misses

- Retrieval (5 answerable questions with no gold passage in top-5):
  `pdf-testing-01/04/08` (dense spec-table page competing with
  lookalike rotameter/catalogue pages), `tsw114-05` ("weigh" vs the
  page's "Weight" - no lexical or stemming overlap), `tsw114-06`.
- Synthetic: none - answer_supported and citation_valid are both 1.000.
- Stage 5M: counting rows that are only implicit in a dense table (e.g. "how
  many transmitters are listed in Annexure IV?" where the page has 14 numbered
  rows but no stated total) is not reliable with llama3.2:3b - it guessed "4".
  The prompt now forbids guessing and the guard discards unstated counts, so
  such a question abstains instead of producing a confident wrong number. A
  count printed in the passage ("NO OF CYCLES : 10") is still answered.
- Stage 5M / retrieval scope: a multi-part question is only as complete as
  the retrieved passages. On the valve document, "which valves open in
  downward backwash and how long does it last?" got the valve names but not
  the duration (the duration line was outside top_k 4), and the
  upward-vs-downward flow-direction question got only the upward half
  (the page holding the downward sentence was outside top_k 4). A supplier
  name/address implied by the document footer also abstained; those
  sentences were not in the passages. Retrieval ranking was deliberately
  left unchanged in Stage 5M, so these remain known limitations.

## Reading the delta

1. **Hybrid retrieval lifts ranking precision.** Weighted RRF
   (vector 0.7 / BM25 0.3) recovered exact-term and navigation
   questions: ODAMS hit@1 0.375 → 0.750 and overall hit@1 0.389 → 0.500
   over the vector-only adaptive run; MRR 0.559 → 0.616. Every headline
   metric now beats the original fixed-window baseline.
2. **BM25 query-term pruning matters.** Question tokens present in more
   than half of the candidate chunks (repeated page headers, generic
   words) are dropped before scoring - otherwise boilerplate-heavy
   chunks outrank the actual specification rows.
3. **Boilerplate demotion fixes the header-swarm loss (Stage 5K).**
   TSW114 repeats one copyright/version footer line across 7 near-
   identical chunks; they match generic question words, cluster in BOTH
   retriever rankings and filled fused ranks 1-5 ahead of the page-2
   spec chunks. A normalized line unchanged across >= 4 candidate
   chunks is page furniture; each chunk's fused score is scaled by
   1 - 0.3 * (its repeated-character share). Result: tsw114-02 0 → 2,
   tsw114-04 0 → 5, tsw114-09 3 → 1, overall hit@5 0.778 → 0.833, with
   no other question moving (chosen over line-count demotion, which
   regressed tsw114-01; the threshold/penalty plateau 4-8 / 0.2-0.5 all
   score identically).
4. **Remaining misses concentrate in dense spec tables** (TSW114 page 2
   packs many specs per chunk; the BHEL pack's page 7 competes with
   lookalike rotameter pages; `tsw114-05` asks "weigh" while the page
   says "Weight" - no lexical/stemming overlap). The ODAMS "Table
   Contents" wording gap (`odams-07`) was fixed in Stage 5L by the
   dotted-leader contents boost; see the Stage 5L section below.
5. Weight sweep (offline, same benchmark): hit@1 peaks at 0.500 for
   vector weights 0.70/0.67 and degrades to 0.389 at 1.0/0.0
   (vector-only); hit@3/hit@5 are stable at 0.722/0.778 across the
   plateau. Coverage-weighted third RRF lists and BM25 k1 variants
   (0.9/1.2) all scored at or below baseline and were not adopted.
