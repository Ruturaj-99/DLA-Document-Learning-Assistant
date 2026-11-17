"""Offline, deterministic runner for the 100-question RAG evaluation.

It reuses the PRODUCTION pipeline untouched (src.pdf_loader ->
src.chunking -> src.embeddings -> src.retrieval -> src.qa) and replaces
only the LLM call with a scripted oracle function passed as `ask=` -
the exact seam tests/test_qa.py uses. No Ollama is contacted.

What the oracle simulates:
    - a PERFECT reader: when every gold snippet made it into the
      context, it answers; otherwise it replies with
      qa.NOT_FOUND_MARKER (as the real model is instructed to).
    So "model_no_support" in a result means: retrieval failed to put
    the evidence in front of the LLM - which is exactly what the
    metrics grade.

Usage (from the project root, no app/server needed):
    .venv\\Scripts\\python.exe -m evals.run_evaluation
    .venv\\Scripts\\python.exe -m evals.run_evaluation --top-k 4 --json report.json
    .venv\\Scripts\\python.exe -m evals.run_evaluation --limit 10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from src import qa
from src.chunking import chunk_pages
from src.embeddings import embed_chunks
from src.pdf_loader import extract_pdf_pages
from src.qa import DEFAULT_MIN_SCORE, NOT_FOUND_MARKER, answer_question
from src.retrieval import search_chunks

from evals.document import build_document, document_id
from evals.metrics import (
    Aggregates,
    QuestionResult,
    citation_metrics,
    normalized,
    retrieval_metrics,
)
from evals.questions import QUESTIONS, count_by_category


def make_oracle(gold_snippets: tuple[str, ...], expected_answer: str | None):
    """A scripted `ask=` function for one question (never touches Ollama).

    Answer only when ALL gold snippets are present in the prompt's
    context block - otherwise reply with the not-found marker, exactly
    what the real LLM is instructed to do when passages lack the answer.
    """

    def ask(prompt: str) -> str:
        context = normalized(prompt)
        if expected_answer and all(
            normalized(snippet) in context for snippet in gold_snippets
        ):
            return expected_answer
        return NOT_FOUND_MARKER

    return ask


def build_corpus(
    chunk_size: int = 1200, overlap: int = 150
) -> tuple[list, dict[int, str]]:
    """Run the real ingestion pipeline on the synthetic document.

    Args:
        chunk_size / overlap: the chunking knobs. The DEFAULTS are the
            production values (chunking.DEFAULT_CHUNK_SIZE/DEFAULT_OVERLAP)
            so the run measures the app exactly as configured; different
            values answer "what if we chunked differently?" WITHOUT
            touching production code.

    Returns (embedded_chunks, page_texts) - page texts are kept for the
    provenance round-trip check in the citation metrics.
    """
    pages = extract_pdf_pages(build_document(), enable_ocr=False)
    page_texts = {page.page_number: page.text for page in pages}
    chunks = chunk_pages(document_id(), pages, chunk_size, overlap)
    embedded = embed_chunks(chunks)
    if not embedded:
        raise RuntimeError("Evaluation document produced no chunks.")
    return embedded, page_texts


def evaluate_corpus(
    embedded_chunks,
    page_texts: dict[int, str],
    questions=None,
    top_k: int = 5,
    min_score: float = DEFAULT_MIN_SCORE,
) -> "EvaluationReport":
    """Run every question through retrieval + qa and aggregate metrics."""
    questions = questions if questions is not None else QUESTIONS
    per_question: list[QuestionResult] = []
    answerable_totals = Aggregates(ks=(1, 3, 5))
    unanswerable_totals = Aggregates(ks=(1, 3, 5))

    for question in questions:
        unanswerable = question.category == "unanswerable"

        # --- retrieval only (what search_chunks really returns) ---------
        results = search_chunks(
            question.question, embedded_chunks, top_k=top_k
        )
        retrieval = retrieval_metrics(
            results, question.gold_pages, question.gold_snippets
        )

        # --- the full answer pipeline with the scripted oracle ----------
        bundle = answer_question(
            question.question,
            embedded_chunks,
            top_k=top_k,
            min_score=min_score,
            ask=make_oracle(
                question.gold_snippets, question.expected_answer
            ),
        )
        bundle_metrics = citation_metrics(
            bundle,
            page_texts,
            question.gold_pages,
            question.gold_snippets,
        )

        result = QuestionResult(
            question_id=question.id,
            question=question.question,
            category=question.category,
            retrieval=retrieval,
            bundle_metrics=bundle_metrics,
        )
        per_question.append(result)
        (unanswerable_totals if unanswerable else answerable_totals).add(
            result, unanswerable
        )

    return EvaluationReport(
        per_question=per_question,
        answerable=answerable_totals,
        unanswerable=unanswerable_totals,
        top_k=top_k,
        min_score=min_score,
    )


class EvaluationReport:
    """Aggregated results for one evaluation run."""

    def __init__(
        self,
        per_question: list[QuestionResult],
        answerable: Aggregates,
        unanswerable: Aggregates,
        top_k: int,
        min_score: float,
    ):
        self.per_question = per_question
        self.answerable = answerable
        self.unanswerable = unanswerable
        self.top_k = top_k
        self.min_score = min_score

    # --- output helpers ------------------------------------------------

    def to_dict(self) -> dict:
        """JSON-serializable form of the whole run (all questions kept)."""
        return {
            "settings": {"top_k": self.top_k, "min_score": self.min_score},
            "dataset": count_by_category(),
            "answerable": self.answerable.as_dict(),
            "unanswerable": self.unanswerable.as_dict(),
            "questions": [
                {
                    "id": item.question_id,
                    "category": item.category,
                    "retrieval": item.retrieval,
                    "bundle": item.bundle_metrics,
                }
                for item in self.per_question
            ],
        }

    def format_report(self) -> str:
        """Human-readable console report."""
        answerable = self.answerable.as_dict()
        unanswerable = self.unanswerable.as_dict()
        lines = [
            "=" * 62,
            "RAG EVALUATION - 100 questions over the synthetic handbook",
            "=" * 62,
            f"settings: top_k={self.top_k}, min_score={self.min_score}",
            f"dataset : {count_by_category()}",
            "",
            f"ANSWERABLE questions ({answerable['questions']}):",
            f"  hit@1 = {answerable['hit@1']:<6} hit@3 = {answerable['hit@3']:<6} hit@5 = {answerable['hit@5']}",
            f"  MRR   = {answerable['mrr']}",
            f"  context_coverage   = {answerable['context_coverage']}",
            f"  answer_supported   = {answerable['answer_supported']}",
            f"  citation_valid     = {answerable['citation_valid']}",
            "",
            f"UNANSWERABLE questions ({unanswerable['questions']}):",
            f"  correct_rejection  = {unanswerable['correct_rejection']}",
            f"  citation_valid     = {unanswerable['citation_valid']}",
            "",
            "worst answerable questions (no gold in top 5):",
        ]
        misses = [
            item
            for item in self.per_question
            if item.category != "unanswerable"
            and item.retrieval.get("first_gold_rank", 0) == 0
        ]
        if misses:
            for item in misses:
                rank = item.retrieval.get("first_gold_rank", 0)
                lines.append(
                    f"  {item.question_id} [{item.category}] "
                    f"rank={rank or '-'} : {item.question}"
                )
        else:
            lines.append("  (none - every gold passage surfaced)")
        lines.append("=" * 62)
        return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Offline 100-question RAG evaluation (no Ollama needed)."
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="passages fetched per question (default 5)",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=DEFAULT_MIN_SCORE,
        help=f"score gate (default {DEFAULT_MIN_SCORE})",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1200,
        help="chunking knob (production default 1200) for what-if runs",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=150,
        help="chunking knob (production default 150) for what-if runs",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="evaluate only the first N questions (smoke runs)",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="write the full report as JSON to this path",
    )
    args = parser.parse_args(argv)

    started = time.perf_counter()
    print(
        "Building corpus (extraction -> chunking -> embeddings) "
        f"[chunk_size={args.chunk_size}, overlap={args.overlap}]..."
    )
    embedded_chunks, page_texts = build_corpus(
        chunk_size=args.chunk_size, overlap=args.overlap
    )
    print(
        f"  {len(page_texts)} pages, "
        f"{len(embedded_chunks)} embedded chunks."
    )

    questions = QUESTIONS[: args.limit] if args.limit else QUESTIONS
    report = evaluate_corpus(
        embedded_chunks,
        page_texts,
        questions=questions,
        top_k=args.top_k,
        min_score=args.min_score,
    )

    print(report.format_report())
    elapsed = time.perf_counter() - started
    print(f"elapsed: {elapsed:.1f}s")

    if args.json:
        args.json.write_text(
            json.dumps(report.to_dict(), indent=2), encoding="utf-8"
        )
        print(f"JSON report written to {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
