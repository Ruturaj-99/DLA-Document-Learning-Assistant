"""Stage 5G: multi-document baseline benchmark.

Runs the PRODUCTION pipeline (pdf_loader -> chunking -> embeddings ->
retrieval -> qa) on several real PDFs and scores each verified question.

The LLM is replaced by the same scripted oracle used by the synthetic RAG
evaluation, so this run is deterministic and does not need Ollama:

    gold snippets all present in the retrieved context  -> answer
    otherwise                                           -> not found

That makes the headline numbers retrieval-grounded accuracy, not live
model-writing accuracy. A live-model mode can be added later without
changing the dataset or the metrics.

Usage (from the project root):
    .venv\\Scripts\\python.exe -m evals.run_multidoc
    .venv\\Scripts\\python.exe -m evals.run_multidoc --document odams
    .venv\\Scripts\\python.exe -m evals.run_multidoc --json multidoc.json

Set DLA_BENCHMARK_DIR when the PDFs are not in the current user's Downloads
folder. Missing PDFs are reported as skipped, never as failures.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from evals.metrics import (
    Aggregates,
    QuestionResult,
    citation_metrics,
    retrieval_metrics,
)
from evals.multidoc_questions import (
    DOCUMENTS,
    BenchmarkDocument,
    MultiDocQuestion,
    benchmark_dir,
    count_by_category,
    document_path,
    questions_by_document,
)
from evals.run_evaluation import make_oracle
from src.chunking import chunk_pages
from src.embeddings import embed_chunks
from src.pdf_loader import extract_pdf_pages
from src.qa import DEFAULT_MIN_SCORE, answer_question
from src.retrieval import search_chunks


@dataclass
class DocumentReport:
    """Everything measured for one benchmark PDF."""

    document: BenchmarkDocument
    page_count: int = 0
    chunk_count: int = 0
    answerable: Aggregates = field(default_factory=lambda: Aggregates(ks=(1, 3, 5)))
    unanswerable: Aggregates = field(default_factory=lambda: Aggregates(ks=(1, 3, 5)))
    results: list[QuestionResult] = field(default_factory=list)
    skipped: bool = False
    skip_reason: str = ""

    @property
    def question_count(self) -> int:
        return len(self.results)

    def to_dict(self) -> dict:
        return {
            "key": self.document.key,
            "label": self.document.label,
            "file_name": self.document.file_name,
            "page_count": self.page_count,
            "chunk_count": self.chunk_count,
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
            "answerable": self.answerable.as_dict()
            if self.answerable.count
            else {"questions": 0},
            "unanswerable": self.unanswerable.as_dict()
            if self.unanswerable.count
            else {"questions": 0},
            "questions": [
                {
                    "id": item.question_id,
                    "category": item.category,
                    "retrieval": item.retrieval,
                    "bundle": item.bundle_metrics,
                }
                for item in self.results
            ],
        }


def evaluate_questions(
    document: BenchmarkDocument,
    questions: list[MultiDocQuestion],
    embedded_chunks,
    page_texts: dict[int, str],
    top_k: int = 5,
    min_score: float = DEFAULT_MIN_SCORE,
) -> DocumentReport:
    """Score already-embedded chunks against one document's questions."""
    report = DocumentReport(
        document=document,
        page_count=len(page_texts),
        chunk_count=len(embedded_chunks),
    )

    for question in questions:
        unanswerable = question.category == "unanswerable"

        results = search_chunks(
            question.question, embedded_chunks, top_k=top_k
        )
        retrieval = retrieval_metrics(
            results, question.gold_pages, question.gold_snippets
        )

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
        report.results.append(result)
        (report.unanswerable if unanswerable else report.answerable).add(
            result, unanswerable
        )

    return report


def evaluate_document(
    document: BenchmarkDocument,
    top_k: int = 5,
    min_score: float = DEFAULT_MIN_SCORE,
) -> DocumentReport:
    """Extract, chunk, embed and score one PDF."""
    path = document_path(document)
    if not path.exists():
        return DocumentReport(
            document=document,
            skipped=True,
            skip_reason=f"file not found: {path}",
        )

    pages = extract_pdf_pages(path.read_bytes())
    chunks = chunk_pages(document.document_id, pages)
    embedded_chunks = embed_chunks(chunks)
    page_texts = {page.page_number: page.text for page in pages}

    return evaluate_questions(
        document,
        questions_by_document()[document.key],
        embedded_chunks,
        page_texts,
        top_k=top_k,
        min_score=min_score,
    )


def _merge_into(target: Aggregates, source: Aggregates) -> None:
    """Add one report's counters into an overall aggregate."""
    target.count += source.count
    for k in source.ks:
        target.hit_counts[k] = (
            target.hit_counts.get(k, 0) + source.hit_counts.get(k, 0)
        )
    target.reciprocal_rank_sum += source.reciprocal_rank_sum
    target.context_coverage_count += source.context_coverage_count
    target.correct_rejection_count += source.correct_rejection_count
    target.citation_valid_count += source.citation_valid_count
    target.answer_supported_count += source.answer_supported_count
    target.not_found_count += source.not_found_count


def combine_reports(
    reports: list[DocumentReport],
) -> tuple[Aggregates, Aggregates]:
    """Combine every evaluated document into overall answerable/unanswerable."""
    answerable = Aggregates(ks=(1, 3, 5))
    unanswerable = Aggregates(ks=(1, 3, 5))
    for report in reports:
        if report.skipped:
            continue
        _merge_into(answerable, report.answerable)
        _merge_into(unanswerable, report.unanswerable)
    return answerable, unanswerable


def _rate(values: dict, key: str) -> str:
    value = values.get(key)
    return "-" if value is None else f"{value:.3f}"


def format_report(
    reports: list[DocumentReport],
    top_k: int,
    min_score: float,
) -> str:
    """Human-readable per-document and overall table."""
    evaluated = [report for report in reports if not report.skipped]
    skipped = [report for report in reports if report.skipped]

    answerable_total = sum(report.answerable.count for report in evaluated)
    unanswerable_total = sum(
        report.unanswerable.count for report in evaluated
    )
    lines = [
        "=" * 118,
        "MULTI-DOCUMENT BENCHMARK (retrieval-grounded, scripted oracle)",
        "=" * 118,
        f"settings: top_k={top_k}, min_score={min_score}, "
        f"benchmark_dir={benchmark_dir()}",
        f"dataset : {count_by_category()}",
        f"documents found={len(evaluated)} skipped={len(skipped)} | "
        f"answerable={answerable_total} unanswerable={unanswerable_total}",
        "",
        f"{'document':<36} {'pg':>3} {'chk':>4} {'Q':>3} "
        f"{'hit@1':>6} {'hit@3':>6} {'hit@5':>6} {'MRR':>6} "
        f"{'cover':>6} {'supp':>6} {'cite':>6} | {'unans':>5} {'reject':>6}",
        "-" * 118,
    ]

    for report in reports:
        if report.skipped:
            lines.append(
                f"{report.document.label:<36} SKIPPED  {report.skip_reason}"
            )
            continue
        answerable = (
            report.answerable.as_dict() if report.answerable.count else {}
        )
        unanswerable = (
            report.unanswerable.as_dict()
            if report.unanswerable.count
            else {}
        )
        lines.append(
            f"{report.document.label:<36} "
            f"{report.page_count:>3} {report.chunk_count:>4} "
            f"{report.question_count:>3} "
            f"{_rate(answerable, 'hit@1'):>6} "
            f"{_rate(answerable, 'hit@3'):>6} "
            f"{_rate(answerable, 'hit@5'):>6} "
            f"{_rate(answerable, 'mrr'):>6} "
            f"{_rate(answerable, 'context_coverage'):>6} "
            f"{_rate(answerable, 'answer_supported'):>6} "
            f"{_rate(answerable, 'citation_valid'):>6} | "
            f"{unanswerable.get('questions', 0):>5} "
            f"{_rate(unanswerable, 'correct_rejection'):>6}"
        )

    overall_answerable, overall_unanswerable = combine_reports(evaluated)

    if overall_answerable.count:
        values = overall_answerable.as_dict()
        unans = (
            overall_unanswerable.as_dict()
            if overall_unanswerable.count
            else {}
        )
        lines.append("-" * 118)
        lines.append(
            f"{'OVERALL':<36} {'':>3} {'':>4} "
            f"{overall_answerable.count + overall_unanswerable.count:>3} "
            f"{_rate(values, 'hit@1'):>6} "
            f"{_rate(values, 'hit@3'):>6} "
            f"{_rate(values, 'hit@5'):>6} "
            f"{_rate(values, 'mrr'):>6} "
            f"{_rate(values, 'context_coverage'):>6} "
            f"{_rate(values, 'answer_supported'):>6} "
            f"{_rate(values, 'citation_valid'):>6} | "
            f"{overall_unanswerable.count:>5} "
            f"{_rate(unans, 'correct_rejection'):>6}"
        )

        misses = [
            item
            for report in evaluated
            for item in report.results
            if item.category != "unanswerable"
            and item.retrieval.get("first_gold_rank", 0) == 0
        ]
        lines.append("")
        lines.append("answerable questions with no gold passage in top-k:")
        if misses:
            for item in misses:
                lines.append(f"  {item.question_id}: {item.question}")
        else:
            lines.append("  (none - every gold passage surfaced)")

    lines.append("=" * 118)
    return "\n".join(lines)


def build_reports(
    document_keys: list[str] | None,
    top_k: int,
    min_score: float,
) -> list[DocumentReport]:
    """Evaluate the requested documents in dataset order."""
    keys = document_keys or list(DOCUMENTS)
    unknown = [key for key in keys if key not in DOCUMENTS]
    if unknown:
        raise ValueError(
            f"unknown document key(s): {', '.join(unknown)}; "
            f"choose from {', '.join(DOCUMENTS)}"
        )

    reports: list[DocumentReport] = []
    for key in keys:
        document = DOCUMENTS[key]
        reports.append(
            evaluate_document(document, top_k=top_k, min_score=min_score)
        )
    return reports


def report_to_dict(
    reports: list[DocumentReport],
    top_k: int,
    min_score: float,
) -> dict:
    """JSON-serializable baseline report."""
    evaluated = [report for report in reports if not report.skipped]
    answerable, unanswerable = combine_reports(evaluated)

    return {
        "settings": {
            "top_k": top_k,
            "min_score": min_score,
            "benchmark_dir": str(benchmark_dir()),
        },
        "dataset": count_by_category(),
        "overall": {
            "answerable": answerable.as_dict()
            if answerable.count
            else {"questions": 0},
            "unanswerable": unanswerable.as_dict()
            if unanswerable.count
            else {"questions": 0},
        },
        "documents": [report.to_dict() for report in reports],
        "questions": [
            {
                "id": item.question_id,
                "document": report.document.key,
                "category": item.category,
                "retrieval": item.retrieval,
                "bundle": item.bundle_metrics,
            }
            for report in evaluated
            for item in report.results
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 5G multi-document baseline benchmark "
            "(no Ollama required)."
        )
    )
    parser.add_argument(
        "--document",
        action="append",
        choices=list(DOCUMENTS),
        help="run only this document key (repeatable; default: all)",
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
        "--json",
        type=Path,
        default=None,
        help="write the full baseline report as JSON to this path",
    )
    args = parser.parse_args(argv)

    started = time.perf_counter()
    try:
        reports = build_reports(args.document, args.top_k, args.min_score)
    except ValueError as error:
        parser.error(str(error))
        return 2

    print(format_report(reports, args.top_k, args.min_score))
    elapsed = time.perf_counter() - started
    print(f"elapsed: {elapsed:.1f}s")

    if args.json:
        payload = report_to_dict(reports, args.top_k, args.min_score)
        args.json.write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        print(f"JSON report written to {args.json}")

    return 0 if any(not report.skipped for report in reports) else 2


if __name__ == "__main__":
    sys.exit(main())
