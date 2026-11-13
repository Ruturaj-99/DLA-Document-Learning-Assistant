"""Smoke tests for the evals/ RAG evaluation framework.

These keep the framework honest without slowing the suite down: a small
subset of the 100 questions runs through the REAL pipeline (MiniLM
embeddings, real retrieval, real qa.score gate) with a scripted oracle
instead of Ollama - the same seam tests/test_qa.py uses.

The full run is a CLI command, not a test:
    .venv\\Scripts\\python.exe -m evals.run_evaluation --json report.json
"""

import sys
from pathlib import Path

import pytest

# Make the project root importable (tests/ and evals/ both live there).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("sentence_transformers", reason="MiniLM not installed")

from evals.document import PAGE_TEXTS, build_document, document_id  # noqa: E402
from evals.metrics import citation_metrics, is_gold_chunk  # noqa: E402
from evals.questions import QUESTIONS, count_by_category  # noqa: E402
from evals.run_evaluation import (  # noqa: E402
    EvaluationReport,
    build_corpus,
    evaluate_corpus,
    make_oracle,
)
from src.qa import NOT_FOUND_MARKER  # noqa: E402


def _small_report() -> EvaluationReport:
    """The first 4 answerable + 2 unanswerable questions (fast subset)."""
    subset = [QUESTIONS[0], QUESTIONS[5], QUESTIONS[12], QUESTIONS[45],
              QUESTIONS[75], QUESTIONS[90]]
    embedded, page_texts = build_corpus()
    return evaluate_corpus(embedded, page_texts, questions=subset)


def test_dataset_has_exactly_100_questions_in_known_categories():
    counts = count_by_category()
    assert sum(counts.values()) == 100
    assert set(counts) == {"fact", "paraphrase", "cross", "unanswerable"}
    assert len({question.id for question in QUESTIONS}) == 100


def test_evaluation_document_builds_and_quotes_hold():
    """The PDF builds, and every gold snippet really appears in its gold
    page text - a dataset whose snippets don't match its document would
    silently grade everything as a miss."""
    pdf_bytes = build_document()
    assert len(pdf_bytes) > 1000

    from src.pdf_loader import extract_pdf_pages

    pages = extract_pdf_pages(pdf_bytes, enable_ocr=False)
    page_texts = {page.page_number: page.text for page in pages}

    for question in QUESTIONS:
        if question.category == "unanswerable":
            continue
        assert question.gold_pages, f"{question.id} has no gold page"
        assert len(question.gold_pages) == len(question.gold_snippets)
        for page, snippet in zip(
            question.gold_pages, question.gold_snippets
        ):
            assert page in page_texts, f"{question.id}: page {page} missing"
            assert is_gold_chunk(
                page, page_texts[page], (page,), (snippet,)
            ), f"{question.id}: snippet not found on page {page}"


def test_smoke_run_produces_sane_metrics():
    report = _small_report()
    assert report.answerable.count == 4
    assert report.unanswerable.count == 2

    # Hit rates are fractions within [0, 1].
    answerable = report.answerable.as_dict()
    for key in ("hit@1", "hit@3", "hit@5", "mrr", "context_coverage",
                "answer_supported", "citation_valid"):
        assert 0.0 <= answerable[key] <= 1.0, key
    unanswerable = report.unanswerable.as_dict()
    assert 0.0 <= unanswerable["correct_rejection"] <= 1.0

    # The provenance round-trip must hold for every cited source.
    for item in report.per_question:
        if not item.bundle_metrics.get("not_found"):
            assert item.bundle_metrics["citation_valid"], item.question_id


def test_oracle_returns_marker_when_evidence_is_missing():
    ask = make_oracle(("twelve newton metres",), "twelve newton metres")
    prompt_with = "Reference passages:\n1 ... torqued to twelve newton metres ..."
    prompt_without = "Reference passages:\n1 ... completely unrelated text ..."
    assert ask(prompt_with) == "twelve newton metres"
    assert ask(prompt_without) == NOT_FOUND_MARKER


def test_citation_metrics_round_trip_with_real_page_text():
    """A fake source whose slice does NOT reproduce its text is invalid."""
    embedded, page_texts = build_corpus()
    chunk = embedded[0]

    good = citation_metrics(
        _fake_bundle([chunk]), page_texts
    )
    assert good["citation_valid"] is True

    bad_chunk = type(chunk)(
        document_id(), chunk.page_number, chunk.text, 0, 3, chunk.vector
    )  # slice [0:3] cannot equal the full text
    bad = citation_metrics(_fake_bundle([bad_chunk]), page_texts)
    assert bad["citation_valid"] is False


def _fake_bundle(sources):
    from src.qa import AnswerBundle

    return AnswerBundle(
        question="test",
        answer="test answer",
        sources=list(sources),
        not_found=False,
        reason="ok",
    )


def test_page_texts_are_the_document_source_of_truth():
    assert len(PAGE_TEXTS) == 10
    assert set(PAGE_TEXTS) == set(range(1, 11))
