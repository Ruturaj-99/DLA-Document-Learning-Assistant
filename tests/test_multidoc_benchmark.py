"""Stage 5G tests: dataset integrity and the multi-document runner.

The real-PDF checks skip cleanly on a machine without the benchmark PDFs.
The runner test uses a tiny in-memory corpus, so it always runs.
"""

import sys
from pathlib import Path

import pytest

# Make "src" importable when pytest is run from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.metrics import normalized  # noqa: E402
from evals.multidoc_questions import (  # noqa: E402
    DOCUMENTS,
    QUESTIONS,
    BenchmarkDocument,
    MultiDocQuestion,
    count_by_category,
    document_path,
    questions_by_document,
)
from evals.run_multidoc import (  # noqa: E402
    evaluate_document,
    evaluate_questions,
    format_report,
    report_to_dict,
)
from src.chunking import chunk_pages  # noqa: E402
from src.embeddings import embed_chunks  # noqa: E402
from src.pdf_loader import PageContent, extract_pdf_pages  # noqa: E402


def test_dataset_covers_multiple_documents():
    assert len(DOCUMENTS) >= 5
    assert len(QUESTIONS) >= 40
    assert {question.document_key for question in QUESTIONS} == set(DOCUMENTS)


def test_dataset_has_the_expected_categories():
    counts = count_by_category()
    assert counts.get("fact", 0) >= 20
    assert counts.get("paraphrase", 0) >= 5
    assert counts.get("toc", 0) >= 2
    assert counts.get("unanswerable", 0) >= 4


def test_question_ids_are_unique():
    ids = [question.id for question in QUESTIONS]
    assert len(ids) == len(set(ids))


def test_question_references_and_gold_fields_are_consistent():
    for question in QUESTIONS:
        assert question.document_key in DOCUMENTS
        assert len(question.gold_pages) == len(question.gold_snippets)
        if question.category == "unanswerable":
            assert question.gold_pages == ()
            assert question.expected_answer is None
        else:
            assert question.gold_pages, f"{question.id} has no gold page"
            assert question.expected_answer, f"{question.id} has no answer"


def test_every_document_has_questions():
    grouped = questions_by_document()
    for key in DOCUMENTS:
        assert grouped[key], f"{key} has no questions"


def test_verified_snippets_still_exist_on_their_gold_pages():
    checked_documents = 0
    failures = []

    for key, document in DOCUMENTS.items():
        path = document_path(document)
        if not path.exists():
            continue
        checked_documents += 1
        pages = {
            page.page_number: page.text
            for page in extract_pdf_pages(
                path.read_bytes(), enable_ocr=False
            )
        }
        for question in questions_by_document()[key]:
            if question.category == "unanswerable":
                continue
            for page_number, snippet in zip(
                question.gold_pages, question.gold_snippets
            ):
                page_text = pages.get(page_number)
                if page_text is None:
                    failures.append(
                        f"{question.id}: page {page_number} does not exist"
                    )
                elif normalized(snippet) not in normalized(page_text):
                    failures.append(
                        f"{question.id}: snippet not found on page "
                        f"{page_number}: {snippet!r}"
                    )

    if checked_documents == 0:
        pytest.skip("no benchmark PDFs available on this machine")
    assert not failures, "\n".join(failures)


def test_runner_scores_a_tiny_answerable_and_unanswerable_corpus():
    document = BenchmarkDocument(
        key="synthetic",
        file_name="synthetic.pdf",
        label="Synthetic",
        document_id="synthetic",
    )
    pages = [
        PageContent(1, "The blue valve is part number BV-100.")
    ]
    embedded = embed_chunks(chunk_pages("synthetic", pages))
    questions = [
        MultiDocQuestion(
            id="synthetic-01",
            document_key="synthetic",
            question="Which part number is the blue valve?",
            category="fact",
            gold_pages=(1,),
            gold_snippets=("bv-100",),
            expected_answer="BV-100",
        ),
        MultiDocQuestion(
            id="synthetic-02",
            document_key="synthetic",
            question="What is the price of the blue valve?",
            category="unanswerable",
        ),
    ]

    report = evaluate_questions(
        document,
        questions,
        embedded,
        {1: pages[0].text},
        top_k=5,
    )

    answerable = report.answerable.as_dict()
    unanswerable = report.unanswerable.as_dict()
    assert answerable["questions"] == 1
    assert answerable["hit@1"] == 1.0
    assert answerable["answer_supported"] == 1.0
    assert answerable["citation_valid"] == 1.0
    assert unanswerable["questions"] == 1
    assert unanswerable["correct_rejection"] == 1.0

    payload = report_to_dict([report], top_k=5, min_score=0.25)
    assert payload["overall"]["answerable"]["hit@1"] == 1.0
    assert payload["documents"][0]["chunk_count"] == 1

    text = format_report([report], top_k=5, min_score=0.25)
    assert "Synthetic" in text
    assert "OVERALL" in text


def test_missing_pdf_is_skipped_not_counted_as_failure():
    missing = BenchmarkDocument(
        key="missing",
        file_name="definitely-not-a-real-benchmark-file.pdf",
        label="Missing PDF",
        document_id="missing",
    )
    report = evaluate_document(missing)
    assert report.skipped is True
    assert "file not found" in report.skip_reason
