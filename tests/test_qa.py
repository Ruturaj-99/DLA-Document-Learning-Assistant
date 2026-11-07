"""Step 4C: tests for src/qa.py.

Run from the project root with:
    .venv\\Scripts\\python -m pytest tests/ -v

Strategy (beginner-friendly and honest):
    - Embeddings are REAL (MiniLM is cached locally, loading takes ~1 s).
    - The LLM is FAKE: we pass a small function as `ask=` instead of
      `ollama.ask_ollama`, so tests never depend on the model's exact words.
    - One LIVE demo test talks to the real model but skips itself when
      Ollama is unavailable.
"""

import sys
import unittest.mock as mock
from pathlib import Path

import pytest

# Make "src" importable when pytest is run from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import qa  # noqa: E402
from src.chunking import Chunk  # noqa: E402
from src.embeddings import embed_chunks  # noqa: E402
from src.ollama import ask_ollama  # noqa: E402
from src.retrieval import SearchResult  # noqa: E402

TECHNICAL_CHUNKS = [
    Chunk("manual-heater", 2, "To reset the heater, hold the power button for five seconds.", 55, 115),
    Chunk("manual-heater", 1, "The heater has a two-speed fan for warm air.", 0, 50),
    Chunk("garden-notes", 3, "Roses need sunlight and regular watering in spring.", 0, 50),
]
EMBEDDED = embed_chunks(TECHNICAL_CHUNKS)


def fail_if_called(prompt: str) -> str:
    raise AssertionError("the LLM must not be called in this scenario")


def test_blank_question_raises_without_any_call():
    with pytest.raises(ValueError, match="non-empty question"):
        qa.answer_question("   ", EMBEDDED, ask=fail_if_called)


def test_no_chunks_reports_no_passages_without_calling_model():
    bundle = qa.answer_question("How do I reset the heater?", [], ask=fail_if_called)
    assert bundle.not_found is True
    assert bundle.reason == "no_passages"
    assert bundle.sources == []


def test_score_gate_skips_model_when_retrieval_is_weak():
    # min_score above any realistic similarity forces the gate.
    bundle = qa.answer_question(
        "How do I reset the heater?", EMBEDDED, min_score=0.99, ask=fail_if_called
    )
    assert bundle.not_found is True
    assert bundle.reason == "low_relevance"
    assert bundle.sources == []  # no fake citations for weak evidence


def test_model_admitting_no_support_is_reported_honestly():
    def model_says_not_in_documents(prompt: str) -> str:
        return qa.NOT_FOUND_MARKER

    bundle = qa.answer_question(
        "How do I reset the heater?", EMBEDDED, ask=model_says_not_in_documents
    )
    assert bundle.not_found is True
    assert bundle.reason == "model_no_support"
    assert "could not find an answer" in bundle.answer
    assert bundle.sources == []


def test_extra_words_around_marker_still_caught():
    def model_hedges(prompt: str) -> str:
        return "Sorry, the passages don't contain that: ANSWER_NOT_IN_DOCUMENTS"

    bundle = qa.answer_question("How do I reset the heater?", EMBEDDED, ask=model_hedges)
    assert bundle.not_found is True


def test_success_returns_answer_and_our_citations():
    def fake_model(prompt: str) -> str:
        return "Hold the power button for five seconds."

    bundle = qa.answer_question(
        "How do I reset the heater?", EMBEDDED, ask=fake_model, top_k=2
    )
    assert bundle.not_found is False
    assert bundle.reason == "ok"
    assert bundle.answer == "Hold the power button for five seconds."
    # Citations come from OUR pipeline: metadata preserved 1:1 from retrieval.
    assert len(bundle.sources) == 2
    top = bundle.sources[0]
    assert top.document_id == "manual-heater"  # the reset passage wins
    assert top.page_number == 2
    assert top.start_char == 55 and top.end_char == 115
    assert top.score >= 0  # scores travel with the citations


def test_prompt_contains_cheat_sheet_and_rules():
    seen_prompt = []

    def capture(prompt: str) -> str:
        seen_prompt.append(prompt)
        return "ok"

    qa.answer_question("How do I reset the heater?", EMBEDDED, ask=capture, top_k=3)
    prompt = seen_prompt[0]
    # The numbered cheat sheet is there...
    assert "[1] (document manual-heater, page 2)" in prompt
    assert "Roses need sunlight" in prompt
    # ...and so are the grounding rules.
    assert "ONLY the numbered reference passages" in prompt
    assert qa.NOT_FOUND_MARKER in prompt
    assert "How do I reset the heater?" in prompt


def test_mocked_retrieval_success_flows_into_prompt_and_citations():
    """Deterministic end-to-end check: retrieval is MOCKED (no model load).

    Proves the chain works with any SearchResult payload:
    search -> prompt cheat sheet -> fake LLM -> code-made citations.
    """
    fake_result = SearchResult(
        document_id="archive.pdf",
        page_number=7,
        text="Hold the reset key for five seconds.",
        start_char=10,
        end_char=34,
        score=0.61,  # a made-up score; deterministic unit test
    )
    seen_prompt = []

    def capture(prompt: str) -> str:
        seen_prompt.append(prompt)
        return "Hold the reset key for five seconds."

    with mock.patch.object(qa, "search_chunks", return_value=[fake_result]):
        # chunks list is EMPTY on purpose: proves the mocked search result is used.
        bundle = qa.answer_question("how do I reset?", [], ask=capture)

    assert bundle.reason == "ok" and bundle.not_found is False
    assert bundle.sources[0].page_number == 7
    assert bundle.sources[0].document_id == "archive.pdf"
    prompt = seen_prompt[0]
    assert "[1] (document archive.pdf, page 7)" in prompt
    assert "Hold the reset key for five seconds." in prompt


def test_mocked_retrieval_empty_reports_no_passages_without_calling_model():
    with mock.patch.object(qa, "search_chunks", return_value=[]):
        bundle = qa.answer_question("anything", [], ask=fail_if_called)
    assert bundle.reason == "no_passages" and bundle.sources == []


def _ollama_ready() -> bool:
    from src.ollama import ollama_available

    return ollama_available()


@pytest.mark.skipif(not _ollama_ready(), reason="live test: local Ollama unavailable")
def test_live_rag_demo_model_uses_passage():
    """Flight check with the REAL local model. May vary in wording."""
    bundle = qa.answer_question(
        "How do I reset the heater?",
        EMBEDDED,
        ask=ask_ollama,
        top_k=3,
    )
    assert bundle.reason == "ok"
    assert bundle.sources, "a supported answer needs at least one source"
    assert any("reset the heater" in s.text for s in bundle.sources)
