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


# --- Stage 5L: refusal-variant detection ----------------------------------------
#
# The local model does not always reproduce the marker exactly: llama3.2:3b
# was observed replying "NO ANSWER_IN_DOCUMENTS". The old exact-substring
# check let that variant leak through as a REAL answer, with citations
# attached to a refusal.


SPEC_CHUNKS = [
    Chunk("spec-manual", 2, "The boiling point of acetone is 56 degrees Celsius.", 60, 118),
    Chunk("spec-manual", 1, "Storage: keep containers tightly closed and cool.", 0, 59),
    Chunk("catalogue", 9, "Our range includes solvents, gauges and valves for industries.", 0, 80),
]
SPEC_EMBEDDED = embed_chunks(SPEC_CHUNKS)


def test_variant_refusal_marker_is_not_leaked_as_an_answer():
    calls = []

    def model_says_variant(prompt: str) -> str:
        calls.append(prompt)
        return "NO ANSWER_IN_DOCUMENTS"

    bundle = qa.answer_question(
        "How do I reset the heater?", EMBEDDED, ask=model_says_variant
    )
    assert bundle.not_found is True
    assert bundle.reason == "model_no_support"
    assert bundle.sources == []  # never cite a refusal
    # Stage 5M: every refusal gets one neutral re-read; the variant is still
    # caught after the retry and never leaks through as an answer.
    assert len(calls) == 2


def test_hedged_refusal_is_detected():
    def model_hedges(prompt: str) -> str:
        return "I am sorry, but I cannot find that information in the passages."

    bundle = qa.answer_question("How do I reset the heater?", EMBEDDED, ask=model_hedges)
    assert bundle.not_found is True
    assert bundle.reason == "model_no_support"


def test_looks_like_refusal_covers_observed_wordings():
    assert qa.looks_like_refusal("ANSWER_NOT_IN_DOCUMENTS")
    assert qa.looks_like_refusal("NO ANSWER_IN_DOCUMENTS")
    assert qa.looks_like_refusal("The passages do not contain the answer.")
    assert qa.looks_like_refusal("not found in the documents")
    assert qa.looks_like_refusal("Insufficient information.")
    assert not qa.looks_like_refusal("It is 56 degrees Celsius.")
    assert not qa.looks_like_refusal("")


# --- Stage 5L: evidence-based retry when the model refuses wrongly ---------------


def test_wrong_refusal_gets_one_evidence_backed_retry():
    calls = []

    def model_first_refuses_then_answers(prompt: str) -> str:
        calls.append(prompt)
        if len(calls) == 1:
            return qa.NOT_FOUND_MARKER
        return "The boiling point of acetone is 56 degrees Celsius."

    bundle = qa.answer_question(
        "Boiling point of acetone?", SPEC_EMBEDDED, ask=model_first_refuses_then_answers
    )
    assert len(calls) == 2  # exactly one retry, never a forced loop
    assert bundle.not_found is False
    assert bundle.reason == "ok"
    assert "56" in bundle.answer
    assert bundle.sources


def test_retry_that_still_refuses_is_reported_honestly():
    calls = []

    def model_always_refuses(prompt: str) -> str:
        calls.append(prompt)
        return qa.NOT_FOUND_MARKER

    bundle = qa.answer_question(
        "Boiling point of acetone?", SPEC_EMBEDDED, ask=model_always_refuses
    )
    assert len(calls) == 2
    assert bundle.not_found is True
    assert bundle.reason == "model_no_support"
    assert bundle.sources == []


# --- Stage 5L: lexical evidence tier of the answerability gate -------------------


def test_low_cosine_with_strong_lexical_evidence_is_answerable():
    """Tier 2: the passage literally contains the question's rare terms, so a
    low cosine score must not block it (e.g. BM25-led dense spec rows)."""
    def fake_model(prompt: str) -> str:
        return "56 degrees Celsius."

    bundle = qa.answer_question(
        "Boiling point of acetone?",
        SPEC_EMBEDDED,
        min_score=0.99,  # tier 1 (cosine) deliberately impossible
        ask=fake_model,
    )
    assert bundle.not_found is False
    assert bundle.reason == "ok"


def test_weak_cosine_without_lexical_evidence_still_rejected():
    bundle = qa.answer_question(
        "How do I reset the heater?", EMBEDDED, min_score=0.99, ask=fail_if_called
    )
    assert bundle.not_found is True
    assert bundle.reason == "low_relevance"
    assert bundle.sources == []


# --- Stage 5L: borderline band - the model validates answerability ---------------


def _mocked_result(score: float, text: str) -> SearchResult:
    return SearchResult("manual", 2, text, 10, 10 + len(text), score)


def test_borderline_passage_is_validated_by_the_model_not_auto_rejected():
    """Gold passage inside the top-1 chunk at cosine 0.23 (measured q050):
    just below the gate. The model - not a threshold - decides, and its
    grounded answer is accepted."""
    seen = []

    def fake_model(prompt: str) -> str:
        seen.append(prompt)
        return "Deliveries take five working days."

    weak_but_gold = _mocked_result(0.23, "Deliveries take five working days.")
    with mock.patch.object(qa, "search_chunks", return_value=[weak_but_gold]):
        bundle = qa.answer_question(
            "How long do standard deliveries take?", [], ask=fake_model
        )
    assert bundle.not_found is False
    assert bundle.reason == "ok"
    assert len(seen) == 1


def test_borderline_passage_refused_by_model_is_reported_honestly():
    calls = []

    def refusing_model(prompt: str) -> str:
        calls.append(prompt)
        return qa.NOT_FOUND_MARKER

    weak = _mocked_result(0.23, "Order consumables with the exact part number.")
    with mock.patch.object(qa, "search_chunks", return_value=[weak]):
        bundle = qa.answer_question(
            "What is the price of the blue valve?", [], ask=refusing_model
        )
    assert bundle.not_found is True
    assert bundle.reason == "model_no_support"
    assert bundle.sources == []


def test_passage_below_borderline_window_is_rejected_without_model():
    def fail(prompt: str) -> str:
        raise AssertionError("model must not be called with no evidence at all")

    hopeless = _mocked_result(0.10, "Some unrelated text.")
    with mock.patch.object(qa, "search_chunks", return_value=[hopeless]):
        bundle = qa.answer_question("What is the price?", [], ask=fail)
    assert bundle.not_found is True
    assert bundle.reason == "low_relevance"


# --- Stage 5L: grounding check ---------------------------------------------------


def test_answer_with_invented_number_is_discarded():
    def model_invents(prompt: str) -> str:
        return "The boiling point of acetone is 101 degrees Celsius."  # 101 not in evidence

    bundle = qa.answer_question(
        "Boiling point of acetone?", SPEC_EMBEDDED, ask=model_invents
    )
    assert bundle.not_found is True
    assert bundle.reason == "ungrounded_answer"
    assert bundle.sources == []


def test_word_number_paraphrase_is_not_rejected():
    def model_spells_out(prompt: str) -> str:
        return "The boiling point of acetone is fifty-six degrees."

    bundle = qa.answer_question(
        "Boiling point of acetone?", SPEC_EMBEDDED, ask=model_spells_out
    )
    assert bundle.not_found is False


def test_ungrounded_numbers_boundary_aware():
    context = "jumbo frame size 9216 bytes; accuracy 0.8 or higher; range 19 - 30 VDC"
    assert qa._ungrounded_numbers("size is 9216 bytes.", context) == []
    assert qa._ungrounded_numbers("accuracy is 0.8.", context) == []
    assert qa._ungrounded_numbers("range is 19-30 VDC.", context) == []
    assert qa._ungrounded_numbers("weight is 174 g.", context) == ["174"]
    # "9" must not match inside "9216"
    assert qa._ungrounded_numbers("exactly 9 units.", context) == ["9"]
    # a sentence-final period does not break matching (BV-100.)
    assert qa._ungrounded_numbers("part number BV-100.", "blue valve BV-100.") == []


def test_ungrounded_numbers_accepts_comma_and_point_decimals():
    context = "specific gravity minimum 1.5"
    assert qa._ungrounded_numbers("gravity is 1,5", context) == []
    assert qa._ungrounded_numbers("gravity is 1.5", context) == []
    assert qa._ungrounded_numbers("gravity is 2.5", context) == ["2.5"]


def test_spoken_numbers_in_context_are_groundable():
    # The document spells values out; the answer may write digits. Same VALUE.
    assert qa._ungrounded_numbers(
        "more than 0.2 bar", "leak fails if the drop is greater than zero point two bar"
    ) == []
    assert qa._ungrounded_numbers(
        "-40 to +300 degrees Celsius",
        "probes are rated minus forty to plus three hundred degrees Celsius",
    ) == []
    assert qa._ungrounded_numbers(
        "exceeded 95 degrees", "the heater exceeded ninety five degrees"
    ) == []
    assert qa._ungrounded_numbers(
        "accuracy is plus or minus 1.5 degrees",
        "accurate to plus or minus one point five degrees",
    ) == []


def test_spoken_number_converter_forms():
    convert = qa._spoken_numbers_to_digits
    assert convert("zero point two bar") == "0.2 bar"
    assert convert("ninety five degrees") == "95 degrees"
    assert convert("three hundred units") == "300 units"
    assert convert("forty to fifty") == "40 to 50"
    assert convert("one point five degrees") == "1.5 degrees"
    # non-numeric text is untouched
    assert convert("no numbers in this phrase") == "no numbers in this phrase"


# --- Stage 5L: source prioritization ---------------------------------------------


def test_sources_prioritize_evidence_bearing_passages():
    catalogue = SearchResult("catalogue", 40, "Our range includes gauges and valves.", 0, 44, 0.61)
    evidence = SearchResult("spec", 7, "Float Size: 73 x 240h. Weight 174 g.", 0, 36, 0.55)
    out = qa.prioritize_sources("The weight is 174 g.", [catalogue, evidence])
    assert out[0] is evidence
    assert out[1] is catalogue  # generic passage stays listed, just later


def test_prioritize_sources_no_numbers_keeps_order():
    catalogue = SearchResult("catalogue", 40, "Our range includes gauges and valves.", 0, 44, 0.61)
    evidence = SearchResult("spec", 7, "Wipe the enclosure with a damp cloth.", 0, 37, 0.55)
    sources = [catalogue, evidence]
    assert qa.prioritize_sources("Wipe it with a damp cloth.", sources) is sources


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
