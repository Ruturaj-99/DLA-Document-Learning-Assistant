"""Stage 5M: evidence-to-answer improvements.

These tests pin the two measured real-PDF failure modes:

1. A dense Annexure row that literally contains the asked-for tag is buried
   by the hybrid ranking (vector rank 59 / BM25 rank 32 on a real datasheet),
   so the model never sees it. `select_evidence` adds that chunk.
2. A model refusal on a passage that does hold the answer used to be final
   unless the lexical coverage tier passed. Stage 5M re-reads the same
   evidence once with a neutral prompt; a second refusal is still honoured.

The model is always a fake `ask=`, so no Ollama server is needed.
"""

import sys
import unittest.mock as mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import qa  # noqa: E402
from src.chunking import Chunk  # noqa: E402
from src.embeddings import embed_chunks  # noqa: E402
from src.retrieval import SearchResult  # noqa: E402

TAG_ROW = (
    "1 PT-Seal GQB01CP001 0 - 3.55 kg/cm2 "
    "266HRH.M.R.R.A.8-..L1.B7 + S26RA.H.E1.C.F.SM.1.A.N.N.N 1"
)
ACCURACY_PAGE = (
    "Accuracy rating\nF from 1:1 to 10:1 +/- 0.06 %\n"
    "F from 10:1 to 60:1 +/- (0.006 x TD) %"
)
ANNEXURE = (
    "ANNEXURE IV\n"
    "SI NO. TYPE TAG NO. CALIBRATION RANGE UNIT QUANTITY\n"
    "1 PT-Seal GQB01CP001 0 - 3.55 kg/cm2 1\n"
    "2 PT-Seal GQB01CP002 0 - 3.55 kg/cm2 1\n"
    "3 PT-Seal GQB02CP001 0 - 3.3 kg/cm2 1\n"
)

CHUNKS = [
    Chunk("datasheet", 10, ACCURACY_PAGE, 0, len(ACCURACY_PAGE)),
    Chunk("datasheet", 3, ANNEXURE, 0, len(ANNEXURE)),
    Chunk("manual", 7, "The seal diaphragm isolates the process fluid while the "
          "filled capillary tube hydraulically transmits the pressure.", 0, 110),
    Chunk("manual", 1, "Storage: keep the transmitter in a dry place.", 0, 50),
]
EMBEDDED = embed_chunks(CHUNKS)


def _result(chunk: Chunk, score: float) -> SearchResult:
    return SearchResult(
        chunk.document_id,
        chunk.page_number,
        chunk.text,
        chunk.start_char,
        chunk.end_char,
        score,
    )


# --- Threshold 1: exact literal evidence salvage --------------------------------


def test_exact_tag_salvage_adds_the_buried_chunk_before_the_prompt():
    base = _result(CHUNKS[0], 0.44)  # accuracy page, not the tag row
    seen = []

    def fake_model(prompt: str) -> str:
        seen.append(prompt)
        return "0 - 3.55 kg/cm2"

    with mock.patch.object(qa, "search_chunks", return_value=[base]):
        bundle = qa.answer_question(
            "What is the calibration range of tag GQB01CP001, including its unit?",
            EMBEDDED,
            ask=fake_model,
        )

    assert bundle.reason == "ok"
    assert "GQB01CP001" in seen[0]
    # The salvaged Annexure chunk is first, so the small model reads it first.
    assert bundle.sources[0].page_number == 3
    assert "0 - 3.55 kg/cm2" in bundle.sources[0].text


def test_rare_section_phrase_salvage_finds_annexure_iv():
    base = _result(CHUNKS[0], 0.57)

    def fake_model(prompt: str) -> str:
        return "There are 3 pressure transmitters listed, one of each tag."

    with mock.patch.object(qa, "search_chunks", return_value=[base]):
        bundle = qa.answer_question(
            "Which pressure transmitters are listed in Annexure IV, "
            "and what is the quantity specified for each tag?",
            EMBEDDED,
            ask=fake_model,
        )

    assert bundle.reason == "ok"
    assert bundle.sources[0].page_number == 3


def test_common_phrases_do_not_add_evidence():
    base = _result(CHUNKS[1], 0.30)
    evidence = qa.select_evidence(
        "How many pressure transmitters are listed?", [base], EMBEDDED
    )
    assert evidence == [base]


# --- Threshold 2: one neutral re-read after a refusal --------------------------


def test_low_lexical_coverage_refusal_gets_one_neutral_retry():
    base = _result(CHUNKS[2], 0.62)
    calls = []

    def fake_model(prompt: str) -> str:
        calls.append(prompt)
        if len(calls) == 1:
            return qa.NOT_FOUND_MARKER
        return (
            "The diaphragm isolates the process fluid, and the filled "
            "capillary tube hydraulically transmits the pressure."
        )

    with mock.patch.object(qa, "search_chunks", return_value=[base]):
        bundle = qa.answer_question(
            "Explain how the remote seal moves process pressure to the sensor.",
            EMBEDDED,
            ask=fake_model,
        )

    assert len(calls) == 2
    assert bundle.reason == "ok"
    assert bundle.sources
    # The retry must NOT claim the answer is present: on an unanswerable
    # question that assertion coaches the model to invent one.
    assert "IS present" not in calls[1]
    assert "Re-read every passage" in calls[1]


def test_second_refusal_is_still_an_honest_not_found():
    base = _result(CHUNKS[0], 0.60)
    calls = []

    def always_refuses(prompt: str) -> str:
        calls.append(prompt)
        return qa.NOT_FOUND_MARKER

    with mock.patch.object(qa, "search_chunks", return_value=[base]):
        bundle = qa.answer_question(
            "What is the price of this transmitter?", EMBEDDED, ask=always_refuses
        )

    assert len(calls) == 2
    assert bundle.not_found is True
    assert bundle.reason == "model_no_support"
    assert bundle.sources == []


# --- Prompt shape ---------------------------------------------------------------


def test_prompt_states_exact_values_table_rows_and_multi_part_answers():
    prompt = qa._build_prompt(
        "What are the output signal and supply voltage?",
        [_result(CHUNKS[0], 0.5)],
    )
    assert "ONLY the numbered reference passages" in prompt
    assert qa.NOT_FOUND_MARKER in prompt
    assert "exactly as written" in prompt
    assert "row that matches the tag" in prompt
    assert "several things" in prompt
    assert "Do not quote whole passages" in prompt
    assert "do not guess a count" in prompt
    assert "never begin with a bracketed number" in prompt


# --- Grounding variants ---------------------------------------------------------


def test_thousands_separators_are_not_hallucinations():
    assert qa._ungrounded_numbers("weight is 1200 kg", "weight 1 200 kg") == []
    assert qa._ungrounded_numbers("weight is 1200 kg", "weight 1,200 kg") == []
    assert qa._ungrounded_numbers("power is 2.3 W", "power 2.3 w") == []
    assert qa._ungrounded_numbers("power is 4.6 W", "power 2.3 w") == ["4.6"]


def test_long_invented_codes_are_rejected_but_short_labels_are_not():
    context = "Model 266HRH with order code GQB01CP001; valves EV5 and CEB1."
    assert qa._ungrounded_codes("model 266HRH", context) == []
    assert qa._ungrounded_codes("code GQB01CP001", context) == []
    assert qa._ungrounded_codes("valves EV5 and CEB1", context) == []
    assert qa._ungrounded_codes("code GQB99CP001", context) == ["GQB99CP001"]


# --- Unsupported attribute guard -------------------------------------------------


def test_short_row_number_is_not_accepted_as_a_serial_number():
    row = _result(CHUNKS[1], 0.60)  # tag row; its only nearby number is quantity 1

    def says_quantity(prompt: str) -> str:
        return "1"

    with mock.patch.object(qa, "search_chunks", return_value=[row]):
        bundle = qa.answer_question(
            "What is the serial number of the pressure transmitter with tag "
            "GQB01CP001?",
            EMBEDDED,
            ask=says_quantity,
        )

    assert bundle.not_found is True
    assert bundle.reason == "unsupported_attribute"
    assert bundle.sources == []


def test_real_numeric_identifier_and_count_answers_are_kept():
    # A four-digit identifier is not a likely row index / quantity.
    assert not qa._is_unverified_short_identifier(
        "What is the serial number of tag GQB01CP001?", "Serial number: 4821."
    )
    # Counts are legitimate bare numbers and must not be affected.
    assert not qa._is_unverified_short_identifier(
        "How many pressure transmitters are listed in Annexure IV?", "14"
    )


def test_guessed_row_count_is_discarded_but_stated_count_is_kept():
    rows = (
        "1 PT-Seal A 0 - 3 kg/cm2 1\n"
        "2 PT-Seal B 0 - 3 kg/cm2 1\n"
        "3 PT-Seal C 0 - 3 kg/cm2 1\n"
        "4 PT-Seal D 0 - 3 kg/cm2 1\n"
        "SI NO. TYPE TAG NO. QUANTITY"
    )
    assert qa._is_unverified_count(
        "How many pressure transmitters are listed?", "There are 4.", rows
    )
    # The document states this count explicitly, so the answer is supported.
    cycles = "NO OF CYCLES : 10 appx."
    assert not qa._is_unverified_count(
        "How many automatic sequence cycles are performed?", "10 cycles.", cycles
    )


def test_identifier_and_inline_count_regressions_are_not_discarded():
    # "part number of" contains "number of" but is an identifier question.
    assert not qa._is_unverified_count(
        "What is the part number of the heater element?",
        "HX-2207",
        "The part number HX-2207 is printed on the heater.",
    )
    # The count is stated inline beside the counted noun, with extra words.
    assert not qa._is_unverified_count(
        "How many Ethernet ports and at what speeds does the TSW114 operate?",
        "5 Ethernet ports at 10/100/1000 Mbps",
        "5 x eth ports, 10/100/1000 mbps",
    )
    # A worded count has no number to verify - leave it to normal grounding.
    assert not qa._is_unverified_count(
        "How many SIM cards does the SW-245 support?", "Dual SIM", "dual-sim"
    )

