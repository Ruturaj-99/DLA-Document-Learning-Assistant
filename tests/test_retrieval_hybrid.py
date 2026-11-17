"""Stage 5J: hybrid retrieval tests (vector + BM25 + RRF fusion).

The point of hybrid search: embeddings find MEANING but blur exact
values; BM25 finds EXACT words, part numbers and numeric values but
misses paraphrases. Reciprocal Rank Fusion combines the two rankings
without any per-document tuning.

These tests pin down:
    - BM25 behaviour in isolation (tokenization, IDF, ranking)
    - fusion rescuing exact-term questions that pure vector search blurs
    - fusion keeping meaning-based ranking when there is no keyword hook
    - the returned `score` keeping its cosine meaning (QA gate safety)
    - Stage 5F TOC filtering still holding for both retrievers
    - determinism, top_k handling and empty-input edge cases

Run from the project root with:
    .venv\\Scripts\\python -m pytest tests/test_retrieval_hybrid.py -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bm25 import BM25Index, tokenize  # noqa: E402
from src.chunking import Chunk  # noqa: E402
from src.embeddings import embed_chunks  # noqa: E402
from src.retrieval import (  # noqa: E402
    BOILERPLATE_LINE_THRESHOLD,
    BOILERPLATE_PENALTY,
    RRF_BM25_WEIGHT,
    RRF_K,
    RRF_VECTOR_WEIGHT,
    boilerplate_factors,
    is_table_of_contents,
    search_chunks,
)


# --- BM25 unit tests -----------------------------------------------------------


def test_tokenize_keeps_words_numbers_and_codes():
    assert tokenize("Output: 4-20 mA; model TSW114000000") == [
        "output",
        "4",
        "20",
        "ma",
        "model",
        "tsw114000000",
    ]


def test_bm25_ranks_exact_term_first():
    index = BM25Index(
        [
            "The relay module switches loads up to 10 amperes.",
            "The jumbo frame size is 9216 bytes.",
            "Frame relay is a legacy WAN technology.",
        ]
    )
    ranked = index.ranked("What is the jumbo frame size?")
    assert ranked[0] == 1  # the chunk with "jumbo frame size" verbatim


def test_bm25_ignores_documents_without_any_overlap():
    index = BM25Index(["alpha beta gamma", "delta epsilon zeta"])
    scores = index.score("totally unrelated tokens here")
    assert scores == [0.0, 0.0]
    assert index.ranked("totally unrelated tokens here") == []


def test_common_question_tokens_are_pruned_from_scoring():
    """A token present in every document is not discriminative: repeated
    headers and boilerplate must not let one chunk match 'more' of the
    question than the chunk that holds the actual spec."""
    header = "ACME DATASHEET SPECIFICATIONS"
    index = BM25Index(
        [
            header + " Relay output 250 VAC contact rating",
            header + " Power supply DC 24V 1A input terminals",
            header + " Output second bank terminals wiring diagram",
        ]
    )
    # "acme"/"datasheet"/"specifications" appear in all 3 documents ->
    # pruned; only "power", "supply", "24v" decide the ranking.
    ranked = index.ranked("What power supply does the ACME datasheet require?")
    assert ranked[0] == 1
    assert index.question_tokens("acme acme acme relay") == ["relay"]


def test_bm25_scores_are_deterministic():
    documents = ["valve opens at 2 bar", "valve closes at 5 bar"] * 3
    first = BM25Index(documents).score("valve 2 bar")
    second = BM25Index(documents).score("valve 2 bar")
    assert first == second


def test_empty_corpus_yields_no_ranking():
    index = BM25Index([])
    assert index.score("anything") == []
    assert index.ranked("anything") == []


# --- Fusion behaviour ------------------------------------------------------------
#
# A corpus where vector search and BM25 disagree, built around one hard fact:

CORPUS = [
    # 0: paraphrase, vector-friendly, but NEVER contains the exact code.
    Chunk(
        "device-manual",
        1,
        "The identification sticker on the housing carries the ordering "
        "information and manufacturing details for the instrument.",
        0,
        130,
    ),
    # 1: exact part number, the BM25 hook. Vector similarity is weak.
    Chunk(
        "device-manual",
        2,
        "Order code TSW114000000 refers to the standard package without "
        "a power supply unit.",
        131,
        210,
    ),
    # 2: unrelated filler.
    Chunk(
        "device-manual",
        3,
        "Wipe the enclosure with a damp cloth during routine maintenance.",
        211,
        280,
    ),
]
EMBEDDED = embed_chunks(CORPUS)


def test_exact_order_code_question_finds_the_code_chunk():
    """'TSW114000000' appears in exactly one chunk; BM25 must pull it up."""
    results = search_chunks(
        "What is the order code TSW114000000?", EMBEDDED, top_k=3
    )
    assert results
    assert any("TSW114000000" in r.text for r in results[:1]), (
        f"expected the exact-code chunk first, got: {[r.text[:60] for r in results]}"
    )


def test_meaning_question_without_keyword_hook_keeps_vector_ranking():
    """With no exact-term hook, fusion must not wreck semantic ranking."""
    results = search_chunks(
        "How should the enclosure be maintained?", EMBEDDED, top_k=3
    )
    assert "maintenance" in results[0].text or "maintained" in results[0].text


def test_chunk_ranked_first_in_both_lists_wins_the_fusion():
    """A passage top-ranked by BOTH retrievers beats a specialist of one."""
    results = search_chunks("reset the heater power button", EMBEDDED_HYBRID, top_k=3)
    assert "reset" in results[0].text and "power button" in results[0].text


EMBEDDED_HYBRID = embed_chunks(
    [
        Chunk("manual", 1, "To reset the heater, hold the power button for five seconds.", 0, 60),
        Chunk("manual", 2, "The heater warms the room using a two-speed fan.", 61, 110),
        Chunk("manual", 3, "Resetting clocks twice a year keeps them accurate.", 111, 160),
    ]
)


def test_scores_keep_cosine_meaning_and_monotonic_order():
    results = search_chunks("What is the order code TSW114000000?", EMBEDDED, top_k=3)
    scores = [r.score for r in results]
    # Monotonic non-increasing (consistent with the fused order)...
    assert scores == sorted(scores, reverse=True)
    # ...and still within cosine range (gate/UI semantics preserved).
    assert all(-1.0 <= score <= 1.0 for score in scores)


def test_fusion_changes_nothing_about_top_k_and_edge_cases():
    assert search_chunks("reset", EMBEDDED, top_k=2) == search_chunks(
        "reset", EMBEDDED, top_k=2
    )
    assert len(search_chunks("reset the heater", EMBEDDED, top_k=100)) == 3
    assert search_chunks("   ", EMBEDDED, top_k=3) == []
    assert search_chunks("anything", [], top_k=3) == []
    with pytest.raises(ValueError):
        search_chunks("question", EMBEDDED, top_k=0)


def test_toc_chunks_are_hidden_for_technical_questions():
    toc_text = (
        "SL. NO. PARTICULARS PAGE NO.\n"
        "1 DATASHEET OF TRANSMITTER 3\n"
        "2 DRAWING OF TRANSMITTER 4 TO 6\n"
        "3 MAINTENANCE INSTRUCTIONS 7\n"
        "Page: 2 of 20"
    )
    assert is_table_of_contents(toc_text)
    chunks = embed_chunks(
        [
            Chunk("doc", 4, toc_text, 0, len(toc_text)),
            Chunk(
                "doc",
                2,
                "Input voltage range 9 - 30 VDC, consumption 0.3 W idle.",
                200,
                260,
            ),
        ]
    )
    results = search_chunks("What is the input voltage range?", chunks, top_k=5)
    assert [r.page_number for r in results] == [2]


def test_toc_question_still_ranks_the_contents_page_first():
    toc_text = (
        "TABLE OF CONTENTS\n"
        "1 Introduction 2\n"
        "2 Technical specifications 5\n"
        "3 Maintenance instructions 9\n"
        "Page: 1 of 12"
    )
    chunks = embed_chunks(
        [
            Chunk("doc", 2, "Technical specifications: accuracy plus/minus 2%.", 300, 350),
            Chunk("doc", 1, toc_text, 0, len(toc_text)),
        ]
    )
    results = search_chunks(
        "Which sections are listed in the table of contents?", chunks, top_k=5
    )
    assert results[0].page_number == 1


def test_rrf_constants_are_the_tuned_standard_values():
    assert RRF_K == 60
    assert RRF_VECTOR_WEIGHT + RRF_BM25_WEIGHT == pytest.approx(1.0)
    assert RRF_VECTOR_WEIGHT > RRF_BM25_WEIGHT  # vector leads, BM25 rescues


# --- Boilerplate demotion (Stage 5K) --------------------------------------------
#
# Real PDFs repeat the same header/footer line on many pages. Those chunks
# cluster in BOTH retriever rankings and can swarm the fused top-k ahead of
# the chunk that actually holds the answer.


def test_boilerplate_constants_are_sane():
    assert BOILERPLATE_LINE_THRESHOLD >= 2
    assert 0.0 < BOILERPLATE_PENALTY < 1.0


def test_boilerplate_factors_clean_text_untouched():
    texts = [
        "alpha beta gamma",
        "delta epsilon zeta",
        "mango papaya fruit",
        "gravel stones cement",
        "winter frost ice",
        "paper ink binding",
    ]
    assert boilerplate_factors(texts) == [1.0] * len(texts)


def test_boilerplate_factors_threshold_respected():
    # A line repeated in fewer than BOILERPLATE_LINE_THRESHOLD texts is NOT
    # page furniture; nothing is demoted.
    shared = "common header line"
    texts = [shared, shared, shared, "unrelated text"]
    assert all(f == pytest.approx(1.0) for f in boilerplate_factors(texts))


def test_boilerplate_factors_scale_by_repeated_character_share():
    shared = "copyright acme networks"
    pure_header = [shared, shared, shared, shared]
    texts = pure_header + ["unique specification row"]
    factors = boilerplate_factors(texts)
    # Chunks made entirely of the repeated line are demoted the most...
    for factor in factors[:4]:
        assert factor == pytest.approx(1.0 - BOILERPLATE_PENALTY)
    # ...a chunk with no repeated lines not at all.
    assert factors[4] == pytest.approx(1.0)


def test_boilerplate_factors_hybrid_chunk_barely_demoted():
    shared = "copyright acme networks"
    texts = [f"{shared}\nspec row {i} value {i}" for i in range(4)]
    factors = boilerplate_factors(texts)
    for factor in factors:
        assert 1.0 - BOILERPLATE_PENALTY < factor < 1.0


def test_repeated_header_chunks_do_not_swarm_the_top_k():
    """A repeated-header chunk that tops BOTH retriever lists must lose to
    the clean exact-match chunk once demotion is applied.

    Construction: 4 chunks consist ONLY of the shared header line
    "alpha beta gamma" (so its tokens sit in 4 of 8 chunks = the df-prune
    threshold, hence still scored), one clean chunk adds more words, and
    three fillers share nothing. The pure header chunk is rank 1 in both
    lists, but its fused score is scaled to 70% while the clean chunk keeps
    100% - so the clean chunk must finish first.
    """
    header = "alpha beta gamma"
    chunks = [
        Chunk("doc", 1, header, 0, 16),
        Chunk("doc", 2, "alpha beta gamma zebra unicorn parade", 17, 54),
        Chunk("doc", 3, header, 55, 71),
        Chunk("doc", 4, header, 72, 88),
        Chunk("doc", 5, header, 89, 105),
        Chunk("doc", 6, "mango papaya fruit salad", 106, 130),
        Chunk("doc", 7, "gravel stones cement mix", 131, 155),
        Chunk("doc", 8, "winter frost ice sheet", 156, 178),
    ]
    embedded = embed_chunks(chunks)
    results = search_chunks("What does alpha beta gamma do?", embedded, top_k=8)
    assert results[0].text == "alpha beta gamma zebra unicorn parade"


def test_demotion_preserves_monotonic_scores_and_gate_semantics():
    header = "alpha beta gamma"
    chunks = [
        Chunk("doc", 1, header, 0, 16),
        Chunk("doc", 2, "alpha beta gamma zebra unicorn parade", 17, 54),
        Chunk("doc", 3, header, 55, 71),
        Chunk("doc", 4, header, 72, 88),
        Chunk("doc", 5, header, 89, 105),
    ]
    embedded = embed_chunks(chunks)
    results = search_chunks("What does alpha beta gamma do?", embedded, top_k=5)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert all(-1.0 <= score <= 1.0 for score in scores)


def test_determinism_across_repeated_calls():
    question = "What is the order code TSW114000000?"
    first = search_chunks(question, EMBEDDED, top_k=3)
    second = search_chunks(question, EMBEDDED, top_k=3)
    assert [(r.page_number, r.start_char, r.score) for r in first] == [
        (r.page_number, r.start_char, r.score) for r in second
    ]
