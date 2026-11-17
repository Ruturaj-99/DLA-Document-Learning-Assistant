"""RAG question answering.

Stage 4C of the AI Document Learning Assistant.

In simple words, this module turns "here are search results" into
"here is a written answer, and here are the sources it came from".

The pipeline:
    1. Search the document chunks for passages similar to the question (Stage 3C).
       -> Answerability is decided by EVIDENCE, not by one number: the gate
          passes when the best passage is meaning-similar (cosine gate), OR
          when the passages literally contain enough of the question's rare
          terms (lexical evidence tier), OR - for passages just below the
          gate - the model itself validates under the strict passages-only
          prompt (borderline band, Stage 5L). A BM25-led low-cosine passage
          with the exact spec wording is therefore not rejected.
    2. Build a short "cheat sheet": numbered passages, each labeled with its
       document and page.
    3. Ask the local model to answer ONLY from that cheat sheet, and to reply
       with a special marker phrase if the passages don't contain the answer.
    4. Validate refusals against the evidence (Stage 5L): the model may be
       wrong when it claims the passages are insufficient. If the lexical
       evidence says otherwise, we ask once more, explicitly. If it still
       refuses, we honestly report "not found" - we never force an answer.
    5. Ground the answer (Stage 5L): every number in the model's reply must
       appear in the retrieved passages. A reply that invents values is
       discarded (safe abstention) instead of shown with fake citations.
    6. Attach citations USING OUR CODE, not the model's words: every passage
       we handed over already knows its document, page, and exact characters
       (guaranteed since Stage 3A). The model therefore cannot invent page
       numbers we never gave it. Passages that carry the answer's key values
       are listed first, generic catalogue-style passages after them.

IMPORTANT: this module never shows the model anything but the numbered
passages, and never trusts the model's own page references.
"""

import re
from dataclasses import dataclass

from src.bm25 import BM25Index, tokenize
from src.embeddings import EmbeddedChunk
from src.ollama import ask_ollama
from src.retrieval import SearchResult, search_chunks

# What the model must reply when the passages don't contain the answer.
# A marker phrase (instead of free wording "I don't know") keeps the check
# simple and reliable - we look for exactly this string.
NOT_FOUND_MARKER = "ANSWER_NOT_IN_DOCUMENTS"

# Small local models do not always reproduce the marker exactly: the same
# llama3.2:3b was observed replying "NO ANSWER_IN_DOCUMENTS". Treating only
# the exact marker as a refusal would leak that variant as a REAL answer,
# complete with citations. These patterns cover the observed and plausible
# refusal wordings; they only match phrases no factual answer would contain.
_REFUSAL_PATTERN = re.compile(
    r"("
    r"ANSWER\s*_?\s*NOT\s*_?\s*IN|"          # ANSWER_NOT_IN_DOCUMENTS / not in
    r"\bNO\s+ANSWER|"                        # NO ANSWER_IN_DOCUMENTS
    r"NOT\s+FOUND\s+IN\b|"                   # "not found in the passages"
    r"NOT\s+FOUND\s+IN\s+THE\b|"
    r"\bCANNOT\s+(?:FIND|ANSWER|DETERMINE)\b|"
    r"\bNOT\s+ENOUGH\s+(?:INFORMATION|DETAIL)|"
    r"INSUFFICIENT\s+INFORMATION|"
    r"DOES\s+NOT\s+CONTAIN\s+THE\s+ANSWER|"
    r"DO\s+NOT\s+CONTAIN\s+THE\s+ANSWER"
    r")",
    re.IGNORECASE,
)

# Below this similarity score, the best passage looks unrelated in MEANING -
# unless the lexical evidence tier rescues it (see below). Passage-question
# pairs that truly match typically score 0.5+; unrelated pairs stay near
# 0.0-0.2 (measured in Stage 3B tests).
DEFAULT_MIN_SCORE = 0.25

# Borderline window (Stage 5L): a passage scoring within BORDERLINE_WINDOW
# below the gate is NOT auto-rejected. It is sent to the model under the
# strict passages-only prompt, and the MODEL validates answerability - the
# same treatment passages scoring above the gate already get. Its refusal is
# trusted (unless the lexical tier overrides), and its answer must pass the
# grounding check. Measured case: q050/q055 - the gold passage sits INSIDE
# the top-1 chunk at cosine 0.21-0.23, lexically too thin for any threshold
# that still keeps the unanswerable questions out (their lexical coverage
# reaches 0.36-0.58, HIGHER than these). The window is relative to min_score
# so min_score stays the single tuning knob.
BORDERLINE_WINDOW = 0.10

# Lexical evidence tier (Stage 5L): the share of IDF-weighted question tokens
# (after document-frequency pruning) that the best retrieved passage contains.
# Measured over both evaluation suites: answerable questions reach 0.36+
# only when the cosine gate already passes; NO unanswerable question ever
# exceeded 0.36 (max seen 0.358). 0.5 keeps a safe margin on both sides:
# a passage containing at least half of the question's rare terms is real
# evidence, a passage matching a third of them is not.
ANSWERABILITY_MIN_COVERAGE = 0.5


@dataclass
class AnswerBundle:
    """The result of one question, with its supporting sources.

    Attributes:
        question: the question that was asked.
        answer: the model's answer text (or an honest "not found" message).
        sources: the passages that support the answer, each with document
            id, page number, exact character positions and its similarity
            score. EMPTY whenever we could not confirm an answer.
        not_found: True when no passage support could be established.
        reason: "ok" | "no_passages" | "low_relevance" | "model_no_support"
            | "ungrounded_answer".
    """

    question: str
    answer: str
    sources: list[SearchResult]
    not_found: bool
    reason: str


def _not_found_bundle(question: str, reason: str) -> AnswerBundle:
    """One uniform honest response for all "could not answer" cases.

    Sources are deliberately EMPTY so the UI can never accidentally show
    a highlight or citation for an answer we could not verify.
    """
    return AnswerBundle(
        question=question,
        answer="I could not find an answer to this in the uploaded documents.",
        sources=[],
        not_found=True,
        reason=reason,
    )


def looks_like_refusal(text: str) -> bool:
    """True when the model's reply says the passages lack the answer.

    Checks the exact marker AND the observed/hedged variants; the check is
    deliberately one-way: it can only ever turn a reply INTO a refusal,
    never turn a refusal into an answer.
    """
    if not text:
        return False
    if NOT_FOUND_MARKER in text.upper():
        return True
    return bool(_REFUSAL_PATTERN.search(text))


def best_token_coverage(question: str, texts: list[str]) -> float:
    """IDF-weighted share of the question's discriminative tokens that the
    best single text contains (0.0-1.0).

    Uses the same statistical machinery as retrieval: tokens present in
    most of the texts are pruned (they cannot separate anything), the
    remaining tokens are weighted by rarity. No term lists, no
    document-specific rules.
    """
    if not texts:
        return 0.0
    index = BM25Index(texts)
    kept = set(index.question_tokens(question))
    if not kept:
        return 0.0
    total_weight = sum(index.idf(token) for token in kept)
    if total_weight <= 0:
        return 0.0
    best = 0.0
    for text in texts:
        tokens = set(tokenize(text))
        got = sum(index.idf(token) for token in kept if token in tokens)
        best = max(best, got / total_weight)
    return best


def _is_answerable_evidence(
    question: str, results: list[SearchResult], min_score: float
) -> bool:
    """Evidence-based answerability decision (Stage 5L).

    Tier 1 (meaning): the best passage is cosine-similar enough.
    Tier 2 (lexicon): the best passage literally contains enough of the
    question's rare terms - this rescues BM25-led passages whose cosine
    is low (dense spec rows, contents pages) without touching the
    threshold itself.
    Tier 3 (borderline): a passage just below the gate goes to the model,
    which validates answerability under the strict passages-only prompt.
    Below that window there is no usable evidence and the model is
    never called.
    """
    if results[0].score >= min_score:
        return True
    coverage = best_token_coverage(question, [r.text for r in results])
    if coverage >= ANSWERABILITY_MIN_COVERAGE:
        return True
    return results[0].score >= min_score - BORDERLINE_WINDOW


_NUMBER_PATTERN = re.compile(r"\d+(?:[.,]\d+)?")

# Documents often spell values out ("greater than zero point two bar") while
# an answer writes the same value as digits ("0.2 bar") - a legitimate
# conversion, not a hallucination. The grounding check therefore compares
# VALUES: spoken number words are normalised to digits on both sides first.
_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
}


def _spoken_numbers_to_digits(text: str) -> str:
    """Digit-form the spelled-out numbers in `text`.

    Handles the forms measured in real documents and answers:
    decimals ("one point five" -> "1.5"), tens with units
    ("ninety five" -> "95"), hundreds ("three hundred" -> "300") and
    remaining single words ("forty" -> "40"). Longest/most specific
    patterns run first so compounds are not split.
    """
    units = "|".join(_UNITS)
    tens = "|".join(_TENS)
    result = re.sub(
        rf"\b({units})\s+point\s+({units})\b",
        lambda m: f"{_UNITS[m.group(1)]}.{_UNITS[m.group(2)]}",
        text,
        flags=re.IGNORECASE,
    )
    result = re.sub(
        rf"\b({tens})(?:\s+|-)({units})\b",
        lambda m: str(_TENS[m.group(1).casefold()] + _UNITS[m.group(2).casefold()]),
        result,
        flags=re.IGNORECASE,
    )
    result = re.sub(
        rf"\b({units})\s+hundred\b",
        lambda m: str(_UNITS[m.group(1).casefold()] * 100),
        result,
        flags=re.IGNORECASE,
    )
    result = re.sub(
        rf"\b({units}|{tens})\b",
        lambda m: str(
            _UNITS[m.group(1).casefold()]
            if m.group(1).casefold() in _UNITS
            else _TENS[m.group(1).casefold()]
        ),
        result,
        flags=re.IGNORECASE,
    )
    return result


def _ungrounded_numbers(answer: str, context_text: str) -> list[str]:
    """Numbers in the answer that appear NOWHERE in the retrieved passages.

    These are the hallucination-prone part of an answer: a small local
    model can paraphrase words freely (that is fine) but must not invent
    values. Matching is boundary-aware, so "9" does not match inside
    "9216", comma/point decimal variants are both accepted, and spoken
    number words in the passage count as their digit form. Answers
    written in words ("five seconds") carry no digits and are never
    rejected by this check.
    """
    context = " ".join(_spoken_numbers_to_digits(context_text).split()).casefold()
    missing: list[str] = []
    for raw in _NUMBER_PATTERN.findall(answer):
        variants = {raw, raw.replace(",", ".")}
        if any(
            re.search(rf"(?<!\d){re.escape(v)}(?!\d)", context) for v in variants
        ):
            continue
        missing.append(raw)
    return missing


def _numbers_in_text(text: str) -> list[str]:
    return _NUMBER_PATTERN.findall(text)


def prioritize_sources(
    answer: str, sources: list[SearchResult]
) -> list[SearchResult]:
    """Evidence-bearing passages first, generic catalogue passages after.

    A passage that contains one of the answer's key values (its numbers,
    boundary-matched) is project-specific evidence for THIS answer; a
    passage that merely shares topic words is generic. Both stay in the
    list (the answer is still grounded in all of them); only the display
    order changes. Stable: ties keep the fused ranking order.
    """
    keys = {value.casefold() for value in _numbers_in_text(answer)}
    if not keys or not sources:
        return sources

    def has_evidence(source: SearchResult) -> bool:
        text = " ".join(_spoken_numbers_to_digits(source.text).split()).casefold()
        return any(
            re.search(rf"(?<!\d){re.escape(key)}(?!\d)", text) for key in keys
        )

    with_evidence = [s for s in sources if has_evidence(s)]
    if not with_evidence or len(with_evidence) == len(sources):
        return sources
    return with_evidence + [s for s in sources if not has_evidence(s)]


def build_context_block(results: list[SearchResult]) -> str:
    """Render the search results as the numbered cheat sheet for the model.

    Example line:
        [1] (document manual-heater, page 2) To reset the heater, hold ...
    """
    lines = []
    for number, result in enumerate(results, start=1):
        lines.append(
            f"[{number}] (document {result.document_id}, page {result.page_number}) "
            f"{result.text}"
        )
    return "\n".join(lines)


def _build_prompt(question: str, results: list[SearchResult]) -> str:
    return (
        "You answer questions using ONLY the numbered reference passages below.\n"
        "Rules:\n"
        "- Use the passages, not outside knowledge.\n"
        f"- If the passages do not contain the answer, reply exactly "
        f"{NOT_FOUND_MARKER} and nothing else.\n"
        "- Answer in plain text, briefly.\n"
        "- Do NOT mention page numbers or document names yourself.\n"
        "\nReference passages:\n"
        f"{build_context_block(results)}\n"
        f"\nQuestion: {question}"
    )


def _build_retry_prompt(question: str, results: list[SearchResult]) -> str:
    """Second-chance prompt for a model that wrongly claimed insufficiency.

    It states (truthfully - we only send it when the lexical evidence tier
    passed) that the information IS present, but keeps the refusal marker
    available, so the model is never forced to answer.
    """
    return (
        "You answer questions using ONLY the numbered reference passages below.\n"
        "A previous reply wrongly claimed the passages were insufficient. "
        "Re-check them carefully: the information needed to answer IS present "
        "in the passages.\n"
        "Rules:\n"
        "- Use the passages, not outside knowledge.\n"
        "- Answer in plain text, briefly.\n"
        "- Do NOT mention page numbers or document names yourself.\n"
        f"- Only if, after re-checking, the passages really do not contain "
        f"it, reply exactly {NOT_FOUND_MARKER} and nothing else.\n"
        "\nReference passages:\n"
        f"{build_context_block(results)}\n"
        f"\nQuestion: {question}"
    )


def answer_question(
    question: str,
    embedded_chunks: list[EmbeddedChunk],
    top_k: int = 4,
    min_score: float = DEFAULT_MIN_SCORE,
    ask=None,
) -> AnswerBundle:
    """Answer a question from uploaded documents, with verified sources.

    Args:
        question: the user's question.
        embedded_chunks: document chunks with vectors (Stage 3B output).
        top_k: how many passages to show the model (Stage 3C handles safe capping).
        min_score: cosine gate before we trust retrieval (see DEFAULT_MIN_SCORE;
            the lexical evidence tier can rescue lower-cosine passages).
        ask: optional replacement for the model call (used by tests). Must map
            a prompt string to an answer string, like ollama.ask_ollama does.

    Returns:
        An AnswerBundle.

    Raises:
        ValueError: for an empty/blank question.
        RuntimeError: propagates connection problems from src.ollama - the
            messages there tell the user how to fix them.
    """
    if not question.strip():
        raise ValueError("Please ask a non-empty question.")

    if ask is None:
        ask = ask_ollama  # the real local model (Step 4B bridge)

    # --- 1. Find the best passages (Stage 3C) ------------------------------
    results = search_chunks(question, embedded_chunks, top_k=top_k)
    if not results:
        return _not_found_bundle(question, reason="no_passages")

    # --- 2. Evidence-based answerability (Stage 5L) -------------------------
    if not _is_answerable_evidence(question, results, min_score):
        return _not_found_bundle(question, reason="low_relevance")

    prompt = _build_prompt(question, results)

    # --- 3. Ask the local model (Step 4B handles connection errors) ---------
    raw_answer = ask(prompt).strip()

    # --- 4. Validate refusals against the evidence (Stage 5L) ---------------
    # The model may wrongly claim the passages are insufficient (small local
    # models do). When the lexical evidence tier says the passages DO contain
    # the question's rare terms, ask once more, explicitly - the retry keeps
    # the refusal marker available, so the model is never forced.
    if looks_like_refusal(raw_answer):
        coverage = best_token_coverage(question, [r.text for r in results])
        if coverage < ANSWERABILITY_MIN_COVERAGE:
            return _not_found_bundle(question, reason="model_no_support")
        raw_answer = ask(_build_retry_prompt(question, results)).strip()
        if looks_like_refusal(raw_answer):
            return _not_found_bundle(question, reason="model_no_support")

    # --- 5. Grounding check (Stage 5L): no invented values ------------------
    context_text = " ".join(result.text for result in results)
    if _ungrounded_numbers(raw_answer, context_text):
        return _not_found_bundle(question, reason="ungrounded_answer")

    # --- 6. Attach OUR citations, evidence-bearing passages first -----------
    return AnswerBundle(
        question=question,
        answer=raw_answer,
        sources=prioritize_sources(raw_answer, results),
        not_found=False,
        reason="ok",
    )
