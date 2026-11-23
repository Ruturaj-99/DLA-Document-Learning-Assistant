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
    4. Select the evidence handed to the model (Stage 5M). The hybrid ranking
       is good at meaning but can bury a chunk that literally contains the
       question's exact tag / model code / rare phrase (dense Annexure tables
       score low for both vector and BM25). When the question carries such a
       literal, the chunk(s) containing it are added to the context - the
       retrieval ranking itself is untouched.
    5. Validate refusals against the evidence (Stage 5L/5M): the model may be
       wrong when it claims the passages are insufficient. After a refusal
       we re-read the SAME passages once with a focused, neutral prompt. If
       it still refuses, we honestly report "not found" - we never force an
       answer.
    6. Ground the answer (Stage 5L): every number (and long alphanumeric
       code) in the model's reply must appear in the passages we handed
       over. A reply that invents values is discarded (safe abstention)
       instead of shown with fake citations.
    7. Attach citations USING OUR CODE, not the model's words: every passage
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
from src.embeddings import (
    EmbeddedChunk,
    cosine_similarity,
    embed_question,
)
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

# Exact-literal evidence salvage (Stage 5M). Hybrid RRF is built for meaning,
# so a dense table that literally holds the asked-for tag/model code can lose
# to prose about the same topic: measured on a real 29-page datasheet,
# "calibration range of tag GQB01CP001" ranked the Annexure-IV row at vector
# rank 59 / BM25 rank 32, while the accuracy table (which does not contain the
# tag) filled the top-k. The tag row is unambiguous evidence, so QA adds the
# chunk(s) that contain such a literal to the context. Only literals that are
# rare in the document are used, and at most two chunks are added; the
# retrieval ranking and the answerability gate are unchanged.
EVIDENCE_SALVAGE_MAX_ADDED = 2
EVIDENCE_SALVAGE_MAX_MATCHES = 3
# An identifier must be long enough and mix letters with digits ("GQB01CP001",
# "266HRH", "HX-2207"). A bare number ("3.55") is left to BM25.
_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[./_-][A-Za-z0-9]+)*")
# A rare wording literal ("Annexure IV") can also name the table/section that
# holds the answer, even when it is not a code. Two or three adjacent question
# words are enough; the document-frequency cap below rejects common phrases.
_MIN_PHRASE_CHARS = 10


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
            | "ungrounded_answer" | "unsupported_attribute".
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


def _question_literals(question: str) -> list[str]:
    """Exact literals from the question that can name one evidence chunk.

    Two kinds are collected, most specific first:

    - identifiers that mix letters and digits (tag / model / order codes
      such as "GQB01CP001", "266HRH", "HX-2207");
    - rare adjacent word phrases ("Annexure IV", "seal transfers").

    Whether a literal is actually rare is decided against the document in
    `select_evidence`; this function only proposes candidates.
    """
    literals: list[str] = []

    def add(value: str) -> None:
        value = value.strip("._-/ ").casefold()
        if value and value not in literals:
            literals.append(value)

    for raw in _IDENTIFIER_PATTERN.findall(question):
        token = raw.strip("._-/")
        digits = sum(character.isdigit() for character in token)
        letters = sum(character.isalpha() for character in token)
        # At least two digits keeps ordinary words/dates out; a hyphenated
        # code such as "HX-2207" still counts as one identifier.
        if len(token) >= 4 and digits >= 2 and letters >= 1:
            add(token)

    words = tokenize(question)
    for size in (3, 2):
        for start in range(len(words) - size + 1):
            group = words[start : start + size]
            phrase = " ".join(group)
            if len(phrase) < _MIN_PHRASE_CHARS:
                continue
            if max(len(word) for word in group) < 5:
                continue
            add(phrase)

    return literals


def select_evidence(
    question: str,
    results: list[SearchResult],
    embedded_chunks: list[EmbeddedChunk],
) -> list[SearchResult]:
    """Add rare exact-literal chunks to the evidence, ranking untouched.

    The hybrid ranking optimises meaning, so a dense table row containing
    the exact asked-for tag can lose to prose about the same subject
    (measured: tag row at vector rank 59 / BM25 rank 32). When the question
    carries a literal that occurs in only a few chunks, those chunks are
    unambiguous evidence for THIS question. They are prepended to the
    context (so the small model reads them first) and their real cosine
    score is kept for display. Nothing is reordered inside the retrieval
    results and the answerability gate still sees the original list.
    """
    if not embedded_chunks or not results:
        return results

    literals = _question_literals(question)
    if not literals:
        return results

    normalized = [
        " ".join(re.sub(r"[^a-z0-9]+", " ", chunk.text.casefold()).split())
        for chunk in embedded_chunks
    ]
    seen = {
        (item.document_id, item.page_number, item.start_char, item.end_char)
        for item in results
    }
    added: list[SearchResult] = []
    question_vector = None

    def match_form(value: str) -> str:
        return " ".join(re.sub(r"[^a-z0-9]+", " ", value.casefold()).split())

    for literal in literals:
        needle = match_form(literal)
        if not needle:
            continue
        matches = [
            index for index, text in enumerate(normalized) if needle in text
        ]
        # A common phrase ("pressure transmitter") cannot identify a chunk;
        # a literal that matches nothing cannot help answer this question.
        if not matches or len(matches) > EVIDENCE_SALVAGE_MAX_MATCHES:
            continue
        # Dense spec/annexure rows are short and repeat the literal often;
        # prefer them over a long page that merely mentions the words.
        matches.sort(
            key=lambda index: (
                -normalized[index].count(needle),
                len(normalized[index]),
                index,
            )
        )
        for index in matches:
            chunk = embedded_chunks[index]
            key = (
                chunk.document_id,
                chunk.page_number,
                chunk.start_char,
                chunk.end_char,
            )
            if key in seen:
                continue
            seen.add(key)
            if question_vector is None:
                question_vector = embed_question(question)
            added.append(
                SearchResult(
                    document_id=chunk.document_id,
                    page_number=chunk.page_number,
                    text=chunk.text,
                    start_char=chunk.start_char,
                    end_char=chunk.end_char,
                    score=cosine_similarity(question_vector, chunk.vector),
                )
            )
            break
        if len(added) >= EVIDENCE_SALVAGE_MAX_ADDED:
            break

    return added + results if added else results


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


def _compact_thousands(text: str) -> str:
    """Join digit groups split by a thousands space/comma ("1 200" -> "1200").

    Real datasheets use both "1 200" and "1,200" for the same value, and a
    model may write either form. Only a separator followed by exactly three
    trailing digits is joined, so decimal points ("1.5") are untouched.
    """
    return re.sub(r"(?<=\d)[ ,\u00a0](?=\d{3}(?!\d))", "", text)


def _ungrounded_numbers(answer: str, context_text: str) -> list[str]:
    """Numbers in the answer that appear NOWHERE in the retrieved passages.

    These are the hallucination-prone part of an answer: a small local
    model can paraphrase words freely (that is fine) but must not invent
    values. Matching is boundary-aware, so "9" does not match inside
    "9216", comma/point decimal variants are both accepted, thousands
    separators are ignored on both sides, and spoken number words in the
    passage count as their digit form. Answers written in words ("five
    seconds") carry no digits and are never rejected by this check.
    """
    context = " ".join(
        _compact_thousands(_spoken_numbers_to_digits(context_text)).split()
    ).casefold()
    answer = _compact_thousands(answer)
    missing: list[str] = []
    for raw in _NUMBER_PATTERN.findall(answer):
        variants = {raw, raw.replace(",", ".")}
        if any(
            re.search(rf"(?<!\d){re.escape(v)}(?!\d)", context) for v in variants
        ):
            continue
        missing.append(raw)
    return missing


def _ungrounded_codes(answer: str, context_text: str) -> list[str]:
    """Long alphanumeric codes in the answer absent from the passages.

    Numbers are checked by `_ungrounded_numbers`; this covers invented
    model / ordering / part codes such as "266XYZ" or "GQB99CP001". The
    check is deliberately conservative: a token must be at least six
    characters long and mix letters with at least two digits, so ordinary
    short labels ("EV5", "CEB1", "4-20") are left alone. Punctuation and
    line breaks are ignored on both sides, so OCR line wrapping inside a
    code does not cause a false rejection.
    """
    context_compact = re.sub(r"[^a-z0-9]", "", context_text.casefold())
    missing: list[str] = []
    for raw in _IDENTIFIER_PATTERN.findall(answer):
        token = raw.strip("._-/")
        if len(token) < 6:
            continue
        if sum(character.isdigit() for character in token) < 2:
            continue
        if not any(character.isalpha() for character in token):
            continue
        if re.sub(r"[^a-z0-9]", "", token.casefold()) in context_compact:
            continue
        missing.append(token)
    return missing


def _numbers_in_text(text: str) -> list[str]:
    return _NUMBER_PATTERN.findall(text)


# Attribute questions ask for one named identifier, not a count. When a row
# is missing the asked column, a small model may still grab a nearby number
# from the same row (row index, SI No., quantity). Measured on a real
# Annexure row whose only visible number was its quantity "1": asked for the
# tag's serial number, the model answered "1". A bare one- or two-digit reply
# cannot be verified as the requested identifier, so it is discarded instead
# of being shown as a fact. Longer numeric identifiers (a real 4+ digit
# serial) and alphanumeric codes are still accepted.
_IDENTIFIER_ATTRIBUTE_PATTERN = re.compile(
    r"\b(?:serial|model|part|drawing|tag|order|certificate|asset)\s+"
    r"(?:number|no\.?|code)\b",
    re.IGNORECASE,
)
_SHORT_BARE_NUMBER = re.compile(r"\d{1,2}")


def _is_unverified_short_identifier(question: str, answer: str) -> bool:
    """True when an identifier question was answered with only a 1-2 digit
    number (a likely row index / quantity, not the asked identifier)."""
    if not _IDENTIFIER_ATTRIBUTE_PATTERN.search(question):
        return False
    text = answer.strip().strip(".,;:!?()[]{}\"'")
    return bool(_SHORT_BARE_NUMBER.fullmatch(text))


# Counting rows in a dense table is a known weakness of small local models:
# asked "how many transmitters are listed", llama3.2:3b confidently answered
# "4" from a page holding 14 numbered rows. The prompt already tells it not
# to guess; this check enforces that rule. A count is accepted only when the
# SAME number is stated in the passages with a count label ("NO OF CYCLES:
# 10", "total 14", "quantity ... 1") - never when it is merely a row index or
# a value from an unrelated column.
_COUNT_QUESTION_PATTERN = re.compile(r"\b(?:how many|number of)\b", re.IGNORECASE)
# "part number of ..." contains "number of" but asks for an identifier, not a
# count; such questions are covered by the identifier guard above.
_COUNT_NOUN_PATTERN = re.compile(
    r"\bhow many\s+(?:the\s+)?([\w\s-]+?)"
    r"(?=\s+(?:are|is|does|do|and|at|in|on|for|that|which|with|listed|"
    r"specified|supported|operate|the)\b|\s*\?|$)",
    re.IGNORECASE,
)
_COUNT_STOP_WORDS = {"the", "of", "in", "a", "an", "and", "for", "to"}


def _counted_noun(question: str) -> str:
    """Last content word of the noun phrase after 'how many'."""
    match = _COUNT_NOUN_PATTERN.search(question)
    if not match:
        return ""
    words = [
        word
        for word in tokenize(match.group(1))
        if word not in _COUNT_STOP_WORDS
    ]
    return words[-1] if words else ""


def _is_unverified_count(
    question: str, answer: str, context_text: str
) -> bool:
    """True when a numeric count in the answer is NOT stated in the evidence.

    A count is accepted when the passages either label it ("NO OF CYCLES: 10",
    "total 14") or state it inline next to the counted noun ("5 x eth ports").
    Word answers ("dual SIM") and identifier questions ("part number of")
    are left to the normal grounding rules.
    """
    if not _COUNT_QUESTION_PATTERN.search(question):
        return False
    if _IDENTIFIER_ATTRIBUTE_PATTERN.search(question):
        return False
    numbers = _NUMBER_PATTERN.findall(_spoken_numbers_to_digits(answer))
    if not numbers:
        return False  # e.g. "dual SIM" - a valid non-numeric answer
    count = numbers[0].replace(",", ".")
    context = _spoken_numbers_to_digits(context_text)

    labelled = (
        r"\b(?:no\.?|number|count|total|qty|quantity)\b"
        r"[^\n]{0,50}?\b"
        + re.escape(count)
        + r"\b"
    )
    if re.search(labelled, context, re.IGNORECASE):
        return False

    noun = _counted_noun(question)
    if noun:
        inline = (
            r"\b"
            + re.escape(count)
            + r"\b(?:\W+\w+){0,2}\W+"
            + re.escape(noun)
            + r"\b"
        )
        if re.search(inline, context, re.IGNORECASE):
            return False
    return True


def prioritize_sources(
    answer: str, sources: list[SearchResult]
) -> list[SearchResult]:
    """Evidence-bearing passages first, generic catalogue passages after.

    A passage that contains one of the answer's key values (numbers or long
    alphanumeric codes, boundary-matched) is project-specific evidence for
    THIS answer; a passage that merely shares topic words is generic. Both
    stay in the list (the answer is still grounded in all of them); only the
    display order changes. Stable: ties keep the fused ranking order.
    """
    number_keys = {value.casefold() for value in _numbers_in_text(answer)}
    code_keys = {
        token.casefold()
        for token in (raw.strip("._-/") for raw in _IDENTIFIER_PATTERN.findall(answer))
        if len(token) >= 6
        and sum(character.isdigit() for character in token) >= 2
        and any(character.isalpha() for character in token)
    }
    if (not number_keys and not code_keys) or not sources:
        return sources

    def has_evidence(source: SearchResult) -> bool:
        text = " ".join(_spoken_numbers_to_digits(source.text).split()).casefold()
        if any(
            re.search(rf"(?<!\d){re.escape(key)}(?!\d)", text)
            for key in number_keys
        ):
            return True
        compact = re.sub(r"[^a-z0-9]", "", text)
        return any(
            re.sub(r"[^a-z0-9]", "", key) in compact for key in code_keys
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


# Shared answer-style rules (Stage 5M). Small local models answer exact
# technical questions better when the expected shape is explicit: the same
# value/unit as the document, the matching table row, and every part of a
# multi-part question. "One or two sentences per part" keeps the model from
# copying the whole passage back (observed with llama3.2:3b).
_ANSWER_RULES = (
    "- Use the passages, not outside knowledge.\n"
    "- Report values, labels and units exactly as written in the passages; "
    "do not round or convert them.\n"
    "- When the answer is in a table, do not copy the table. Find the column "
    "the question asks about, find the row that matches the tag, model, label "
    "or step, and write only the requested value in a short sentence.\n"
    "- A row number, SI No., quantity, or value from another column is NOT "
    f"the serial number, model, date or other attribute asked for. If the "
    f"asked column is absent, reply exactly {NOT_FOUND_MARKER} and nothing "
    "else.\n"
    "- If the question asks how many items are listed, answer only when the "
    f"count is stated in the passages; do not guess a count. Otherwise reply "
    f"exactly {NOT_FOUND_MARKER} and nothing else.\n"
    "- If the question asks for several things, answer every part the "
    "passages support, in the same order.\n"
    "- Answer in plain text, one or two short sentences per part. Do not "
    "quote whole passages, never begin with a bracketed number, and do not "
    "mention page numbers or document names.\n"
)


def _build_prompt(question: str, results: list[SearchResult]) -> str:
    return (
        "You answer questions using ONLY the numbered reference passages below.\n"
        "Rules:\n"
        f"- If the passages do not contain the answer, reply exactly "
        f"{NOT_FOUND_MARKER} and nothing else.\n"
        f"{_ANSWER_RULES}"
        "\nReference passages:\n"
        f"{build_context_block(results)}\n"
        f"\nQuestion: {question}"
    )


def _build_retry_prompt(question: str, results: list[SearchResult]) -> str:
    """Second-chance prompt for a model that claimed insufficiency.

    Stage 5M makes the wording NEUTRAL: it says a previous reply did not
    answer and asks the model to re-read, but it does NOT assert that the
    answer is present. The assertion is tempting because it lifted recall,
    but on an unanswerable question it would coach the model toward
    inventing one. The refusal marker stays available, so the model is
    never forced to answer.
    """
    return (
        "You answer questions using ONLY the numbered reference passages below.\n"
        "A previous reply said the answer was not present. Re-read every "
        "passage carefully, line by line. The answer may be inside a table "
        "row, a model or tag code, or a sentence split across lines.\n"
        "Rules:\n"
        f"- Only if the passages really do not contain it, reply exactly "
        f"{NOT_FOUND_MARKER} and nothing else.\n"
        f"{_ANSWER_RULES}"
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

    # --- 2b. Exact-literal evidence salvage (Stage 5M) ----------------------
    # Add (never reorder) chunks that literally contain a rare tag/model code
    # or phrase from the question. The gate above already saw the original
    # retrieval list, so this cannot make a weak question answerable.
    evidence = select_evidence(question, results, embedded_chunks)

    prompt = _build_prompt(question, evidence)

    # --- 3. Ask the local model (Step 4B handles connection errors) ---------
    raw_answer = ask(prompt).strip()

    # --- 4. Re-read once after a refusal (Stage 5L/5M) ----------------------
    # Small local models sometimes refuse although the passages hold the
    # answer (measured: a correct remote-seal explanation whose question
    # shared almost no rare terms with the page). We re-read the SAME
    # evidence once with a focused, neutral prompt. The refusal marker is
    # still available, so the model is never forced; a second refusal is
    # honoured as an honest "not found".
    if looks_like_refusal(raw_answer):
        raw_answer = ask(_build_retry_prompt(question, evidence)).strip()
        if looks_like_refusal(raw_answer):
            return _not_found_bundle(question, reason="model_no_support")

    # --- 5. Grounding check (Stage 5L/5M): no invented values or codes -----
    context_text = " ".join(result.text for result in evidence)
    if _ungrounded_numbers(raw_answer, context_text) or _ungrounded_codes(
        raw_answer, context_text
    ):
        return _not_found_bundle(question, reason="ungrounded_answer")

    # A bare row number / quantity is not the serial/model/part number that
    # was asked for (Stage 5M; measured on a tag table that has no serial
    # column). Discard it rather than show a wrong fact.
    if _is_unverified_short_identifier(question, raw_answer):
        return _not_found_bundle(question, reason="unsupported_attribute")
    if _is_unverified_count(question, raw_answer, context_text):
        return _not_found_bundle(question, reason="unsupported_attribute")

    # --- 6. Attach OUR citations, evidence-bearing passages first -----------
    return AnswerBundle(
        question=question,
        answer=raw_answer,
        sources=prioritize_sources(raw_answer, evidence),
        not_found=False,
        reason="ok",
    )
