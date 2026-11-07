"""RAG question answering.

Stage 4C of the AI Document Learning Assistant.

In simple words, this module turns "here are search results" into
"here is a written answer, and here are the sources it came from".

The pipeline:
    1. Search the document chunks for passages similar to the question (Stage 3C).
       -> If the BEST passage barely matches the question, we STOP and admit
          we couldn't find it. The LLM is never even called. (score gate)
    2. Build a short "cheat sheet": numbered passages, each labeled with its
       document and page.
    3. Ask the local model to answer ONLY from that cheat sheet, and to reply
       with a special marker phrase if the passages don't contain the answer.
    4. Attach citations USING OUR CODE, not the model's words: every passage
       we handed over already knows its document, page, and exact characters
       (guaranteed since Stage 3A). The model therefore cannot invent page
       numbers we never gave it.

IMPORTANT: this module never shows the model anything but the numbered
passages, and never trusts the model's own page references.
"""

from dataclasses import dataclass

from src.embeddings import EmbeddedChunk
from src.ollama import ask_ollama
from src.retrieval import SearchResult, search_chunks

# What the model must reply when the passages don't contain the answer.
# A marker phrase (instead of free wording "I don't know") keeps the check
# simple and reliable - we look for exactly this string.
NOT_FOUND_MARKER = "ANSWER_NOT_IN_DOCUMENTS"

# Below this similarity score, even the best passage looks unrelated, and we
# skip the LLM entirely. Passage-question pairs that truly match typically
# score 0.5+; unrelated pairs stay near 0.0-0.2 (measured in Stage 3B tests).
DEFAULT_MIN_SCORE = 0.25


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
        reason: "ok" | "no_passages" | "low_relevance" | "model_no_support".
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
        min_score: similarity gate before we trust retrieval (see DEFAULT_MIN_SCORE).
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

    # --- 2. The score gate: is retrieval even confident? --------------------
    if results[0].score < min_score:
        return _not_found_bundle(question, reason="low_relevance")

    # --- 3. Build the cheat-sheet prompt ------------------------------------
    prompt = (
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

    # --- 4. Ask the local model (Step 4B handles connection errors) ---------
    raw_answer = ask(prompt).strip()

    # --- 5. Honesty check: did the model admit the passages were insufficient?
    if NOT_FOUND_MARKER in raw_answer.upper():
        return _not_found_bundle(question, reason="model_no_support")

    # --- 6. Attach OUR citations: the passages we actually provided ----------
    return AnswerBundle(
        question=question,
        answer=raw_answer,
        sources=results,
        not_found=False,
        reason="ok",
    )
