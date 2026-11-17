"""Metrics for the RAG evaluation.

All metrics are computed from the same objects the app really produces:
retrieval.SearchResult lists and qa.AnswerBundle instances.

Retrieval metrics (answerable questions):
    hit@k           - a chunk matching the gold page AND containing the
                      gold snippet appears within the first k results.
    mrr             - mean reciprocal rank of the first gold chunk.
    context_coverage- every gold snippet is present in the retrieved
                      context (what the LLM would actually have seen).

Behaviour metrics (via qa.answer_question with a scripted oracle):
    gate_accepted   - retrieval passed the score gate (LLM was called).
    correct_rejection- for unanswerable questions: the pipeline returned
                      "not found" instead of inventing an answer.
    citation_valid  - every returned source round-trips its provenance:
                      page_text[start_char:end_char] == source.text.
    answer_supported- every gold snippet is contained in at least one
                      returned source (the citations carry the evidence).
"""

from __future__ import annotations

from dataclasses import dataclass, field


def normalized(text: str) -> str:
    """Case/whitespace-insensitive form used for snippet matching.

    pymupdf wraps lines when rendering the evaluation PDF, so the same
    sentence can extract with newlines inside it. Normalizing collapses
    all whitespace runs and lowercases, making matching robust.
    """
    return " ".join(text.split()).casefold()


def is_gold_chunk(
    chunk_page_number: int, chunk_text: str, gold_pages, gold_snippets
) -> bool:
    """Does this chunk come from a gold page and contain that page's snippet?"""
    for page, snippet in zip(gold_pages, gold_snippets):
        if chunk_page_number == page and normalized(snippet) in normalized(
            chunk_text
        ):
            return True
    return False


def retrieval_metrics(
    results, gold_pages, gold_snippets, ks=(1, 3, 5)
) -> dict[str, float | int | bool]:
    """Score one ranked retrieval list against the gold annotations."""
    ranks: list[int] = []
    for index, result in enumerate(results, start=1):
        if is_gold_chunk(
            result.page_number, result.text, gold_pages, gold_snippets
        ):
            ranks.append(index)
    first_rank = ranks[0] if ranks else 0

    metrics: dict[str, float | int | bool] = {
        "first_gold_rank": first_rank,
        "mrr": 1.0 / first_rank if first_rank else 0.0,
    }
    for k in ks:
        metrics[f"hit@{k}"] = bool(first_rank and first_rank <= k)
    metrics["context_coverage"] = all(
        normalized(snippet) in normalized(
            " ".join(result.text for result in results)
        )
        for snippet in gold_snippets
    )
    return metrics


def citation_metrics(
    bundle, page_texts: dict[int, str], gold_pages=None, gold_snippets=None
) -> dict[str, float | int | bool]:
    """Score one AnswerBundle: gate behaviour, rejection and citations."""
    metrics: dict[str, float | int | bool] = {
        "not_found": bundle.not_found,
        "reason": bundle.reason,
        "gate_accepted": bundle.reason != "low_relevance"
        and bundle.reason != "no_passages",
        "n_sources": len(bundle.sources),
        # Provenance round-trip: the cited slice must reproduce the chunk
        # from the real page text (the invariant guaranteed by Stage 3A).
        "citation_valid": (
            not bundle.not_found
            and bool(bundle.sources)
            and all(
                page_texts[source.page_number][
                    source.start_char : source.end_char
                ]
                == source.text
                for source in bundle.sources
                if source.page_number in page_texts
            )
        ),
    }
    if gold_snippets is not None:
        cited_text = normalized(
            " ".join(source.text for source in bundle.sources)
        )
        metrics["answer_supported"] = all(
            normalized(snippet) in cited_text for snippet in gold_snippets
        )
    return metrics


@dataclass
class QuestionResult:
    """Everything recorded for one question."""

    question_id: str
    question: str
    category: str
    retrieval: dict = field(default_factory=dict)
    bundle_metrics: dict = field(default_factory=dict)


@dataclass
class Aggregates:
    """Aggregated metric values (overall or per category)."""

    count: int = 0
    hit_counts: dict[int, int] = field(default_factory=dict)
    reciprocal_rank_sum: float = 0.0
    context_coverage_count: int = 0
    correct_rejection_count: int = 0
    citation_valid_count: int = 0
    answer_supported_count: int = 0
    not_found_count: int = 0
    ks: tuple[int, ...] = (1, 3, 5)

    def add(self, result: QuestionResult, unanswerable: bool) -> None:
        self.count += 1
        if not unanswerable:
            for k in self.ks:
                if result.retrieval.get(f"hit@{k}"):
                    self.hit_counts[k] = self.hit_counts.get(k, 0) + 1
            self.reciprocal_rank_sum += float(result.retrieval.get("mrr", 0.0))
            if result.retrieval.get("context_coverage"):
                self.context_coverage_count += 1
            if result.bundle_metrics.get("answer_supported"):
                self.answer_supported_count += 1
        else:
            if result.bundle_metrics.get("not_found"):
                self.correct_rejection_count += 1
        if result.bundle_metrics.get("citation_valid"):
            self.citation_valid_count += 1
        if result.bundle_metrics.get("not_found"):
            self.not_found_count += 1

    def as_dict(self) -> dict[str, float | int]:
        """Rates computed over THIS aggregate's own subset (the runner
        keeps answerable and unanswerable questions in separate
        Aggregates, so each rate is over the right denominator)."""
        values: dict[str, float | int] = {"questions": self.count}
        for k in self.ks:
            values[f"hit@{k}"] = round(
                self.hit_counts.get(k, 0) / self.count, 3
            )
        values["mrr"] = round(self.reciprocal_rank_sum / self.count, 3)
        values["context_coverage"] = round(
            self.context_coverage_count / self.count, 3
        )
        values["answer_supported"] = round(
            self.answer_supported_count / self.count, 3
        )
        values["correct_rejection"] = round(
            self.correct_rejection_count / self.count, 3
        )
        values["citation_valid"] = round(
            self.citation_valid_count / self.count, 3
        )
        return values
