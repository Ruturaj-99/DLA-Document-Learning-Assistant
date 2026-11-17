"""BM25 keyword scoring (Stage 5J).

In simple words: while the embeddings (Stage 3B) find chunks by MEANING,
BM25 finds chunks by EXACT WORDS. Technical language is full of things a
small embedding model easily blurs - part numbers ("TSW114000000"),
numerical values ("9216 bytes", "9 - 30 VDC") and rare technical terms.
BM25 rewards a chunk that literally contains those tokens, weighted by:

    - how rare the token is across the document (IDF): a token that
      appears in many chunks tells us little;
    - how often it appears in THIS chunk (TF, saturating at k1);
    - how long the chunk is (length normalisation b): a hit inside a
      short chunk counts more than one inside a sprawling page.

This is a small, dependency-free implementation of the classic Okapi
BM25 formula (k1 = 1.5, b = 0.75) with the non-negative IDF variant
used by Lucene, so scores never go below zero.

Like the rest of the pipeline it is document-agnostic: no term list is
tuned to any one document. Instead, question tokens that appear in more
than half of the candidate chunks are ignored while scoring - repeated
page headers and boilerplate words cannot separate chunks, so they must
not be allowed to rank every chunk above the real specification row.
"""

import math
import re

# Standard Okapi BM25 parameters.
K1 = 1.5  # term-frequency saturation point
B = 0.75  # length-normalisation strength

# Tokens are alphanumeric runs: words, numbers and codes alike.
# "4-20 mA" -> 4, 20, ma; "TSW114000000" stays one token.
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens of a question or chunk."""
    return _TOKEN_PATTERN.findall(text.lower())


class BM25Index:
    """A tiny BM25 index over a list of documents (chunk texts).

    Built once per search call over the candidate chunks; fast enough
    for the corpus sizes this app handles (a few thousand chunks).

    Question tokens that appear in more than half of the documents are
    ignored while scoring (MAX_QUERY_DF_RATIO): such tokens are shared
    by repeated page headers and boilerplate, so they rank every chunk
    instead of separating the relevant ones. This is purely statistical
    - no term list is tuned to any document.
    """

    MAX_QUERY_DF_RATIO = 0.5

    def __init__(self, documents: list[str], k1: float = K1, b: float = B):
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(document) for document in documents]
        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens]
        self.total_docs = len(documents)
        self.average_length = (
            sum(self.doc_lengths) / self.total_docs if self.total_docs else 0.0
        )
        # term -> in how many documents it appears (document frequency).
        self.doc_frequency: dict[str, int] = {}
        for tokens in self.doc_tokens:
            for token in set(tokens):
                self.doc_frequency[token] = self.doc_frequency.get(token, 0) + 1

    def question_tokens(self, question: str) -> list[str]:
        """Discriminative tokens of the question.

        A token is dropped when it appears in more than half of the
        documents: it cannot separate relevant chunks from filler, and
        its accumulated low-IDF matches let boilerplate-heavy chunks
        outrank the real spec row.
        """
        tokens = tokenize(question)
        if self.total_docs == 0:
            return []
        limit = self.MAX_QUERY_DF_RATIO * self.total_docs
        return [
            token
            for token in tokens
            if self.doc_frequency.get(token, 0) <= limit
        ]

    def idf(self, token: str) -> float:
        """Non-negative inverse document frequency (Lucene variant)."""
        frequency = self.doc_frequency.get(token, 0)
        return math.log(1.0 + (self.total_docs - frequency + 0.5) / (frequency + 0.5))

    def score(self, question: str) -> list[float]:
        """BM25 score of every document against the question."""
        if self.total_docs == 0:
            return []
        scores = [0.0] * self.total_docs
        question_tokens = self.question_tokens(question)
        if not question_tokens or self.average_length == 0:
            return scores
        for doc_index, (tokens, length) in enumerate(
            zip(self.doc_tokens, self.doc_lengths)
        ):
            if not tokens:
                continue
            term_counts: dict[str, int] = {}
            for token in tokens:
                term_counts[token] = term_counts.get(token, 0) + 1
            length_norm = self.k1 * (
                1.0 - self.b + self.b * length / self.average_length
            )
            doc_score = 0.0
            for token in question_tokens:
                count = term_counts.get(token)
                if not count:
                    continue
                weight = (count * (self.k1 + 1.0)) / (count + length_norm)
                doc_score += self.idf(token) * weight
            scores[doc_index] = doc_score
        return scores

    def ranked(self, question: str) -> list[int]:
        """Document indices with a positive score, best first.

        Documents that share NO token with the question are omitted:
        in Reciprocal Rank Fusion an absent ranking simply contributes
        nothing instead of pretending to know an order.
        Ties break by original document order, so results are stable.
        """
        scores = self.score(question)
        order = sorted(
            (index for index, score in enumerate(scores) if score > 0.0),
            key=lambda index: (-scores[index], index),
        )
        return order
