"""Embeddings - turning text into numbers a computer can compare.

Stage 3B of the AI Document Learning Assistant.

In simple words: MiniLM reads a piece of text and outputs a list of 384
numbers (a "vector"). Two texts with the same meaning get lists of numbers
that point in almost the same direction. So to find passages that answer
a question, we compare directions instead of comparing words.

This module:
    - loads the model once and reuses it (loading takes a few seconds,
      so we must not do it per call),
    - turns Stage 3A chunks into vectors while keeping every piece of
      source info (document_id, page, text, start_char, end_char),
    - turns a question into a vector with the SAME model (numbers from
      different models can never be compared).

Everything runs locally on your CPU. Nothing is uploaded anywhere.
"""

from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer

from src.chunking import Chunk

MODEL_NAME = "all-MiniLM-L6-v2"
EXPECTED_DIMENSIONS = 384  # MiniLM always outputs 384 numbers per text

# Loaded lazily on first use, then shared for the rest of the session.
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Load the embedding model once and reuse it afterwards.

    device="cpu" keeps this working on any machine, including yours
    (a 6-core Intel i5 is more than enough for this small model).
    """
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME, device="cpu")
    return _model


@dataclass
class EmbeddedChunk:
    """One chunk plus its vector - source info never separated from text.

    Attributes:
        document_id / page_number / text / start_char / end_char:
            copied 1:1 from the Stage 3A Chunk.
        vector: the embedding as a plain list of floats (384 numbers).
    """

    document_id: str
    page_number: int
    text: str
    start_char: int
    end_char: int
    vector: list[float]


def embed_chunks(chunks: list[Chunk]) -> list[EmbeddedChunk]:
    """Generate one embedding per chunk, keeping all source metadata.

    Args:
        chunks: the output of chunking.chunk_pages().

    Returns:
        EmbeddedChunks in the same order as the input chunks.
        Empty input returns an empty list (no model load, no crash).
    """
    if not chunks:
        return []
    model = get_model()
    vectors = model.encode(
        [chunk.text for chunk in chunks],
        normalize_embeddings=True,  # vectors get length 1 -> cosine = simple dot product
        show_progress_bar=False,
    )
    return [
        EmbeddedChunk(
            document_id=chunk.document_id,
            page_number=chunk.page_number,
            text=chunk.text,
            start_char=chunk.start_char,
            end_char=chunk.end_char,
            vector=vector.tolist(),  # numpy row -> ordinary Python list of floats
        )
        for chunk, vector in zip(chunks, vectors)
    ]


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed arbitrary texts with the same model (used by tests and demos)."""
    if not texts:
        return []
    vectors = get_model().encode(
        texts, normalize_embeddings=True, show_progress_bar=False
    )
    return [v.tolist() for v in vectors]


def embed_question(question: str) -> list[float]:
    """Embed a user question with the same model used for document chunks."""
    model = get_model()
    vector = model.encode(question, normalize_embeddings=True)
    return vector.tolist()


def cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    """Similarity score in the range -1..1 (practical range: 0..1).

    Rule of thumb for MiniLM with normalized vectors:
        above ~0.5  -> strongly related meaning
        around 0.1  -> probably unrelated
    """
    a = np.asarray(vector_a)
    b = np.asarray(vector_b)
    return float(np.dot(a, b))
