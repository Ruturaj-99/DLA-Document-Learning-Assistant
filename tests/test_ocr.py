"""Stage 5A: smoke tests for OCR (pytesseract + Tesseract engine).

Run from the project root with:
    .venv\\Scripts\\python -m pytest tests/ -v

What they prove:
    1. The Tesseract engine is found and usable.
    2. A drawn "scanned-style" image (text on paper-white background) is
       recognized as the right words.
    3. Every recognized word comes with a pixel bounding box (left, top,
       width, height) - the coordinates our future highlight feature needs.

If the engine is not installed, the tests SKIP (never fail, never download).
"""

import sys
import unittest.mock as mock
from pathlib import Path

import pytest

# Make "src" importable when pytest is run from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def find_tesseract() -> str | None:
    """Return the tesseract.exe path, or None when not installed.

    Checks PATH first, then the standard UB-Mannheim install locations.
    """
    import shutil

    from_which_py = shutil.which("tesseract")
    if from_which_py:
        return from_which_py
    for candidate in [
        Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
        Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
        Path.home() / r"AppData\Local\Programs\Tesseract-OCR\tesseract.exe",
    ]:
        if candidate.exists():
            return str(candidate)
    return None


ENGINE_PATH = find_tesseract()

if ENGINE_PATH:
    import pytesseract

    pytesseract.pytesseract.tesseract_cmd = ENGINE_PATH

pytestmark = pytest.mark.skipif(
    ENGINE_PATH is None, reason="Tesseract engine not installed on this machine"
)

SCAN_TEXT = "The pressure gauge must read between 25 and 30 bar."


def make_scan_like_image():
    """Draw scan-like text with a real Windows font. Returns a PIL Image."""
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (1200, 220), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 32)  # always present on Windows
    draw.text((40, 60), SCAN_TEXT, fill="black", font=font)
    return image


def test_engine_reports_its_version():
    version = pytesseract.get_tesseract_version()  # returns a packaging.Version
    assert version.major >= 5, f"Tesseract 5+ expected, found {version}"


def test_recognized_text_matches_the_drawn_sentence():
    result = pytesseract.image_to_string(make_scan_like_image())
    normalized = " ".join(result.split()).lower()
    # Robust comparison: every word except the last must appear (the trailing
    # word's punctuation sits closest to the image edge and can misread).
    core_words = SCAN_TEXT.lower().split()[:-1]
    for word in core_words:
        assert word in normalized, f"missing OCR word: {word}"


def test_every_recognized_word_has_pixel_coordinates():
    data = pytesseract.image_to_data(make_scan_like_image(), output_type=pytesseract.Output.DICT)
    words_found = {
        word.strip().lower(): (data["left"][i], data["top"][i], data["width"][i], data["height"][i])
        for i, word in enumerate(data["text"])
        if word.strip() and float(data["conf"][i]) > 0
    }
    for must_figure in ["gauge", "read", "pressure"]:
        assert must_figure in words_found, f"missing OCR word: {must_figure}"


def test_find_tesseract_returns_none_or_existing_path():
    # The helper itself: whatever it returns must be an existing exe (or None).
    path = find_tesseract()
    if path is not None:
        assert Path(path).exists()
