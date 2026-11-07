"""Estimate likely font matches for OCR text in scanned PDF pages.

Scores compare locally installed font renderings with OCR image crops. They
are similarity scores, not probabilities or claims about the source font.
"""

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageFilter

from src.ocr import OCR_DPI, OcrWord, ocr_page, render_pdf_page
import pymupdf

FONT_EXTENSIONS = {".ttf", ".otf", ".ttc"}
MAX_COLLECTION_FACES = 16


@dataclass(frozen=True)
class LocalFontFace:
    """A locally installed font face available for visual comparison."""

    path: str
    face_index: int
    family_name: str
    style_name: str


@dataclass(frozen=True)
class FontCandidateMatch:
    """A ranked font candidate and its visual similarity measurements."""

    family_name: str
    style_name: str
    font_path: str
    confidence_score: float
    estimated_size_points: float
    glyph_shape_similarity: float
    width_similarity: float
    ink_density_similarity: float


@dataclass(frozen=True)
class ScannedTextLineEstimate:
    """OCR line measurements and ranked local font candidates."""

    text: str
    bounds: tuple[float, float, float, float]
    average_interword_spacing_points: float | None
    matches: tuple[FontCandidateMatch, ...]


@dataclass(frozen=True)
class ScannedFontEstimate:
    """Estimated font candidates and layout measurements for one page."""

    page_number: int
    dpi: int
    alignment: str
    line_spacing_points: tuple[float, ...]
    candidate_count: int
    lines: tuple[ScannedTextLineEstimate, ...]
    score_description: str = (
        "Similarity score from local image comparisons; not a probability "
        "or guaranteed font identification."
    )


def _font_directories() -> tuple[Path, ...]:
    directories = [
        Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts",
        Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local"))
        / "Microsoft/Windows/Fonts",
        Path("/Library/Fonts"),
        Path("/System/Library/Fonts"),
        Path.home() / "Library/Fonts",
        Path("/usr/share/fonts"),
        Path("/usr/local/share/fonts"),
        Path.home() / ".fonts",
        Path.home() / ".local/share/fonts",
    ]
    return tuple(path for path in directories if path.is_dir())


@lru_cache(maxsize=1)
def discover_local_font_faces() -> tuple[LocalFontFace, ...]:
    """Discover unique font family/style faces from standard OS font folders."""
    faces: dict[tuple[str, str], LocalFontFace] = {}
    for directory in _font_directories():
        try:
            font_paths = sorted(
                (
                    path
                    for path in directory.rglob("*")
                    if path.is_file() and path.suffix.lower() in FONT_EXTENSIONS
                ),
                key=lambda path: str(path).casefold(),
            )
        except OSError:
            continue

        for path in font_paths:
            for face_index in range(MAX_COLLECTION_FACES):
                try:
                    font = ImageFont.truetype(str(path), size=16, index=face_index)
                    family_name, style_name = font.getname()
                except OSError:
                    break
                key = (family_name.casefold(), style_name.casefold())
                faces.setdefault(
                    key,
                    LocalFontFace(
                        path=str(path),
                        face_index=face_index,
                        family_name=family_name,
                        style_name=style_name,
                    ),
                )
                if path.suffix.lower() != ".ttc":
                    break

    return tuple(faces.values())


def _line_groups(words: Sequence[OcrWord]) -> list[list[OcrWord]]:
    grouped: dict[int, list[OcrWord]] = {}
    for word in words:
        grouped.setdefault(word.line_number, []).append(word)
    return [
        sorted(group, key=lambda word: word.left)
        for _, group in sorted(grouped.items())
    ]


def _line_bounds(
    words: Sequence[OcrWord],
) -> tuple[float, float, float, float]:
    return (
        float(min(word.left for word in words)),
        float(min(word.top for word in words)),
        float(max(word.left + word.width for word in words)),
        float(max(word.top + word.height for word in words)),
    )


def _line_alignment(lines: Sequence[Sequence[OcrWord]]) -> str:
    if len(lines) < 2:
        return "single line"

    bounds = [_line_bounds(line) for line in lines]
    left_edges = [box[0] for box in bounds]
    centers = [(box[0] + box[2]) / 2 for box in bounds]
    right_edges = [box[2] for box in bounds]
    tolerance = 3.0
    if max(left_edges) - min(left_edges) <= tolerance:
        return "left aligned"
    if max(centers) - min(centers) <= tolerance:
        return "center aligned"
    if max(right_edges) - min(right_edges) <= tolerance:
        return "right aligned"
    return "mixed alignment"


def _line_spacing(lines: Sequence[Sequence[OcrWord]]) -> tuple[float, ...]:
    baselines = [
        (
            min(word.top for word in line),
            max(word.top + word.height for word in line),
        )
        for line in lines
    ]
    return tuple(
        round(baselines[index + 1][0] - baselines[index][0], 2)
        for index in range(len(baselines) - 1)
    )


def _average_interword_spacing(words: Sequence[OcrWord]) -> float | None:
    gaps = [
        following.left - (word.left + word.width)
        for word, following in zip(words, words[1:])
    ]
    if not gaps:
        return None
    return round(sum(gaps) / len(gaps), 2)


def _binary_ink(image: Image.Image) -> Image.Image:
    grayscale = image.convert("L")
    return grayscale.point(lambda pixel: 255 if pixel < 190 else 0, mode="1")


def _ink_bounds(mask: Image.Image) -> tuple[int, int, int, int] | None:
    return mask.getbbox()


def _font_size_matching_height(
    face: LocalFontFace, text: str, target_height: int, maximum: int
) -> tuple[ImageFont.FreeTypeFont, int]:
    best_font: ImageFont.FreeTypeFont | None = None
    best_size = 1
    best_difference = max(1, maximum)
    low, high = 4, max(4, maximum)
    while low <= high:
        size = (low + high) // 2
        try:
            font = ImageFont.truetype(face.path, size=size, index=face.face_index)
        except OSError:
            break
        bounds = font.getbbox(text)
        rendered_height = max(1, bounds[3] - bounds[1])
        difference = abs(rendered_height - target_height)
        if difference < best_difference:
            best_font = font
            best_size = size
            best_difference = difference
        if rendered_height < target_height:
            low = size + 1
        else:
            high = size - 1
    if best_font is None:
        raise ValueError(f"Could not render local font candidate: {face.path}")
    return best_font, best_size


def _score_candidate(
    face: LocalFontFace,
    text: str,
    target_crop: Image.Image,
    dpi: int,
) -> FontCandidateMatch:
    padding = 4
    target_mask = _binary_ink(target_crop)
    target_bounds = _ink_bounds(target_mask)
    if target_bounds is None:
        raise ValueError("The OCR text region contains no measurable dark pixels.")

    target_width = target_bounds[2] - target_bounds[0]
    target_height = target_bounds[3] - target_bounds[1]
    font, pixel_size = _font_size_matching_height(
        face,
        text,
        target_height,
        max(target_crop.width, target_crop.height) * 2,
    )

    rendered = Image.new("L", target_crop.size, 255)
    draw = ImageDraw.Draw(rendered)
    draw.text((padding, padding), text, font=font, fill=0)
    candidate_mask = _binary_ink(rendered)
    candidate_bounds = _ink_bounds(candidate_mask)
    if candidate_bounds is None:
        raise ValueError(f"Font candidate rendered no visible glyphs: {face.path}")

    candidate_width = candidate_bounds[2] - candidate_bounds[0]
    candidate_height = candidate_bounds[3] - candidate_bounds[1]
    aligned_candidate = Image.new("L", target_crop.size, 0)
    candidate_glyphs = candidate_mask.crop(candidate_bounds)
    aligned_candidate.paste(candidate_glyphs, target_bounds[:2])
    candidate_mask = aligned_candidate
    candidate_ink = candidate_mask.convert("L")
    target_ink = target_mask.convert("L")
    target_tolerant = target_ink.filter(ImageFilter.MaxFilter(3))
    candidate_tolerant = candidate_ink.filter(ImageFilter.MaxFilter(3))
    target_pixels = target_ink.histogram()[255]
    candidate_pixels = candidate_ink.histogram()[255]
    intersection = ImageChops.multiply(target_tolerant, candidate_ink).histogram()[255]
    reverse_intersection = ImageChops.multiply(
        candidate_tolerant, target_ink
    ).histogram()[255]
    union = max(1, target_pixels + candidate_pixels)
    shape_similarity = min(
        1.0,
        (intersection + reverse_intersection) / union,
    )

    larger_width = max(target_width, candidate_width, 1)
    width_similarity = min(target_width, candidate_width) / larger_width
    target_density = target_pixels / max(1, target_crop.width * target_crop.height)
    candidate_density = candidate_pixels / max(
        1, target_crop.width * target_crop.height
    )
    density_similarity = 1 - min(
        1.0,
        abs(target_density - candidate_density)
        / max(target_density, candidate_density, 1e-6),
    )
    height_similarity = min(target_height, candidate_height) / max(
        target_height, candidate_height, 1
    )
    score = (
        shape_similarity * 0.65
        + width_similarity * 0.2
        + density_similarity * 0.1
        + height_similarity * 0.05
    )
    return FontCandidateMatch(
        family_name=face.family_name,
        style_name=face.style_name,
        font_path=face.path,
        confidence_score=round(score, 4),
        estimated_size_points=round(pixel_size * 72 / dpi, 2),
        glyph_shape_similarity=round(shape_similarity, 4),
        width_similarity=round(width_similarity, 4),
        ink_density_similarity=round(density_similarity, 4),
    )


def estimate_scanned_page_fonts(
    pdf_bytes: bytes,
    page_number: int,
    *,
    candidate_font_paths: Sequence[str | Path] | None = None,
    top_n: int = 5,
    dpi: int = OCR_DPI,
) -> ScannedFontEstimate:
    """Rank installed font candidates by similarity to OCR line images.

    OCR text/word boxes are grouped into lines. Each local font is rendered
    against each line crop; scores combine glyph-shape, width, ink-density,
    and height similarity. Font size, interword spacing, line spacing, and
    coarse line alignment are returned as observed layout estimates.

    Scores are not probabilities and do not guarantee identification of the
    original font. Candidate font paths can be supplied for a controlled
    comparison; otherwise standard system font directories are searched.
    """
    if top_n < 1:
        raise ValueError("top_n must be at least 1.")
    if dpi < 1:
        raise ValueError("dpi must be at least 1.")
    if not pdf_bytes:
        raise ValueError("No PDF data was given for font estimation.")

    try:
        with pymupdf.open(stream=pdf_bytes, filetype="pdf") as document:
            if not 1 <= page_number <= document.page_count:
                raise ValueError(
                    f"Page {page_number} does not exist - this document has "
                    f"{document.page_count} page(s)."
                )
            page_rect = document[page_number - 1].rect
    except ValueError:
        raise
    except Exception as error:
        raise ValueError(f"Could not open this PDF for font estimation: {error}") from error

    ocr_result = ocr_page(pdf_bytes, page_number, dpi=dpi)
    lines = _line_groups(ocr_result.words)
    if not lines:
        raise ValueError("OCR did not find any text lines to estimate.")

    if candidate_font_paths is None:
        faces = discover_local_font_faces()
    else:
        discovered: dict[tuple[str, str, int], LocalFontFace] = {}
        for font_path in candidate_font_paths:
            path = Path(font_path)
            if not path.is_file() or path.suffix.lower() not in FONT_EXTENSIONS:
                raise ValueError(f"Font candidate is not a supported font file: {path}")
            for face_index in range(MAX_COLLECTION_FACES):
                try:
                    font = ImageFont.truetype(
                        str(path), size=16, index=face_index
                    )
                    family_name, style_name = font.getname()
                except OSError:
                    break
                face = LocalFontFace(
                    path=str(path),
                    face_index=face_index,
                    family_name=family_name,
                    style_name=style_name,
                )
                discovered.setdefault(
                    (str(path.resolve()).casefold(), family_name.casefold(), face_index),
                    face,
                )
                if path.suffix.lower() != ".ttc":
                    break
        faces = tuple(discovered.values())

    if not faces:
        raise ValueError(
            "No supported local font files were found for comparison."
        )

    page_image = render_pdf_page(pdf_bytes, page_number, dpi=dpi)
    scale_x = page_image.width / page_rect.width
    scale_y = page_image.height / page_rect.height
    padding_points = 2.0
    line_estimates: list[ScannedTextLineEstimate] = []
    for line in lines:
        left, top, right, bottom = _line_bounds(line)
        crop = page_image.crop(
            (
                max(0, int((left - padding_points) * scale_x)),
                max(0, int((top - padding_points) * scale_y)),
                min(page_image.width, int((right + padding_points) * scale_x + 1)),
                min(page_image.height, int((bottom + padding_points) * scale_y + 1)),
            )
        )
        text = " ".join(word.text for word in line)
        scored: list[FontCandidateMatch] = []
        for face in faces:
            try:
                scored.append(_score_candidate(face, text, crop, dpi))
            except (OSError, ValueError):
                continue
        if not scored:
            raise ValueError(
                f"No font candidates could be rendered for OCR line: {text}"
            )
        line_estimates.append(
            ScannedTextLineEstimate(
                text=text,
                bounds=(left, top, right, bottom),
                average_interword_spacing_points=_average_interword_spacing(line),
                matches=tuple(
                    sorted(
                        scored,
                        key=lambda candidate: candidate.confidence_score,
                        reverse=True,
                    )[:top_n]
                ),
            )
        )

    return ScannedFontEstimate(
        page_number=page_number,
        dpi=dpi,
        alignment=_line_alignment(lines),
        line_spacing_points=_line_spacing(lines),
        candidate_count=len(faces),
        lines=tuple(line_estimates),
    )
