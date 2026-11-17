"""Create and manage movable, embedded comments in PDF working copies."""

import math
import re
import uuid
from dataclasses import dataclass

import pymupdf


_COMMENT_SUBJECT = re.compile(
    r"ai-dla-comment:v[123]:([0-9a-f]{32}):([01]):([0-9a-fA-F]{6})"
    r"(?::(\d+(?:\.\d+)?))?(?::(-?\d+):(-?\d+))?"
)


# The one font size every comment used before per-comment sizes existed.
DEFAULT_COMMENT_FONT_SIZE = 11.0

# Stored normalized positions are rounded to millionths (plenty for pixels).
_POSITION_SCALE = 1_000_000


@dataclass(frozen=True)
class PdfComment:
    """A comment annotation positioned in the rendered page coordinate space."""

    comment_id: str
    page_number: int
    text: str
    x: float
    y: float
    highlighted: bool
    color: str
    font_size: float = DEFAULT_COMMENT_FONT_SIZE


def _validate_position(x: float, y: float) -> tuple[float, float]:
    try:
        normalized_x = float(x)
        normalized_y = float(y)
    except (TypeError, ValueError) as error:
        raise ValueError("Comment position must be numeric.") from error
    if not math.isfinite(normalized_x) or not math.isfinite(normalized_y):
        raise ValueError("Comment position must be finite.")
    if not 0 <= normalized_x <= 1 or not 0 <= normalized_y <= 1:
        raise ValueError("Comment position must be within the page.")
    return normalized_x, normalized_y


def _parse_color(color: str) -> tuple[str, tuple[float, float, float]]:
    if not isinstance(color, str) or re.fullmatch(r"#[0-9a-fA-F]{6}", color) is None:
        raise ValueError("Comment color must be a six-digit hex color.")
    normalized = color.upper()
    rgb = tuple(int(normalized[index : index + 2], 16) / 255 for index in (1, 3, 5))
    return normalized, rgb


def _comment_metadata(annotation: pymupdf.Annot):
    """Decode identity + style + position from the annotation subject.

    Returns ``(id, highlighted, color, font_size, x, y)``; x/y are the
    stored normalized position, or ``None`` for comments written before
    positions were embedded (v1/v2 subjects).
    """
    match = _COMMENT_SUBJECT.fullmatch(annotation.info.get("subject", ""))
    if match is None:
        return None
    font_size = (
        float(match.group(4))
        if match.group(4)
        else DEFAULT_COMMENT_FONT_SIZE
    )
    stored_x = (
        int(match.group(5)) / _POSITION_SCALE if match.group(5) else None
    )
    stored_y = (
        int(match.group(6)) / _POSITION_SCALE if match.group(6) else None
    )
    return (
        match.group(1),
        match.group(2) == "1",
        f"#{match.group(3).upper()}",
        font_size,
        stored_x,
        stored_y,
    )


def _raw_display_rect(page: pymupdf.Page, annotation: pymupdf.Annot) -> pymupdf.Rect:
    """Display-space rect computed from the raw /Rect dictionary key.

    pymupdf 2.0 crashes with a native access violation when reading the
    bound rect (``annotation.rect``) of a reopened FreeText annot, so we
    never touch it: the raw /Rect lives in PDF space (origin bottom-left,
    relative to the crop box) and is converted to the same value
    ``annotation.rect * page.rotation_matrix`` would yield.
    """
    entry = page.parent.xref_get_key(annotation.xref, "Rect")
    if entry[0] == "array":
        values = [float(token) for token in entry[1].strip("[]").split()]
        if len(values) == 4:
            box = page.cropbox
            unrotated = pymupdf.Rect(
                values[0] - box.x0,
                box.y1 - values[3],
                values[2] - box.x0,
                box.y1 - values[1],
            )
            return unrotated * page.rotation_matrix
    return annotation.rect * page.rotation_matrix


def _comment_position(
    page: pymupdf.Page, annotation: pymupdf.Annot, metadata: tuple
) -> tuple[float, float]:
    """Normalized comment position: stored value, else from the raw rect."""
    stored_x, stored_y = metadata[4], metadata[5]
    if stored_x is not None and stored_y is not None:
        return stored_x, stored_y
    point = _raw_display_rect(page, annotation).tl
    return (
        min(max(point.x / page.rect.width, 0), 1),
        min(max(point.y / page.rect.height, 0), 1),
    )


def _page_rect(
    page: pymupdf.Page, x: float, y: float, text: str, fontsize: float = 11.0
) -> pymupdf.Rect:
    left = min(x, max(0.0, page.rect.width - 1.0))
    top = min(y, max(0.0, page.rect.height - 1.0))
    max_width = min(220.0, page.rect.width * 0.42, page.rect.width - left)
    font = pymupdf.Font("helv")
    max_text_width = max(1.0, max_width - 8.0)
    wrapped_lines = []
    for paragraph in text.splitlines() or [""]:
        line = ""
        for word in paragraph.split():
            candidate = f"{line} {word}".strip()
            if font.text_length(candidate, fontsize=fontsize) <= max_text_width:
                line = candidate
                continue
            if line:
                wrapped_lines.append(line)
                line = ""
            part = ""
            for character in word:
                candidate = part + character
                if part and font.text_length(candidate, fontsize=fontsize) > max_text_width:
                    wrapped_lines.append(part)
                    part = character
                else:
                    part = candidate
            line = part
        wrapped_lines.append(line)
    width = max(
        18.0,
        max(font.text_length(line, fontsize=fontsize) for line in wrapped_lines) + 8.0,
    )
    width = min(width, max_width)
    height = min(
        page.rect.height - top,
        max(18.0, len(wrapped_lines) * fontsize * 1.27 + 4.0),
    )
    return pymupdf.Rect(left, top, left + width, top + height)


def list_pdf_comments(pdf_bytes: bytes, page_number: int | None = None) -> list[PdfComment]:
    """Return comments embedded by this app, optionally limited to one page."""
    if not pdf_bytes:
        raise ValueError("No PDF data was given.")
    document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    try:
        if page_number is not None and not 1 <= page_number <= document.page_count:
            raise ValueError(
                f"Page {page_number} does not exist - this document has "
                f"{document.page_count} page(s)."
            )

        comments = []
        page_indexes = (
            [page_number - 1]
            if page_number is not None
            else range(document.page_count)
        )
        for page_index in page_indexes:
            page = document[page_index]
            for annotation in page.annots() or ():
                metadata = _comment_metadata(annotation)
                if metadata is None:
                    continue
                comment_id, highlighted, color, font_size = metadata[:4]
                x, y = _comment_position(page, annotation, metadata)
                comments.append(
                    PdfComment(
                        comment_id=comment_id,
                        page_number=page_index + 1,
                        text=annotation.info.get("content", ""),
                        x=x,
                        y=y,
                        highlighted=highlighted,
                        color=color,
                        font_size=font_size,
                    )
                )
        return comments
    finally:
        document.close()


def _comment_subject(
    comment_id: str,
    highlighted: bool,
    color: str,
    font_size: float,
    x: float,
    y: float,
) -> str:
    """Encode identity + style + position in the annotation subject (v3)."""
    size_text = f"{font_size:g}"
    return (
        f"ai-dla-comment:v3:{comment_id}:{int(highlighted)}:"
        f"{color[1:]}:{size_text}:"
        f"{round(x * _POSITION_SCALE)}:{round(y * _POSITION_SCALE)}"
    )


def _insert_comment(
    document: pymupdf.Document,
    page: pymupdf.Page,
    anchor_x: float,
    anchor_y: float,
    clean_text: str,
    highlighted: bool,
    color: str,
    comment_id: str,
    font_size: float = DEFAULT_COMMENT_FONT_SIZE,
) -> bytes:
    """Attach one FreeText annot at a normalized anchor, then serialize.

    ``anchor_x``/``anchor_y`` are normalized display coordinates and are
    stored in the subject so reading the position back never has to touch
    the annotation rect (which crashes in pymupdf 2.0).

    Default look: PLAIN text - no fill, no border, no highlight. The fill
    (highlight) only appears when the user explicitly asks for it.
    """
    target_x = anchor_x * page.rect.width
    target_y = anchor_y * page.rect.height
    display_rect = _page_rect(
        page, target_x, target_y, clean_text, fontsize=font_size
    )
    page_rect = display_rect * page.derotation_matrix
    annotation = page.add_freetext_annot(
        page_rect,
        clean_text,
        fontsize=font_size,
        fontname="Helv",
        text_color=(0.12, 0.12, 0.12),
        fill_color=_parse_color(color)[1] if highlighted else None,
        border_color=None,
        border_width=0,
        opacity=0.72 if highlighted else 1,
    )
    annotation.set_info(
        title="AI DLA comment",
        subject=_comment_subject(
            comment_id, highlighted, color, font_size, anchor_x, anchor_y
        ),
    )
    annotation.update()
    return document.tobytes(garbage=4, deflate=True)


def add_pdf_comment(
    pdf_bytes: bytes,
    page_number: int,
    x: float,
    y: float,
    text: str,
    *,
    highlighted: bool = False,
    color: str = "#D9D9D9",
    font_size: float = DEFAULT_COMMENT_FONT_SIZE,
) -> bytes:
    """Add visible, movable comment text.

    Default: plain text with NO highlight and NO border. A highlight (a soft
    colour fill behind the text) is only drawn when `highlighted=True` - the
    user must explicitly ask for it.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Enter comment text before placing it on the page.")
    try:
        normalized_size = float(font_size)
    except (TypeError, ValueError) as error:
        raise ValueError("Comment font size must be numeric.") from error
    if not math.isfinite(normalized_size) or not 4 <= normalized_size <= 72:
        raise ValueError(
            "Comment font size must be between 4 and 72 points."
        )
    normalized_x, normalized_y = _validate_position(x, y)
    normalized_color, rgb = _parse_color(color)
    document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    try:
        if not 1 <= page_number <= document.page_count:
            raise ValueError(
                f"Page {page_number} does not exist - this document has "
                f"{document.page_count} page(s)."
            )
        page = document[page_number - 1]
        clean_text = text.strip()
        # Idempotence: a drop gesture can fire twice (race between the element
        # re-render and the new event). Re-adding identical text at (nearly)
        # the same spot must merge into the existing layer, never stack a
        # second one on top.
        for annotation in page.annots() or ():
            metadata = _comment_metadata(annotation)
            if metadata is None:
                continue
            if annotation.info.get("content", "").strip() != clean_text:
                continue
            x, y = _comment_position(page, annotation, metadata)
            if (
                math.hypot(
                    (x - normalized_x) * page.rect.width,
                    (y - normalized_y) * page.rect.height,
                )
                <= 12.0
            ):
                return document.tobytes(garbage=4, deflate=True)
        return _insert_comment(
            document,
            page,
            normalized_x,
            normalized_y,
            clean_text,
            highlighted,
            normalized_color,
            uuid.uuid4().hex,
            font_size=normalized_size,
        )
    finally:
        document.close()


def move_pdf_comment(
    pdf_bytes: bytes, comment_id: str, x: float, y: float
) -> bytes:
    """Move an embedded comment to a normalized rendered-page position."""
    normalized_x, normalized_y = _validate_position(x, y)
    document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    try:
        for page in document:
            for annotation in page.annots() or ():
                metadata = _comment_metadata(annotation)
                if metadata is None or metadata[0] != comment_id:
                    continue
                _, highlighted, color, font_size = metadata[:4]
                page.delete_annot(annotation)
                return _insert_comment(
                    document,
                    page,
                    normalized_x,
                    normalized_y,
                    annotation.info.get("content", "").strip(),
                    highlighted,
                    color,
                    comment_id,
                    font_size=font_size,
                )
        raise ValueError("That comment no longer exists in this PDF.")
    finally:
        document.close()


def update_pdf_comment(
    pdf_bytes: bytes,
    comment_id: str,
    text: str,
    x: float | None = None,
    y: float | None = None,
    *,
    highlighted: bool | None = None,
    color: str | None = None,
    font_size: float | None = None,
) -> bytes:
    """Change an embedded comment's text/style, optionally moving it as well.

    Editing re-creates the annot so the wrapped text box fits the new text,
    while keeping the original comment id - unless a position is given, in
    which case the comment also relocates there. Style arguments that are
    left as None keep the stored style; passing a value restyles the comment
    (e.g. turning the highlight or a bigger font on at the user's request).
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Enter comment text before saving the comment.")
    document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    try:
        for page in document:
            for annotation in page.annots() or ():
                metadata = _comment_metadata(annotation)
                if metadata is None or metadata[0] != comment_id:
                    continue
                _, stored_highlighted, stored_color, stored_size = metadata[:4]
                effective_highlighted = (
                    stored_highlighted if highlighted is None else bool(highlighted)
                )
                effective_color = stored_color if color is None else color
                normalized_color, rgb = _parse_color(effective_color)
                effective_size = stored_size if font_size is None else float(font_size)
                if not math.isfinite(effective_size) or not 4 <= effective_size <= 72:
                    raise ValueError(
                        "Comment font size must be between 4 and 72 points."
                    )
                anchor_x, anchor_y = _comment_position(page, annotation, metadata)
                if x is not None or y is not None:
                    # Keep whichever axis was not given.
                    anchor_x, anchor_y = _validate_position(
                        x if x is not None else anchor_x,
                        y if y is not None else anchor_y,
                    )
                page.delete_annot(annotation)
                return _insert_comment(
                    document,
                    page,
                    anchor_x,
                    anchor_y,
                    text.strip(),
                    effective_highlighted,
                    normalized_color,
                    comment_id,
                    font_size=effective_size,
                )
        raise ValueError("That comment no longer exists in this PDF.")
    finally:
        document.close()


def delete_pdf_comment(pdf_bytes: bytes, comment_id: str) -> bytes:
    """Remove an embedded comment from the working PDF."""
    document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    try:
        for page in document:
            for annotation in page.annots() or ():
                metadata = _comment_metadata(annotation)
                if metadata is None or metadata[0] != comment_id:
                    continue
                page.delete_annot(annotation)
                return document.tobytes(garbage=4, deflate=True)
        raise ValueError("That comment no longer exists in this PDF.")
    finally:
        document.close()
