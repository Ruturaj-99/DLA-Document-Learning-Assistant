"""Create and manage movable, embedded comments in PDF working copies."""

import math
import re
import uuid
from dataclasses import dataclass

import pymupdf


_COMMENT_SUBJECT = re.compile(
    r"ai-dla-comment:v1:([0-9a-f]{32}):([01]):([0-9a-fA-F]{6})"
)


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
    match = _COMMENT_SUBJECT.fullmatch(annotation.info.get("subject", ""))
    if match is None:
        return None
    return match.group(1), match.group(2) == "1", f"#{match.group(3).upper()}"


def _display_rect(page: pymupdf.Page, annotation: pymupdf.Annot) -> pymupdf.Rect:
    return annotation.rect * page.rotation_matrix


def _page_rect(
    page: pymupdf.Page, x: float, y: float, text: str
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
            if font.text_length(candidate, fontsize=11) <= max_text_width:
                line = candidate
                continue
            if line:
                wrapped_lines.append(line)
                line = ""
            part = ""
            for character in word:
                candidate = part + character
                if part and font.text_length(candidate, fontsize=11) > max_text_width:
                    wrapped_lines.append(part)
                    part = character
                else:
                    part = candidate
            line = part
        wrapped_lines.append(line)
    width = max(
        18.0,
        max(font.text_length(line, fontsize=11) for line in wrapped_lines) + 8.0,
    )
    width = min(width, max_width)
    height = min(
        page.rect.height - top,
        max(18.0, len(wrapped_lines) * 14.0 + 4.0),
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
                comment_id, highlighted, color = metadata
                point = _display_rect(page, annotation).tl
                comments.append(
                    PdfComment(
                        comment_id=comment_id,
                        page_number=page_index + 1,
                        text=annotation.info.get("content", ""),
                        x=min(max(point.x / page.rect.width, 0), 1),
                        y=min(max(point.y / page.rect.height, 0), 1),
                        highlighted=highlighted,
                        color=color,
                    )
                )
        return comments
    finally:
        document.close()


def _comment_subject(
    comment_id: str, highlighted: bool, color: str
) -> str:
    """Encode identity + style in the annotation subject, as before."""
    return f"ai-dla-comment:v1:{comment_id}:{int(highlighted)}:{color[1:]}"


def _insert_comment(
    document: pymupdf.Document,
    page: pymupdf.Page,
    anchor_x: float,
    anchor_y: float,
    clean_text: str,
    highlighted: bool,
    color: str,
    comment_id: str,
) -> bytes:
    """Attach one FreeText annot at a display-space anchor, then serialize."""
    display_rect = _page_rect(page, anchor_x, anchor_y, clean_text)
    page_rect = display_rect * page.derotation_matrix
    annotation = page.add_freetext_annot(
        page_rect,
        clean_text,
        fontsize=11,
        fontname="Helv",
        text_color=(0.12, 0.12, 0.12),
        fill_color=_parse_color(color)[1] if highlighted else None,
        border_color=None,
        border_width=0,
        opacity=0.72 if highlighted else 1,
    )
    annotation.set_info(
        title="AI DLA comment",
        subject=_comment_subject(comment_id, highlighted, color),
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
) -> bytes:
    """Add visible, movable comment text with an optional highlight."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Enter comment text before placing it on the page.")
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
        target_x = normalized_x * page.rect.width
        target_y = normalized_y * page.rect.height
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
            point = _display_rect(page, annotation).tl
            if math.hypot(point.x - target_x, point.y - target_y) <= 12.0:
                return document.tobytes(garbage=4, deflate=True)
        return _insert_comment(
            document,
            page,
            target_x,
            target_y,
            clean_text,
            highlighted,
            normalized_color,
            uuid.uuid4().hex,
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
                display_point = pymupdf.Point(
                    normalized_x * page.rect.width,
                    normalized_y * page.rect.height,
                )
                display_rect = _page_rect(
                    page,
                    display_point.x,
                    display_point.y,
                    annotation.info.get("content", ""),
                )
                annotation.set_rect(display_rect * page.derotation_matrix)
                annotation.update()
                return document.tobytes(garbage=4, deflate=True)
        raise ValueError("That comment no longer exists in this PDF.")
    finally:
        document.close()


def update_pdf_comment(
    pdf_bytes: bytes,
    comment_id: str,
    text: str,
    x: float | None = None,
    y: float | None = None,
) -> bytes:
    """Change an embedded comment's text, optionally moving it as well.

    Editing re-creates the annot so the wrapped text box fits the new text,
    while keeping the original comment id, highlight, and color - unless a
    position is given, in which case the comment also relocates there.
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
                _, highlighted, color = metadata
                if x is None or y is None:
                    anchor = _display_rect(page, annotation).tl
                    if x is not None or y is not None:
                        # Mixed arguments: keep whichever axis is unchanged.
                        kept = _validate_position(
                            x if x is not None else
                            anchor.x / page.rect.width,
                            y if y is not None else
                            anchor.y / page.rect.height,
                        )
                        anchor = pymupdf.Point(
                            kept[0] * page.rect.width,
                            kept[1] * page.rect.height,
                        )
                else:
                    normalized_x, normalized_y = _validate_position(x, y)
                    anchor = pymupdf.Point(
                        normalized_x * page.rect.width,
                        normalized_y * page.rect.height,
                    )
                page.delete_annot(annotation)
                return _insert_comment(
                    document,
                    page,
                    anchor.x,
                    anchor.y,
                    text.strip(),
                    highlighted,
                    color,
                    comment_id,
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
