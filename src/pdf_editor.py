"""Edit a PDF on a separate working copy - safely.

Stage 6A of the AI Document Learning Assistant (Feature B: editing workspace).

The golden rule of the project: **the uploaded PDF is never modified.** Every
function here takes the original bytes, makes changes on a copy, and returns
NEW bytes. The caller keeps the original untouched and swaps the working copy.

Three operations (kept deliberately small; more come later):

    delete_pages   - remove pages (structure change, pypdf)
    rotate_page    - turn a page 90/180/270 degrees (structure change, pypdf)
    insert_blank_page    - add one empty page (structure, pypdf)
    insert_pages_from    - add ALL pages of another PDF (merge/combine, pypdf)
    move_page            - rearrange: move one page to a new position (pypdf)
    replace_text   - find text on ONE page and replace it:
                        native text page  -> real text edit (pymupdf redaction)
                        scanned page      -> white-out the old OCR words and
                                             insert visible, selectable text
    save_text_layer - replace scanned OCR text with the edited visible text,
                      e.g. the output of "Extract & edit OCR text"

Every function returns an `EditedPdf`: the new bytes + how many changes were
made + where the text came from ("structure", "native text", "scanned page
(OCR)"). Invalid input raises `ValueError` with a friendly, plain message.
"""

from dataclasses import dataclass
from difflib import SequenceMatcher
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Optional

import pymupdf
from pypdf import PageObject, PdfReader, PdfWriter

from src.ocr import OcrWord, needs_ocr, ocr_page, page_needs_ocr


@dataclass
class EditedPdf:
    """Result of one editing operation (new bytes, never the caller's input)."""

    pdf_bytes: bytes
    changes: int  # how many pages/occurrences were touched
    origin: str  # "structure" | "native text" | "scanned page (OCR)"


@dataclass(frozen=True)
class _ReplacementTextStyle:
    baseline: float
    fontsize: float
    fontname: str
    color: tuple[float, float, float]
    font_data: bytes | None = None
    font_extension: str | None = None


def _open_with_pypdf(pdf_bytes: bytes) -> PdfReader:
    """Open a PDF for structural work, with a friendly message on failure."""
    try:
        reader = PdfReader(BytesIO(pdf_bytes))
        _ = len(reader.pages)  # force a read so a corrupt file fails HERE
        return reader
    except Exception as error:  # pypdf has many error types; message matters
        raise ValueError(f"Could not open this PDF for editing: {error}") from error


def _open_with_pymupdf(pdf_bytes: bytes) -> pymupdf.Document:
    """Open a PDF for text work, with a friendly message on failure."""
    try:
        document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        _ = document.page_count
        return document
    except Exception as error:
        raise ValueError(f"Could not open this PDF for editing: {error}") from error


def _check_page_number(page_number: int, total: int) -> None:
    if not 1 <= page_number <= total:
        raise ValueError(
            f"Page {page_number} does not exist - this document has {total} page(s)."
        )


def delete_pages(pdf_bytes: bytes, page_numbers: List[int]) -> EditedPdf:
    """Delete the given 1-based pages. The original bytes are never touched."""
    if not page_numbers:
        raise ValueError("Select at least one page to delete.")

    reader = _open_with_pypdf(pdf_bytes)
    total = len(reader.pages)
    targets = sorted({int(number) for number in page_numbers})
    for number in targets:
        _check_page_number(number, total)
    if len(targets) >= total:
        raise ValueError(
            "You cannot delete every page - a PDF must keep at least one page."
        )

    writer = PdfWriter()
    for one_based_index, page in enumerate(reader.pages, start=1):
        if one_based_index not in targets:
            writer.add_page(page)

    buffer = BytesIO()
    writer.write(buffer)
    return EditedPdf(pdf_bytes=buffer.getvalue(), changes=len(targets), origin="structure")


def rotate_page(pdf_bytes: bytes, page_number: int, degrees: int = 90) -> EditedPdf:
    """Rotate ONE page clockwise by a multiple of 90 degrees."""
    if degrees not in (90, 180, 270):
        raise ValueError("Degrees must be 90, 180 or 270 (clockwise).")

    reader = _open_with_pypdf(pdf_bytes)
    total = len(reader.pages)
    _check_page_number(int(page_number), total)

    page = reader.pages[int(page_number) - 1]
    page.rotate(degrees)  # pypdf sets /Rotate; the page CONTENT is unchanged

    writer = PdfWriter()
    for one_based_index, other in enumerate(reader.pages, start=1):
        writer.add_page(other)  # the rotated page object is added as-is

    buffer = BytesIO()
    writer.write(buffer)
    return EditedPdf(pdf_bytes=buffer.getvalue(), changes=1, origin="structure")


# --- Page insertion / combination --------------------------------------------


def insert_blank_page(pdf_bytes: bytes, after_page: int = 0) -> EditedPdf:
    """Insert ONE empty page AFTER the given 1-based page.

    `after_page=0` means "at the very beginning". The page size follows the
    common US Letter format (612 x 792 points).
    """
    reader = _open_with_pypdf(pdf_bytes)
    total = len(reader.pages)
    after_page = int(after_page)
    if not 0 <= after_page <= total:
        raise ValueError(
            f"Cannot insert after page {after_page} - this document has "
            f"{total} page(s). (0 means 'at the very beginning'.)"
        )

    blank = PageObject.create_blank_page(width=612, height=792)
    writer = PdfWriter()
    if after_page == 0:
        writer.add_page(blank)
    for one_based_index, page in enumerate(reader.pages, start=1):
        writer.add_page(page)
        if one_based_index == after_page:
            writer.add_page(blank)

    buffer = BytesIO()
    writer.write(buffer)
    return EditedPdf(pdf_bytes=buffer.getvalue(), changes=1, origin="structure")


def insert_pages_from(
    pdf_bytes: bytes, other_pdf_bytes: bytes, before_page_number: Optional[int] = None
) -> EditedPdf:
    """Insert ALL pages of another PDF into this one (merge / combine).

    `before_page_number=None` appends at the END of the document; otherwise
    the pages go BEFORE the given 1-based page (e.g. 1 = very beginning).
    """
    reader = _open_with_pypdf(pdf_bytes)
    other = _open_with_pypdf(other_pdf_bytes)
    total = len(reader.pages)
    other_total = len(other.pages)
    if other_total == 0:
        raise ValueError("The other PDF has no pages to insert.")

    if before_page_number is None:
        insert_position = total  # 0-based index == page count = append at end
    else:
        insert_position = int(before_page_number) - 1
        if not 0 <= insert_position <= total:
            raise ValueError(
                f"Cannot insert before page {insert_position + 1} - this "
                f"document has {total} page(s)."
            )

    writer = PdfWriter()
    inserted = False
    for zero_based_index, page in enumerate(reader.pages):
        if zero_based_index == insert_position:
            for other_page in other.pages:
                writer.add_page(other_page)
            inserted = True
        writer.add_page(page)
    if not inserted:  # insert position == end (append)
        for other_page in other.pages:
            writer.add_page(other_page)

    buffer = BytesIO()
    writer.write(buffer)
    return EditedPdf(pdf_bytes=buffer.getvalue(), changes=other_total, origin="structure")


def apply_page_order(pdf_bytes: bytes, page_order: List[int]) -> EditedPdf:
    """Rewrite the whole document in the given page order (rearrange).

    `page_order` must contain every existing page number EXACTLY once, e.g.
    [3, 1, 2] puts old page 3 first. Used by the drag-and-drop list.
    """
    reader = _open_with_pypdf(pdf_bytes)
    total = len(reader.pages)
    original_order = list(range(1, total + 1))

    cleaned = [int(number) for number in page_order]
    if sorted(cleaned) != original_order:
        raise ValueError(
            "The new order must contain every page number of the document "
            f"(1 to {total}) exactly once."
        )

    writer = PdfWriter()
    for number in cleaned:
        writer.add_page(reader.pages[number - 1])

    buffer = BytesIO()
    writer.write(buffer)
    return EditedPdf(pdf_bytes=buffer.getvalue(), changes=total, origin="structure")


def move_page(
    pdf_bytes: bytes, page_number: int, new_position: int
) -> EditedPdf:
    """Move ONE page to a new position (1-based) - rearrange pages.

    Example: document A, B, C. move_page(x, 1, 2) -> B, A, C.
    Moving to the SAME position is a no-op (original bytes returned).
    """
    reader = _open_with_pypdf(pdf_bytes)
    total = len(reader.pages)
    _check_page_number(int(page_number), total)
    new_position = int(new_position)
    if not 1 <= new_position <= total:
        raise ValueError(
            f"New position {new_position} does not exist - this document has "
            f"{total} page(s)."
        )

    order = list(range(total))
    moving_page = order.pop(int(page_number) - 1)
    order.insert(new_position - 1, moving_page)

    writer = PdfWriter()
    for source_index in order:
        writer.add_page(reader.pages[source_index])

    buffer = BytesIO()
    writer.write(buffer)
    return EditedPdf(pdf_bytes=buffer.getvalue(), changes=1, origin="structure")


# --- Text replacement ----------------------------------------------------------


# Smallest font we will shrink a replacement down to before calling it
# impossible to place without overlapping the neighbouring text.
_MIN_REPLACEMENT_FONTSIZE = 4.0


def _group_match_rects(rects: list[pymupdf.Rect]) -> list[list[pymupdf.Rect]]:
    """Group the per-line rects of one match into single occurrences.

    `page.search_for` returns one rect per line when the searched phrase
    wraps onto the next line. Those rects belong to the SAME occurrence, so
    the replacement must be inserted once per group - inserting once per
    rect would duplicate the new text on every line.
    """
    groups: list[list[pymupdf.Rect]] = []
    for rect in sorted(rects, key=lambda item: (item.y0, item.x0)):
        if groups and rect.y0 - groups[-1][-1].y1 < 0.5 * rect.height:
            groups[-1].append(rect)
        else:
            groups.append([rect])
    return groups


def _scan_backed_page(page: pymupdf.Page, rects: list[pymupdf.Rect]) -> bool:
    """Does a page-sized image lie behind the matched text?

    Scanned PDFs usually carry an invisible OCR text layer ON TOP of the page
    image. The words are therefore "selectable" but their visible form is
    burned into the image: redacting only the text layer leaves the original
    words visible and the replacement overlaps them.
    """
    page_area = max(1.0, page.rect.get_area())
    big_image_rects = []
    for image in page.get_images(full=True):
        for image_rect in page.get_image_rects(image[0]):
            if image_rect.get_area() >= 0.5 * page_area:
                big_image_rects.append(image_rect)
    return any(
        image_rect.intersects(match)
        for image_rect in big_image_rects
        for match in rects
    )


def _available_replacement_width(
    page: pymupdf.Page, rect: pymupdf.Rect, redacted_rects: list[pymupdf.Rect]
) -> float:
    """How many points of room are to the right of `rect` on its own line?

    The limit is the left edge of the NEXT span that will still exist after the
    redaction (i.e. the word printed right after the matched text), capped at
    the page edge. That is exactly how far replacement text may grow before it
    would collide with the neighbouring text.
    """
    limit = float(page.rect.x1)
    redacted = [pymupdf.Rect(other) for other in redacted_rects]
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                span_rect = pymupdf.Rect(span["bbox"])
                # Same visual row as the matched text? (Spans of one line can
                # live in different blocks, so compare the vertical bands.)
                if span_rect.y1 <= rect.y0 or span_rect.y0 >= rect.y1:
                    continue  # a different line above/below
                if span_rect.x0 < rect.x1 - 0.5:
                    continue  # the matched text itself (or text before it)
                if any(span_rect.intersects(other) for other in redacted):
                    continue  # also being removed with this replacement
                limit = min(limit, span_rect.x0)
    return max(0.0, limit - rect.x0)


def _fitted_replacement_fontsize(
    page: pymupdf.Page,
    rect: pymupdf.Rect,
    new_text: str,
    style: _ReplacementTextStyle,
    redacted_rects: list[pymupdf.Rect],
) -> float:
    """Keep the source font size when the new text fits - shrink when it does not.

    - Fits at the original size  -> return `style.fontsize`, so the replacement
      matches the surrounding text's font size exactly.
    - Too wide                   -> shrink proportionally so it ends before the
      next surviving word (never overlaps).
    - Won't fit even at the floor -> raise a friendly error instead of drawing
      over the neighbouring text.
    """
    available = _available_replacement_width(page, rect, redacted_rects)
    if available <= 0:
        return style.fontsize  # nothing follows on this line; keep the style

    try:
        measure_font = (
            pymupdf.Font(fontbuffer=style.font_data)
            if style.font_data
            else pymupdf.Font(fontname=style.fontname)
        )
    except (RuntimeError, ValueError):
        measure_font = pymupdf.Font(fontname="helv")

    # insert_text draws each "\n" line at the same x, so the widest line rules.
    text_width = max(
        (
            measure_font.text_length(line, fontsize=style.fontsize)
            for line in new_text.split("\n")
        ),
        default=0.0,
    )
    if text_width <= available:
        return style.fontsize  # no overlap possible: match the original size

    smallest_that_fits = available / text_width * style.fontsize
    if smallest_that_fits < _MIN_REPLACEMENT_FONTSIZE:
        raise ValueError(
            "The replacement text is too long for this spot and would overlap "
            "the text next to it. Please use a shorter replacement."
        )
    return smallest_that_fits


def _replace_on_native_page(
    page: pymupdf.Page,
    old_text: str,
    new_text: str,
    page_number: int,
    rects: list[pymupdf.Rect] | None = None,
) -> int:
    """Redact native text and reinsert it with the closest source styling.

    Returns how many occurrences were replaced.
    """
    if rects is None:
        rects = page.search_for(old_text)
    if not rects:
        raise ValueError(
            f"'{old_text}' was not found on page {page_number} as selectable text. "
            "If this page is a scan, the search words must match what OCR reads."
        )

    groups = _group_match_rects(rects)
    styles = [_native_text_style(page, group[0], new_text) for group in groups]
    # Fit BEFORE redacting (while the surrounding text is still measurable), so
    # a too-long replacement can never run over the word printed next to it.
    fitted_sizes = [
        _fitted_replacement_fontsize(page, group[0], new_text, style, rects)
        for group, style in zip(groups, styles)
    ]
    for group in groups:
        for rect in group:
            page.add_redact_annot(rect)  # mark the area: "erase what is underneath"
    # PDF_REDACT_IMAGE_NONE = do NOT delete the images of the page (a scan page
    # is one big image; we only want the text gone).
    page.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_NONE)

    if new_text.strip():
        with TemporaryDirectory() as temporary_directory:
            for index, (rect, style, fitted_size) in enumerate(
                zip(
                    [group[0] for group in groups],
                    styles,
                    fitted_sizes,
                )
            ):
                fontfile = None
                if style.font_data and style.font_extension:
                    fontfile = Path(temporary_directory) / (
                        f"replacement_{index}.{style.font_extension}"
                    )
                    fontfile.write_bytes(style.font_data)
                page.insert_text(
                    (rect.x0, style.baseline),
                    new_text,
                    fontsize=fitted_size,
                    fontname=style.fontname,
                    fontfile=str(fontfile) if fontfile else None,
                    color=style.color,
                )
    return len(groups)


def _native_text_style(
    page: pymupdf.Page, rect: pymupdf.Rect, new_text: str
) -> _ReplacementTextStyle:
    """Find the source span style and reuse its font when it supports the text."""
    spans = [
        span
        for block in page.get_text("dict")["blocks"]
        if "lines" in block
        for line in block["lines"]
        for span in line["spans"]
        if pymupdf.Rect(span["bbox"]).intersects(rect)
    ]
    if not spans:
        raise ValueError("Could not determine the original text's style.")
    span = max(
        spans,
        key=lambda item: pymupdf.Rect(item["bbox"]).intersect(rect).get_area(),
    )

    flags = int(span.get("flags", 0))
    family = "cour" if flags & 8 else "tiro" if flags & 4 else "helv"
    bold = bool(flags & 16)
    italic = bool(flags & 2)
    builtin_fonts = {
        ("helv", False, False): "helv",
        ("helv", True, False): "hebo",
        ("helv", False, True): "heit",
        ("helv", True, True): "hebi",
        ("cour", False, False): "cour",
        ("cour", True, False): "cobo",
        ("cour", False, True): "coit",
        ("cour", True, True): "cobi",
        ("tiro", False, False): "tiro",
        ("tiro", True, False): "tibo",
        ("tiro", False, True): "tiit",
        ("tiro", True, True): "tibi",
    }
    font_name = builtin_fonts[(family, bold, italic)]
    font_data = None
    font_extension = None

    font_resources = page.get_fonts(full=True)
    for font_resource in font_resources:
        xref, _, _, base_font, _, *_ = font_resource
        extracted_font = page.parent.extract_font(xref)
        candidate_data = extracted_font[3]
        if not candidate_data:
            continue
        try:
            source_font = pymupdf.Font(fontbuffer=candidate_data)
            source_name = source_font.name.split("+")[-1].casefold()
            base_name = str(base_font).split("+")[-1].casefold()
            span_name = str(span["font"]).split("+")[-1].casefold()
            if source_name not in {base_name, span_name}:
                continue
            if all(source_font.has_glyph(ord(character)) for character in new_text):
                font_name = f"replacement_{xref}"
                font_data = candidate_data
                font_extension = str(extracted_font[1])
                break
        except (RuntimeError, ValueError):
            continue

    color_value = int(span.get("color", 0))
    color = (
        ((color_value >> 16) & 255) / 255,
        ((color_value >> 8) & 255) / 255,
        (color_value & 255) / 255,
    )
    return _ReplacementTextStyle(
        baseline=float(span["origin"][1]),
        fontsize=float(span["size"]),
        fontname=font_name,
        color=color,
        font_data=font_data,
        font_extension=font_extension,
    )


def _replace_on_hybrid_page(
    document: pymupdf.Document,
    page: pymupdf.Page,
    pdf_bytes: bytes,
    page_number: int,
    old_text: str,
    new_text: str,
    rects: list[pymupdf.Rect],
) -> int:
    """Replace text on a scan-backed page (image + invisible OCR text layer).

    The visible words are burned into the page image while the searchable
    words are an invisible text layer printed on top. A plain redaction only
    removes that invisible layer, so the old words stay VISIBLE and the
    replacement overlaps them. Instead this:
      1. redacts the invisible text layer (keeps search and the AI in sync),
      2. white-outs the burned-in words on the image (deletes the visible
         original),
      3. inserts the replacement with the visually nearest local font and
         the matched size (shrunk only when it would not fit).
    """
    from src.pdf_font_estimation import estimate_scanned_page_fonts

    groups = _group_match_rects(rects)
    try:
        font_estimate = estimate_scanned_page_fonts(
            pdf_bytes, page_number, top_n=1
        )
    except (ValueError, RuntimeError):
        font_estimate = None  # no OCR engine / no fonts: fall back to the layer style

    styles = []
    for group in groups:
        base_style = _native_text_style(page, group[0], new_text)
        font_size = base_style.fontsize
        font_data = base_style.font_data
        font_extension = base_style.font_extension
        if font_estimate is not None and font_estimate.lines:
            group_rect = pymupdf.Rect(group[0])
            for rect in group[1:]:
                group_rect |= rect
            matching_line = max(
                font_estimate.lines,
                key=lambda line: pymupdf.Rect(line.bounds)
                .intersect(group_rect)
                .get_area(),
            )
            if matching_line.matches:
                best_match = matching_line.matches[0]
                font_size = float(best_match.estimated_size_points)
                if best_match.font_path:
                    font_path = Path(best_match.font_path)
                    try:
                        font_data = font_path.read_bytes()
                        font_extension = font_path.suffix.lstrip(".") or "ttf"
                    except OSError:
                        font_data = None
                else:
                    font_data = None
        styles.append(
            _ReplacementTextStyle(
                baseline=base_style.baseline,
                fontsize=font_size,
                fontname="replacement_hybrid" if font_data else "helv",
                color=base_style.color,
                font_data=font_data,
                font_extension=font_extension,
            )
        )

    fitted_sizes = [
        _fitted_replacement_fontsize(page, group[0], new_text, style, rects)
        for group, style in zip(groups, styles)
    ]
    # 1. Erase the invisible text layer (search + AI keep working).
    for group in groups:
        for rect in group:
            page.add_redact_annot(rect)
    page.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_NONE)
    # 2. Delete the visible original burned into the page image.
    for group in groups:
        cover = pymupdf.Rect(group[0])
        for rect in group[1:]:
            cover |= rect
        page.draw_rect(cover + (-2, -2, 2, 2), color=(1, 1, 1), fill=(1, 1, 1))

    # 3. Print the replacement with the matched font, on top of the white-out.
    if new_text.strip():
        with TemporaryDirectory() as temporary_directory:
            for index, (rect, style, fitted_size) in enumerate(
                zip(
                    [group[0] for group in groups],
                    styles,
                    fitted_sizes,
                )
            ):
                fontfile = None
                if style.font_data and style.font_extension:
                    fontfile = Path(temporary_directory) / (
                        f"replacement_{index}.{style.font_extension}"
                    )
                    fontfile.write_bytes(style.font_data)
                page.insert_text(
                    (rect.x0, style.baseline),
                    new_text,
                    fontsize=fitted_size,
                    fontname=style.fontname,
                    fontfile=str(fontfile) if fontfile else None,
                    color=style.color,
                )
    return len(groups)


def _fitted_scanned_fontsize(
    page: pymupdf.Page,
    words: list,
    left: float,
    top: float,
    width: float,
    height: float,
    new_text: str,
    best_match,
    removed_boxes: list[tuple[float, float, float, float]],
) -> float:
    """Shrink a scanned-page replacement so it never covers the next OCR word.

    Same rule as the native path: keep the estimated source size when the new
    text fits in the gap up to the next surviving word, otherwise shrink it.
    """
    limit = float(page.rect.x1)
    line_bottom = top + height
    removed_rects = [
        pymupdf.Rect(box[0], box[1], box[0] + box[2], box[1] + box[3])
        for box in removed_boxes
    ]
    for word in words:
        word_rect = pymupdf.Rect(
            word.left, word.top, word.left + word.width, word.top + word.height
        )
        if word_rect.y1 <= top + 1 or word_rect.y0 >= line_bottom - 1:
            continue  # a different line of the scan
        if word_rect.x0 < left + width - 0.5:
            continue  # part of the replaced words (or text before them)
        if any(word_rect.intersects(box) for box in removed_rects):
            continue  # also being white-out with this replacement
        limit = min(limit, word_rect.x0)
    available = limit - left

    try:
        font = (
            pymupdf.Font(fontfile=str(best_match.font_path))
            if best_match.font_path
            else pymupdf.Font(fontname="helv")
        )
    except (RuntimeError, ValueError):
        font = pymupdf.Font(fontname="helv")

    size = float(best_match.estimated_size_points)
    text_width = max(
        (
            font.text_length(line, fontsize=size)
            for line in new_text.split("\n")
        ),
        default=0.0,
    )
    if available <= 0 or text_width <= available:
        return size
    smallest_that_fits = available / text_width * size
    if smallest_that_fits < _MIN_REPLACEMENT_FONTSIZE:
        raise ValueError(
            "The replacement text is too long for this spot and would overlap "
            "the text next to it. Please use a shorter replacement."
        )
    return smallest_that_fits


def _replace_on_scanned_page(
    document: pymupdf.Document,
    pdf_bytes: bytes,
    page_number: int,
    old_text: str,
    new_text: str,
) -> int:
    """Visual replacement on a scanned page, using OCR word boxes.

    The scan is a picture, so there is no text to redact. Instead we:
      1. OCR the page (Stage 5B) to get each word AND its box,
      2. find where the searched phrase appears in those words,
      3. paint a white rectangle over the old (image) words,
      4. insert REAL text with the new words on top.

    After this, the page has real text again - the next extraction/OCR cycle
    sees the new words.
    """
    # Late import avoids a circular import (document_search imports pdf_loader).
    from src.document_search import WordBox, _match_tokens, _union_box

    page = document[int(page_number) - 1]
    ocr_result = ocr_page(pdf_bytes, page.number + 1)

    words = [
        WordBox(
            text=word.text,
            left=word.left,
            top=word.top,
            width=word.width,
            height=word.height,
        )
        for word in ocr_result.words
    ]
    lowered_tokens = [token.casefold() for token in old_text.split()]
    groups = _match_tokens(words, lowered_tokens)
    if not groups:
        raise ValueError(
            f"'{old_text}' was not found on page {page.number + 1}. On scanned "
            "pages the OCR reading must match - try searching fewer words."
        )

    from src.pdf_font_estimation import estimate_scanned_page_fonts

    font_estimate = estimate_scanned_page_fonts(
        pdf_bytes, page.number + 1, top_n=1
    )
    if not font_estimate.lines:
        raise ValueError(
            "Could not estimate a matching font for this scanned text."
        )

    for group in groups:
        left, top, width, height = _union_box(group)
        cover = pymupdf.Rect(left - 2, top - 2, left + width + 2, top + height + 2)
        page.draw_rect(cover, color=(1, 1, 1), fill=(1, 1, 1))  # white-out
        if new_text.strip():
            group_rect = pymupdf.Rect(
                left, top, left + width, top + height
            )
            matching_line = max(
                font_estimate.lines,
                key=lambda line: pymupdf.Rect(line.bounds)
                .intersect(group_rect)
                .get_area(),
            )
            if not matching_line.matches:
                raise ValueError(
                    "Could not estimate a matching font for this scanned text."
                )
            best_match = matching_line.matches[0]
            fitted_size = _fitted_scanned_fontsize(
                page=page,
                words=words,
                left=left,
                top=top,
                width=width,
                height=height,
                new_text=new_text,
                best_match=best_match,
                removed_boxes=[_union_box(other) for other in groups],
            )
            page.insert_text(
                (left, top + height * 0.8),
                new_text,
                fontsize=fitted_size,
                fontfile=best_match.font_path,
            )
    return len(groups)


def replace_text(
    pdf_bytes: bytes, page_number: int, old_text: str, new_text: str = ""
) -> EditedPdf:
    """Replace `old_text` with `new_text` on ONE page of the working copy.

    Works on native text pages AND scanned pages (via OCR, see
    `_replace_on_scanned_page`). Empty `new_text` simply deletes the old text.
    """
    if not old_text.strip():
        raise ValueError("Type the text you want to find first.")

    document = _open_with_pymupdf(pdf_bytes)
    try:
        _check_page_number(int(page_number), document.page_count)
        page = document[int(page_number) - 1]

        page_text = page.get_text()
        # CAREFUL: a page with very LITTLE text is not necessarily a scan.
        # `needs_ocr` (a < 20 characters heuristic) would misjudge a short
        # native page. So the decision is made by what actually works:
        # if the words are selectable text -> real text edit; only when the
        # page is really image-only do we go the OCR route. A scan-backed
        # page (page image + invisible OCR text layer) gets the hybrid
        # treatment so the VISIBLE original is deleted too.
        match_rects = page.search_for(old_text) if page_text.strip() else []
        if match_rects and _scan_backed_page(page, match_rects):
            changes = _replace_on_hybrid_page(
                document,
                page,
                pdf_bytes,
                int(page_number),
                old_text,
                new_text,
                match_rects,
            )
            origin = "scan-backed page (OCR layer)"
        elif match_rects:
            changes = _replace_on_native_page(
                page, old_text, new_text, int(page_number), match_rects
            )
            origin = "native text"
        elif needs_ocr(page_text) and page.get_images(full=True):
            changes = _replace_on_scanned_page(
                document, pdf_bytes, int(page_number), old_text, new_text
            )
            origin = "scanned page (OCR)"
        else:
            raise ValueError(
                f"'{old_text}' was not found on page {int(page_number)} as "
                "selectable text. If this page is a scan, the search words "
                "must match what OCR reads."
            )

        return EditedPdf(pdf_bytes=document.tobytes(), changes=changes, origin=origin)
    finally:
        document.close()


def _wrapped_text_lines(
    text: str, font: pymupdf.Font, fontsize: float, max_width: float
) -> list[str]:
    """Wrap edited lines to the available page width."""
    wrapped: list[str] = []
    for paragraph in text.splitlines():
        current = ""
        for word in paragraph.split():
            candidate = f"{current} {word}".strip()
            if current and font.text_length(candidate, fontsize) > max_width:
                wrapped.append(current)
                current = word
            else:
                current = candidate
        if current:
            wrapped.append(current)
        elif not paragraph:
            wrapped.append("")
    return wrapped


def _draw_marked_ocr_edit(
    page: pymupdf.Page,
    text: str,
    left: float,
    top: float,
    original_right: float,
    original_bottom: float,
    available_bottom: float,
    font_path: str | Path | None = None,
) -> None:
    """Replace OCR content in its location and mark the edited area."""
    margin = 3.0
    available_width = page.rect.x1 - left - margin * 2
    available_height = available_bottom - top - margin
    if available_width <= 0 or available_height <= 0:
        raise ValueError("There is not enough page space to place this OCR edit.")

    font_file = str(font_path) if font_path is not None else None
    try:
        font = (
            pymupdf.Font(fontfile=font_file)
            if font_file
            else pymupdf.Font("helv")
        )
    except Exception as error:
        raise ValueError(f"Could not load the selected local font: {error}") from error
    font_name = "ocr_custom" if font_file else "helv"
    fontsize = max(6.0, min(12.0, (original_bottom - top) * 0.75))
    wrapped: list[str] = []
    line_height = fontsize * 1.2
    while fontsize >= 6.0:
        wrapped = _wrapped_text_lines(text, font, fontsize, available_width)
        line_height = fontsize * 1.2
        content_height = max(
            original_bottom - top,
            len(wrapped) * line_height,
        )
        if content_height <= available_height:
            break
        fontsize -= 0.5
    else:
        raise ValueError(
            "This edit does not fit in the available OCR text area. "
            "Shorten it or keep the added text on the same line."
        )

    content_height = max(original_bottom - top, len(wrapped) * line_height)
    text_width = max(
        (font.text_length(line, fontsize) for line in wrapped),
        default=0.0,
    )
    cover = pymupdf.Rect(
        left - margin,
        top - margin,
        min(page.rect.x1, max(original_right + margin, left + text_width + margin)),
        min(page.rect.y1, top + content_height + margin),
    )
    page.draw_rect(
        cover,
        color=(0.45, 0.5, 0.55),
        fill=(1, 1, 1),
        fill_opacity=1,
        width=0.5,
        overlay=True,
    )
    if not text.strip():
        return

    text_box = pymupdf.Rect(
        left,
        top - 1,
        page.rect.x1 - margin,
        min(page.rect.y1, top + content_height + margin),
    )
    remaining = page.insert_textbox(
        text_box,
        "\n".join(wrapped),
        fontname=font_name,
        fontfile=font_file,
        fontsize=fontsize,
        lineheight=1.2,
        color=(0.1, 0.1, 0.1),
    )
    if remaining < 0:
        raise ValueError(
            "This edit does not fit in the available OCR text area. "
            "Shorten it and try again."
        )


def save_text_layer(
    pdf_bytes: bytes,
    page_number: int,
    text: str,
    ocr_words: Optional[List[OcrWord]] = None,
    font_path: str | Path | None = None,
) -> EditedPdf:
    """Replace scanned OCR text and clearly mark the edited sections.

    OCR reads an image page, the user checks/edits the text, and saving
    covers the detected words and prints the edited text in their place.
    Edited areas are outlined, and unchanged OCR text stays searchable.
    Pages that already contain selectable text must use `replace_text`.

    Pass `ocr_words` from the preceding OCR read when available to avoid
    recognizing the same page twice.
    """
    if not text.strip():
        raise ValueError("Type or paste the text to place on the page first.")

    document = _open_with_pymupdf(pdf_bytes)
    try:
        _check_page_number(int(page_number), document.page_count)
        page = document[int(page_number) - 1]

        if not page_needs_ocr(pdf_bytes, int(page_number)):
            raise ValueError(
                "This page does not appear to be a scanned/image-only page. "
                "OCR replacement is only for scanned pages; use Replace text "
                "to edit selectable text."
            )
        origin = "scanned page (text layer)"

        if ocr_words is None:
            ocr_words = ocr_page(pdf_bytes, int(page_number)).words
        if not ocr_words:
            raise ValueError(
                "OCR did not find any word positions on this page. "
                "Read the page with OCR again before saving."
            )

        words_by_line: dict[int, list[OcrWord]] = {}
        for word in ocr_words:
            words_by_line.setdefault(word.line_number, []).append(word)
        detected_lines = [
            sorted(line_words, key=lambda word: word.left)
            for _, line_words in sorted(words_by_line.items())
        ]
        original_lines = [
            " ".join(word.text for word in line_words)
            for line_words in detected_lines
        ]
        edited_lines = text.splitlines()
        original_normalized = [" ".join(line.split()) for line in original_lines]
        edited_normalized = [" ".join(line.split()) for line in edited_lines]
        changes = SequenceMatcher(
            None, original_normalized, edited_normalized, autojunk=False
        )

        for operation, old_start, old_end, new_start, new_end in changes.get_opcodes():
            if operation == "equal":
                for line_words in detected_lines[old_start:old_end]:
                    for word in line_words:
                        page.insert_text(
                            (word.left, word.top + word.height * 0.85),
                            word.text,
                            fontsize=max(1.0, min(72.0, float(word.height))),
                            render_mode=3,
                        )
                continue

            if operation == "insert":
                if old_start == 0:
                    raise ValueError(
                        "A new line before the first detected line has no reliable "
                        "position. Add its text to the first line instead."
                    )

                previous_line = detected_lines[old_start - 1]
                left = min(word.left for word in previous_line)
                previous_bottom = max(
                    word.top + word.height for word in previous_line
                )
                fontsize = max(
                    6.0,
                    min(14.0, max(word.height for word in previous_line) * 0.85),
                )
                top = previous_bottom + 5
                if old_start < len(detected_lines):
                    available_bottom = min(
                        word.top for word in detected_lines[old_start]
                    ) - 1
                else:
                    available_bottom = min(page.rect.y1 - 12, top + fontsize * 2)

                _draw_marked_ocr_edit(
                    page,
                    "\n".join(edited_lines[new_start:new_end]),
                    left,
                    top,
                    left,
                    top + fontsize,
                    available_bottom,
                    font_path,
                )
                continue

            line_words_group = detected_lines[old_start:old_end]
            old_words = [word for line in line_words_group for word in line]
            left = min(word.left for word in old_words)
            top = min(word.top for word in old_words)
            right = max(word.left + word.width for word in old_words)
            bottom = max(word.top + word.height for word in old_words)
            next_line_top = (
                min(word.top for word in detected_lines[old_end])
                if old_end < len(detected_lines)
                else page.rect.y1 - 12
            )
            edited_block = "\n".join(edited_lines[new_start:new_end])
            _draw_marked_ocr_edit(
                page,
                edited_block,
                left,
                top,
                right,
                bottom,
                next_line_top - 1,
                font_path,
            )

        return EditedPdf(pdf_bytes=document.tobytes(), changes=1, origin=origin)
    finally:
        document.close()
