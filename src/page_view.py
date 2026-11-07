"""Render a page as a picture, with optional highlighted boxes.

Stage 5.5 of the AI Document Learning Assistant ("real highlighting").

This connects the coordinates we already collect - from Find-in-document
(Stage 5E, native text), or from OCR word boxes (Stage 5B, scanned pages) -
to something the user can SEE: an image of the page with red, semi-
transparent rectangles over the matches.

Nothing is saved: the rectangles are drawn on an in-memory COPY of the page
(workflow rule - the original PDF bytes are never modified), then the copy is
rendered to a PNG and thrown away.
"""

import base64
import html
from io import BytesIO

import pymupdf


def render_selectable_page_html(
    pdf_bytes: bytes, page_number: int, dpi: int = 150
) -> str:
    """Render a page with a transparent selectable text layer for browsers."""
    if not pdf_bytes:
        raise ValueError("No PDF data was given to preview.")
    if dpi < 1:
        raise ValueError("dpi must be at least 1.")

    document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    try:
        if not 1 <= page_number <= document.page_count:
            raise ValueError(
                f"Page {page_number} does not exist - this document has "
                f"{document.page_count} page(s)."
            )
        page = document[page_number - 1]
        pixmap = page.get_pixmap(dpi=dpi)
        image_data = base64.b64encode(pixmap.tobytes("png")).decode("ascii")
        page_rect = page.rect
        scale_x = 100.0 / page_rect.width
        scale_y = 100.0 / page_rect.height
        text_spans = [
            span
            for block in page.get_text("dict")["blocks"]
            if "lines" in block
            for line in block["lines"]
            for span in line["spans"]
            if span["text"].strip()
        ]
        text_items: list[tuple[str, pymupdf.Rect, float]] = [
            (
                str(span["text"]),
                pymupdf.Rect(span["bbox"]) * page.rotation_matrix,
                float(span["size"]),
            )
            for span in text_spans
        ]
        if not text_items and page.get_images(full=True):
            from src.ocr import ocr_page

            ocr_result = ocr_page(pdf_bytes, page_number)
            text_items = [
                (
                    word.text,
                    pymupdf.Rect(
                        word.left,
                        word.top,
                        word.left + word.width,
                        word.top + word.height,
                    )
                    * page.rotation_matrix,
                    float(word.height) * 0.8,
                )
                for word in ocr_result.words
            ]

        text_layer = []
        for text, rect, font_size in text_items:
            left = (rect.x0 - page_rect.x0) * scale_x
            top = (rect.y0 - page_rect.y0) * scale_y
            width = rect.width * scale_x
            height = rect.height * scale_y
            size = font_size * scale_x
            text_layer.append(
                '<span class="selectable-text" '
                f'style="left:{left:.5f}%;top:{top:.5f}%;'
                f'width:{width:.5f}%;height:{height:.5f}%;'
                f'font-size:{size:.5f}cqw;line-height:{height:.5f}cqw">'
                f"{html.escape(text)}</span>"
            )

        layer_html = "".join(text_layer)
        return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<style>
  * {{ box-sizing: border-box; }}
  html, body {{ margin: 0; padding: 0; background: transparent; }}
  .viewport {{ height: 740px; overflow: auto; padding: 8px;
               position: relative; }}
  .zoom-controls {{ position: sticky; top: 8px; right: 8px; z-index: 3;
                    display: flex; justify-content: flex-end; gap: 4px;
                    height: 0; overflow: visible; }}
  .zoom-controls button {{ width: 34px; height: 34px; border: 1px solid #888;
                           border-radius: 5px; background: #fff; color: #222;
                           font: 20px/1 sans-serif; cursor: pointer;
                           box-shadow: 0 1px 4px #0004; }}
  .zoom-controls button:disabled {{ color: #aaa; cursor: default; }}
  .page {{ position: relative; width: calc(100% * var(--zoom, 1));
           max-width: none; margin: 0 auto; container-type: inline-size; }}
  .page-image {{ display: block; width: 100%; height: auto; }}
  .text-layer {{ position: absolute; inset: 0; user-select: text;
                 -webkit-user-select: text; }}
  .selectable-text {{ position: absolute; display: block; overflow: visible;
                      white-space: pre; color: rgba(0, 0, 0, 0.005);
                      user-select: text; -webkit-user-select: text;
                      cursor: text; }}
</style>
</head>
<body>
  <div class="viewport">
    <div class="zoom-controls" aria-label="Page preview zoom">
      <button id="zoom-out" type="button" aria-label="Zoom out"
              title="Zoom out">−</button>
      <button id="zoom-in" type="button" aria-label="Zoom in"
              title="Zoom in">+</button>
    </div>
    <div class="page">
      <img class="page-image" src="data:image/png;base64,{image_data}"
           alt="PDF page {page_number}">
      <div class="text-layer">{layer_html}</div>
    </div>
  </div>
  <script>
    const root = document.documentElement;
    const zoomOut = document.getElementById("zoom-out");
    const zoomIn = document.getElementById("zoom-in");
    let zoom = 1;
    function setZoom(nextZoom) {{
      zoom = Math.max(0.5, Math.min(3, nextZoom));
      root.style.setProperty("--zoom", zoom);
      zoomOut.disabled = zoom <= 0.5;
      zoomIn.disabled = zoom >= 3;
    }}
    zoomOut.addEventListener("click", () => setZoom(zoom - 0.25));
    zoomIn.addEventListener("click", () => setZoom(zoom + 0.25));
  </script>
</body>
</html>"""
    finally:
        document.close()


def render_page_with_boxes(
    pdf_bytes: bytes,
    page_number: int,
    boxes: list[tuple[int, int, int, int]] | None = None,
    dpi: int = 150,
    include_annotations: bool = True,
) -> bytes:
    """Render ONE page to PNG, drawing each box in `boxes` as a highlight.

    Args:
        pdf_bytes: the raw original PDF (never modified).
        page_number: 1-based.
        boxes: list of (left, top, width, height) in PDF POINTS, like the
            `SearchMatch` fields from src/document_search.py.
        dpi: render sharpness (150 is a good screen size).
        include_annotations: whether to render PDF annotations on the page.

    Returns:
        PNG bytes.
    """
    if pdf_bytes is None or len(pdf_bytes) == 0:
        raise ValueError("No PDF data was given to render.")
    if not isinstance(boxes, list):
        boxes = []

    document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    try:
        if not 1 <= page_number <= document.page_count:
            raise ValueError(
                f"Page {page_number} does not exist - this document has "
                f"{document.page_count} page(s)."
            )
        page = document[page_number - 1]

        for left, top, width, height in boxes:
            rect = pymupdf.Rect(left, top, left + width, top + height)
            # Red with 25% opacity: the word stays readable underneath.
            page.draw_rect(
                rect,
                color=(1, 0, 0),
                fill=(1, 0, 0),
                fill_opacity=0.25,
                width=1.2,
            )

        pixmap = page.get_pixmap(dpi=dpi, annots=include_annotations)
        return pixmap.tobytes("png")
    finally:
        document.close()


def render_page_thumbnail(
    pdf_bytes: bytes, page_number: int, dpi: int = 90, border: bool = False
) -> bytes:
    """Render one page small, optionally with a red border.

    Used by the Edit tab's preview grid: `border=True` marks the card the
    tools act on (a red frame communicates "target page" without text).
    """
    png = render_page_with_boxes(pdf_bytes, page_number, dpi=dpi)
    if not border:
        return png

    from PIL import Image, ImageOps

    image = Image.open(BytesIO(png))
    framed = ImageOps.expand(image, border=4, fill=(230, 30, 30))
    buffer = BytesIO()
    framed.save(buffer, format="PNG")
    return buffer.getvalue()


def render_page_card(
    pdf_bytes: bytes, page_number: int, dpi: int = 100, selected: bool = False
) -> bytes:
    """One GRId card: page thumbnail + white padding + "Page N" label.

    `selected=True` draws a red frame around the whole card - that is the
    visual marker for "the tools act on this page" (no text needed).
    """
    page_png = render_page_with_boxes(pdf_bytes, page_number, dpi=dpi)

    from PIL import Image, ImageDraw

    image = Image.open(BytesIO(page_png))
    padding = 16
    label_height = 22
    card_width = image.width + 2 * padding
    card_height = image.height + 2 * padding + label_height

    card = Image.new("RGB", (card_width, card_height), "white")
    draw = ImageDraw.Draw(card)
    card.paste(image, (padding, padding))
    draw.text(
        (card_width // 2 - 26, image.height + padding + 2),
        f"Page {page_number}",
        fill=(90, 90, 90),
    )
    # EVERY card gets a visible border; the selected one gets a red one.
    if selected:
        draw.rectangle(
            [1, 1, card_width - 2, card_height - 2],
            outline=(230, 30, 30),
            width=4,
        )
    else:
        draw.rectangle(
            [1, 1, card_width - 2, card_height - 2],
            outline=(120, 120, 120),
            width=2,
        )

    buffer = BytesIO()
    card.save(buffer, format="PNG")
    return buffer.getvalue()


def render_document_contact_sheet(
    pdf_bytes: bytes,
    columns: int = 2,
    cell_width_px: int = 300,
    highlight_page: int | None = None,
) -> bytes:
    """ALL pages of the document as ONE grid image ("contact sheet").

    The user sees every page in a single scrollable picture. Each cell is
    numbered, and `highlight_page` (1-based) gets a red frame so the user
    can see which page the editing tools target.

    The original PDF is never modified - this is a pure in-memory render.
    """
    if pdf_bytes is None or len(pdf_bytes) == 0:
        raise ValueError("No PDF data was given to render.")
    if columns < 1:
        raise ValueError("At least one column is needed for the preview.")

    # PIL is imported here so this module stays importable without Pillow.
    from PIL import Image, ImageDraw

    document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    try:
        page_count = document.page_count
        if page_count == 0:
            raise ValueError("This PDF has no pages to preview.")
        rows = -(-page_count // columns)  # ceil division: enough rows for all pages

        # Cell size comes from the FIRST page's aspect ratio; all pages are
        # scaled to the same width (their images may then be shorter/taller).
        zoom = cell_width_px / document[0].rect.width
        cell_height_px = round(document[0].rect.height * zoom)
        gap = 10
        padding = 10
        sheet_width = padding * 2 + columns * cell_width_px + (columns - 1) * gap
        sheet_height = padding * 2 + rows * cell_height_px + (rows - 1) * gap

        sheet = Image.new("RGB", (sheet_width, sheet_height), "white")
        draw = ImageDraw.Draw(sheet)

        for index, page in enumerate(document):
            column = index % columns
            row = index // columns
            page_zoom = cell_width_px / page.rect.width
            pixmap = page.get_pixmap(matrix=pymupdf.Matrix(page_zoom, page_zoom))
            image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)

            x = padding + column * (cell_width_px + gap)
            y = padding + row * (cell_height_px + gap)
            sheet.paste(image, (x, y))
            draw.text((x + 4, y + 2), str(index + 1), fill=(150, 150, 150))

            if highlight_page == index + 1:
                draw.rectangle(
                    [x - 1, y - 1, x + image.width + 1, y + image.height + 1],
                    outline=(230, 30, 30),
                    width=3,
                )

        buffer = BytesIO()
        sheet.save(buffer, format="PNG")
        return buffer.getvalue()
    finally:
        document.close()
