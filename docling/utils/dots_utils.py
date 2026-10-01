# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Utilities for parsing dots.ocr / dots.mocr JSON layout format.

dots.ocr (3B, Qwen2.5-VL based) and dots.mocr produce a JSON array of
layout elements::

    [{"bbox": [x1, y1, x2, y2], "category": "Label", "text": "content"}, ...]

Bboxes are pixel coordinates relative to the model input resolution.
If ``model_image_size`` is provided the coords are rescaled to the
original page coordinate space.

11 categories: Caption, Footnote, Formula, List-item, Page-footer,
Page-header, Picture, Section-header, Table, Text, Title.

Hosted general-purpose VLMs given the same layout prompt, which asks for
"a single JSON object", usually wrap that array in an object instead::

    {"layout": [{"bbox": [x1, y1, x2, y2], "category": "Label", ...}, ...]}

Both forms are accepted.

Tables arrive as HTML ``<table>``; formulas as LaTeX; Pictures have no
``text`` field.  The model sometimes truncates output; a caller that has
already reported the reply incomplete (it hit the token limit or the
provider's content filter) may have the parser recover everything up to the
last complete element.  Otherwise a truncated reply is not recovered,
because the elements after the cut would be lost without any sign.  A
response that contains no decodable element list raises ``ValueError`` so
callers can report it instead of silently producing an empty page.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from docling_core.types.doc import (
    BoundingBox,
    CoordOrigin,
    DocItemLabel,
    DoclingDocument,
    DocumentOrigin,
    ImageRef,
    ProvenanceItem,
    Size,
)
from PIL import Image as PILImage

from docling.utils.chandra_utils import _parse_table_html

_log = logging.getLogger(__name__)

# Mapping from dots.ocr/dots.mocr category strings to DocItemLabel.
_LABEL_MAP: dict[str, DocItemLabel] = {
    "Text": DocItemLabel.TEXT,
    "Title": DocItemLabel.TITLE,
    "Section-header": DocItemLabel.SECTION_HEADER,
    "Table": DocItemLabel.TABLE,
    "Picture": DocItemLabel.PICTURE,
    "Caption": DocItemLabel.CAPTION,
    "Footnote": DocItemLabel.FOOTNOTE,
    "Page-header": DocItemLabel.PAGE_HEADER,
    "Page-footer": DocItemLabel.PAGE_FOOTER,
    "List-item": DocItemLabel.LIST_ITEM,
    "Formula": DocItemLabel.FORMULA,
}


def _close_truncated(raw: str) -> str | None:
    """Cut a truncated JSON document after its last complete element and close it.

    Scans *raw* (which must start with ``[`` or ``{``) while tracking string
    literals, and remembers the last position where an object closed directly
    inside an array, i.e. the end of the last complete layout element.  The
    text is cut there and the still-open brackets are closed in order.

    Returns ``None`` if no complete element inside an array was found.
    """
    stack: list[str] = []
    in_string = False
    escaped = False
    cut: tuple[int, list[str]] | None = None
    for pos, char in enumerate(raw):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char in "[{":
            stack.append(char)
        elif char in "]}":
            if not stack:
                break
            stack.pop()
            if not stack:
                # The top-level value is complete; nothing is truncated.
                return None
            if char == "}" and stack[-1] == "[":
                cut = (pos, list(stack))
    if cut is None:
        return None
    pos, open_brackets = cut
    closers = "".join("]" if b == "[" else "}" for b in reversed(open_brackets))
    return raw[: pos + 1] + closers


def _extract_elements(value: Any) -> list[Any] | None:
    """Return the list of layout elements held by a decoded dots response.

    Accepts the native dots.ocr form, a bare array of element objects, and a
    single JSON object wrapping that array under one key (for example
    ``{"layout": [...]}``), which is what hosted models produce for the
    shipped prompt's "single JSON object" instruction.  An object is accepted
    only if exactly one of its values is a non-empty list of objects (or its
    only list value is empty, for a blank page), so the choice is never
    ambiguous.  A lone element object (with a ``bbox``) is treated as a
    one-element list.
    """
    if isinstance(value, list):
        return value
    if not isinstance(value, dict):
        return None
    candidates = [
        item
        for item in value.values()
        if isinstance(item, list) and all(isinstance(elem, dict) for elem in item)
    ]
    if len(candidates) > 1:
        # An empty side list (e.g. ``"warnings": []``) does not compete with
        # the element list.
        candidates = [item for item in candidates if item]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates and "bbox" in value:
        return [value]
    return None


def _load_dots_elements(raw: str, *, recover_truncated: bool = True) -> list[Any]:
    """Decode the layout elements from a raw dots response.

    Leading prose and trailing text (such as a closing Markdown code fence)
    around the JSON value are ignored.  With *recover_truncated*, output
    truncated mid-element is recovered up to the last complete element;
    without it, truncated output is an error.

    Raises:
        ValueError: if no JSON array or object is found, the JSON cannot be
            decoded (or, with *recover_truncated*, recovered), or it does not
            contain a layout element list.
    """
    starts = [idx for idx in (raw.find("["), raw.find("{")) if idx != -1]
    if not starts:
        raise ValueError("no JSON array or object found in dots response")
    text = raw[min(starts) :]

    decoder = json.JSONDecoder()
    try:
        value, _ = decoder.raw_decode(text)
    except json.JSONDecodeError as exc:
        repaired = _close_truncated(text)
        if repaired is None:
            raise ValueError(f"malformed dots JSON: {exc}") from exc
        if not recover_truncated:
            raise ValueError(
                "truncated dots JSON: the reply ends inside its element list "
                f"although it was not reported incomplete ({exc})"
            ) from exc
        try:
            value = json.loads(repaired)
        except json.JSONDecodeError as repair_exc:
            raise ValueError(f"malformed dots JSON: {exc}") from repair_exc
        _log.warning("Recovered truncated dots JSON up to the last complete element")

    elements = _extract_elements(value)
    if elements is None:
        raise ValueError(
            "dots JSON is neither an array of layout elements nor an object "
            f"holding exactly one such array (got {type(value).__name__})"
        )
    return elements


def parse_dots_json(
    content: str,
    original_page_size: Size,
    page_no: int,
    filename: str = "file",
    page_image: PILImage.Image | None = None,
    model_image_size: Size | None = None,
    *,
    recover_truncated: bool = True,
) -> DoclingDocument:
    """Parse dots.ocr / dots.mocr JSON output into a DoclingDocument.

    Args:
        content: Raw model output: a JSON array of element dicts, or a JSON
            object wrapping that array under a single key.
        original_page_size: Physical page dimensions (points).
        page_no: Page number (1-based).
        filename: Source filename.
        page_image: Optional PIL image of the page.
        model_image_size: If provided, bbox pixel coords are rescaled from
            this resolution to *original_page_size*.
        recover_truncated: Keep the complete elements of a reply truncated
            mid-element.  Pass ``False`` when the reply was reported complete,
            so a cut that would silently drop elements is an error instead.

    Returns:
        DoclingDocument populated with parsed elements.  Empty or
        whitespace-only content yields a document with an empty page.

    Raises:
        ValueError: if non-empty content holds no decodable layout element
            list (see :func:`_load_dots_elements`).
    """
    origin = DocumentOrigin(
        filename=filename,
        mimetype="application/json",
        binary_hash=0,
    )
    doc = DoclingDocument(name=filename.rsplit(".", 1)[0], origin=origin)

    pg_width = original_page_size.width
    pg_height = original_page_size.height

    # Compute rescaling factors
    if model_image_size is not None:
        scale_x = pg_width / model_image_size.width
        scale_y = pg_height / model_image_size.height
    else:
        # No rescaling — assume pixel coords already match page coords
        scale_x = 1.0
        scale_y = 1.0

    image_dpi = 72
    if page_image is not None:
        image_dpi = int(72 * page_image.width / pg_width)

    doc.add_page(
        page_no=page_no,
        size=Size(width=pg_width, height=pg_height),
        image=ImageRef.from_pil(image=page_image, dpi=image_dpi)
        if page_image
        else None,
    )

    if not content or not content.strip():
        return doc

    elements = _load_dots_elements(content, recover_truncated=recover_truncated)

    current_list_group = None

    for elem in elements:
        if not isinstance(elem, dict):
            continue

        category = elem.get("category", "")
        raw_bbox = elem.get("bbox")
        text = elem.get("text", "")

        if not raw_bbox or not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
            continue

        try:
            x1, y1, x2, y2 = (float(v) for v in raw_bbox)
        except (ValueError, TypeError):
            continue

        bbox = BoundingBox(
            l=x1 * scale_x,
            t=y1 * scale_y,
            r=x2 * scale_x,
            b=y2 * scale_y,
            coord_origin=CoordOrigin.TOPLEFT,
        )
        prov = ProvenanceItem(page_no=page_no, bbox=bbox, charspan=[0, 0])

        doc_label = _LABEL_MAP.get(category, DocItemLabel.TEXT)

        if category == "Table":
            current_list_group = None
            table_data = _parse_table_html(text)
            doc.add_table(data=table_data, prov=prov)
        elif category == "Picture":
            current_list_group = None
            doc.add_picture(prov=prov)
        elif category == "Title":
            current_list_group = None
            doc.add_title(text=text, prov=prov)
        elif category == "Section-header":
            current_list_group = None
            doc.add_heading(text=text, prov=prov)
        elif category == "List-item":
            if current_list_group is None:
                current_list_group = doc.add_list_group()
            doc.add_list_item(text=text, parent=current_list_group, prov=prov)
        else:
            current_list_group = None
            doc.add_text(label=doc_label, text=text, prov=prov)

    return doc
