# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Keep Markdown pictures and raw HTML through the internal HTML round trip."""

import html
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

from docling_core.transforms.serializer.base import (
    BaseDocSerializer,
    SerializationResult,
)
from docling_core.transforms.serializer.common import create_ser_result
from docling_core.transforms.serializer.html import (
    HTMLDocSerializer,
    HTMLParams,
    HTMLPictureSerializer,
    HTMLTextSerializer,
)
from docling_core.types.doc import (
    CodeItem,
    DoclingDocument,
    ImageRef,
    PictureItem,
    TextItem,
)
from docling_core.types.doc.common.constants import DOCUMENT_TOKENS_EXPORT_LABELS
from docling_core.types.doc.common.content_layer import DEFAULT_CONTENT_LAYERS
from typing_extensions import override

from docling.backend.html_backend import HTMLDocumentBackend
from docling.datamodel.backend_options import HTMLBackendOptions
from docling.datamodel.document import InputDocument


@dataclass(frozen=True)
class MarkdownHTMLPayload:
    text: str
    picture_refs: dict[str, ImageRef | None]


class _MarkdownTextSerializer(HTMLTextSerializer):
    raw_html_by_ref: dict[str, str]

    @override
    def serialize(
        self,
        *,
        item: TextItem,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        # Only parser-registered raw HTML bypasses normal code escaping.
        # User text, including strings that resemble markers, is not identity.
        if isinstance(item, CodeItem) and item.self_ref in self.raw_html_by_ref:
            return create_ser_result(
                text=self.raw_html_by_ref[item.self_ref], span_source=item
            )
        return super().serialize(
            item=item, doc_serializer=doc_serializer, doc=doc, **kwargs
        )


class _MarkdownPictureSerializer(HTMLPictureSerializer):
    def __init__(self, reference_prefix: str) -> None:
        self._reference_prefix = reference_prefix

    @override
    def serialize(
        self,
        *,
        item: PictureItem,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        # The HTML backend must see an img even when fetching was disabled or
        # failed. Its source is a document-local key, not a new fetch request.
        reference = html.escape(self._reference_prefix + item.self_ref, quote=True)
        parts = [
            doc_serializer.serialize_captions(item=item, tag="figcaption", **kwargs),
            create_ser_result(text=f'<img src="{reference}">', span_source=item),
        ]
        text = "".join(part.text for part in parts)
        return create_ser_result(text=f"<figure>{text}</figure>", span_source=parts)


class MarkdownHTMLDocumentBackend(HTMLDocumentBackend):
    def __init__(
        self,
        in_doc: InputDocument,
        *,
        path_or_stream: BytesIO | Path,
        options: HTMLBackendOptions | None,
        picture_refs: dict[str, ImageRef | None] | None = None,
    ) -> None:
        # InputDocument constructs a backend to inspect it before conversion;
        # that instance does not yet need the already parsed Markdown pictures.
        super().__init__(in_doc=in_doc, path_or_stream=path_or_stream, options=options)
        self._picture_refs = picture_refs if picture_refs is not None else {}

    @override
    def _create_image_ref(self, src_url: str) -> ImageRef | None:
        if src_url in self._picture_refs:
            return self._picture_refs[src_url]
        # Images in the original HTML retain the normal loader and its guards.
        return super()._create_image_ref(src_url)


def serialize_markdown_for_html(
    doc: DoclingDocument, *, source_hash: str, raw_html_by_ref: dict[str, str]
) -> MarkdownHTMLPayload:
    # Include the source hash so original HTML cannot collide with a reference
    # merely by using the same picture index. The namespace is deterministic.
    prefix = f"data:application/x-docling-md-picture,{source_hash}/"
    picture_refs = {
        prefix + picture.self_ref: picture.image for picture in doc.pictures
    }
    serializer = HTMLDocSerializer(
        doc=doc,
        params=HTMLParams(
            labels=DOCUMENT_TOKENS_EXPORT_LABELS,
            layers=DEFAULT_CONTENT_LAYERS,
        ),
        text_serializer=_MarkdownTextSerializer(raw_html_by_ref=raw_html_by_ref),
        picture_serializer=_MarkdownPictureSerializer(prefix),
    )
    return MarkdownHTMLPayload(serializer.serialize().text, picture_refs)
