# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Test dots.ocr / dots.mocr JSON parsing in VLM pipeline."""

import json
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from docling_core.types.doc import DocItemLabel, DoclingDocument, Size
from PIL import Image as PILImage
from pydantic import AnyUrl

from docling.datamodel.base_models import ConversionStatus, InputFormat, Page
from docling.datamodel.pipeline_options import VlmConvertOptions, VlmPipelineOptions
from docling.datamodel.pipeline_options_vlm_model import DotsBboxFrame, ResponseFormat
from docling.datamodel.stage_model_specs import VlmModelSpec
from docling.datamodel.vlm_engine_options import ApiVlmEngineOptions, VlmEngineType
from docling.datamodel.vlm_prompts import DOTS_LAYOUT_PROMPT
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline
from docling.utils.dots_utils import parse_dots_json
from tests.fakes.http_service import FakeService
from tests.fakes.openai_compatible import FakeOpenAiApi


def get_dots_test_paths():
    """Get all dots JSON test files."""
    directory = Path("./tests/data/json_dots/sources/")
    return sorted(directory.glob("*.json"))


def test_dots_simple_parsing():
    """Test dots JSON parsing produces expected document structure."""
    path = Path("./tests/data/json_dots/sources/dots_simple.json")
    content = path.read_text()
    source = path.with_suffix(".source.txt").read_text()

    doc = parse_dots_json(
        content=content,
        original_page_size=Size(width=612, height=792),
        page_no=1,
        filename="dots_simple.json",
    )

    assert isinstance(doc, DoclingDocument)
    assert len(doc.texts) > 0, "Should have text elements"

    labels = [
        t.label.value if hasattr(t.label, "value") else str(t.label) for t in doc.texts
    ]
    assert "title" in labels, "Should have a title element"
    assert "section_header" in labels, "Should have section headers"
    assert "caption" in labels, "Should have captions"
    assert "footnote" in labels, "Should have footnotes"

    assert "tests/data/pdf/2206.01062.pdf, page 1" in source
    assert any("DocLayNet" in (t.text or "") for t in doc.texts)
    assert len(doc.pictures) > 0, "Should have picture elements"

    for item in doc.texts:
        assert len(item.prov) > 0, f"Text item should have provenance: {item.text[:30]}"
        bbox = item.prov[0].bbox
        assert bbox is not None, f"Should have bbox: {item.text[:30]}"
        assert bbox.l >= 0 and bbox.t >= 0, "Bbox coords should be non-negative"


def test_dots_list_parsing():
    """Test dots JSON parsing handles real list-item predictions."""
    path = Path("./tests/data/json_dots/sources/dots_list.json")
    content = path.read_text()
    source = path.with_suffix(".source.txt").read_text()

    doc = parse_dots_json(
        content=content,
        original_page_size=Size(width=612, height=792),
        page_no=1,
        filename=path.name,
    )

    labels = [
        t.label.value if hasattr(t.label, "value") else str(t.label) for t in doc.texts
    ]
    list_items = [item for item in doc.texts if item.label == DocItemLabel.LIST_ITEM]

    assert "tests/data/pdf/multi_page.pdf, page 1" in source
    assert "list_item" in labels, "Should have list items"
    assert len(list_items) == 2
    assert "IBM MT/ST" in list_items[0].text
    assert "Microsoft Word" in list_items[1].text


def test_dots_model_image_size_rescaling():
    """Test that model_image_size rescales bboxes correctly."""
    content = '[{"bbox": [0, 0, 560, 560], "category": "Text", "text": "hello"}]'

    doc = parse_dots_json(
        content=content,
        original_page_size=Size(width=612, height=792),
        page_no=1,
        filename="test.json",
        model_image_size=Size(width=560, height=560),
    )

    assert len(doc.texts) == 1
    bbox = doc.texts[0].prov[0].bbox
    assert abs(bbox.r - 612) < 1, f"Right edge should map to page width, got {bbox.r}"
    assert abs(bbox.b - 792) < 1, f"Bottom edge should map to page height, got {bbox.b}"


def test_dots_empty_content():
    """Test that empty/whitespace content returns empty doc."""
    for content in ["", "   ", "\n"]:
        doc = parse_dots_json(
            content=content,
            original_page_size=Size(width=612, height=792),
            page_no=1,
            filename="empty.json",
        )
        assert isinstance(doc, DoclingDocument)
        assert len(doc.texts) == 0


def test_dots_malformed_json():
    """Invalid JSON is reported instead of silently yielding an empty page."""
    with pytest.raises(ValueError):
        parse_dots_json(
            content="this is not json at all",
            original_page_size=Size(width=612, height=792),
            page_no=1,
            filename="bad.json",
        )


def test_dots_truncated_json():
    """Test that truncated JSON (common in model output) is recovered."""
    content = '[{"bbox": [0, 0, 100, 100], "category": "Text", "text": "hello"}, {"bbox": [0, 100, 200, 200], "category": "Tex'
    doc = parse_dots_json(
        content=content,
        original_page_size=Size(width=612, height=792),
        page_no=1,
        filename="truncated.json",
    )
    assert len(doc.texts) >= 1


def test_dots_bad_bbox_elements():
    """Test that elements with invalid bbox are skipped."""
    content = (
        "["
        '{"bbox": "not a list", "category": "Text", "text": "bad"},'
        '{"bbox": [0, 0], "category": "Text", "text": "short"},'
        '{"bbox": [0, 0, 100, 100], "category": "Text", "text": "good"}'
        "]"
    )
    doc = parse_dots_json(
        content=content,
        original_page_size=Size(width=612, height=792),
        page_no=1,
        filename="bad_bbox.json",
    )
    assert len(doc.texts) == 1
    assert doc.texts[0].text == "good"


def test_dots_non_dict_elements():
    """Test that non-dict elements in array are skipped."""
    content = '[42, "string", {"bbox": [0, 0, 100, 100], "category": "Text", "text": "valid"}]'
    doc = parse_dots_json(
        content=content,
        original_page_size=Size(width=612, height=792),
        page_no=1,
        filename="mixed.json",
    )
    assert len(doc.texts) == 1


def test_dots_all_files_parse():
    """Ensure all dots test files parse without errors."""
    for path in get_dots_test_paths():
        content = path.read_text()
        doc = parse_dots_json(
            content=content,
            original_page_size=Size(width=612, height=792),
            page_no=1,
            filename=path.name,
        )
        assert isinstance(doc, DoclingDocument), f"Failed to parse {path.name}"
        assert len(doc.texts) + len(doc.tables) + len(doc.pictures) > 0, (
            f"No elements parsed from {path.name}"
        )


@pytest.fixture
def api() -> Iterator[FakeOpenAiApi]:
    service = FakeService()
    fake = FakeOpenAiApi()
    service.include(fake.router)
    service.start()
    fake.service = service
    try:
        yield fake
    finally:
        service.stop()


def _convert_first_page(api: FakeOpenAiApi, completion: str):
    api.completion = completion
    options = VlmPipelineOptions(
        enable_remote_services=True,
        # The dots_ocr preset ships DOTS_LAYOUT_PROMPT and ResponseFormat.DOTS_JSON.
        vlm_options=VlmConvertOptions.from_preset(
            "dots_ocr",
            engine_options=ApiVlmEngineOptions(
                engine_type=VlmEngineType.API,
                url=AnyUrl(f"{api.service.base_url}/v1/chat/completions"),
            ),
        ),
    )
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=VlmPipeline, pipeline_options=options
            )
        }
    )
    return converter.convert(
        Path("./tests/data/pdf/sources/2206.01062.pdf"),
        page_range=(1, 1),
        raises_on_error=False,
    )


def test_hosted_model_object_reply_is_converted(api: FakeOpenAiApi):
    """A hosted model following DOTS_LAYOUT_PROMPT's "single JSON object"
    instruction must not produce an empty page reported as SUCCESS."""
    reply = json.dumps(
        {
            "layout": [
                {"bbox": [124, 74, 150, 88], "category": "Page-header", "text": "314"},
                {"bbox": [60, 100, 440, 180], "category": "Text", "text": "Body"},
            ]
        },
        indent=2,
    )
    result = _convert_first_page(api, reply)
    assert result.status == ConversionStatus.SUCCESS
    assert [item.text for item in result.document.texts] == ["314", "Body"]


def test_unparseable_reply_is_reported(api: FakeOpenAiApi):
    """An unparseable DOTS reply is reported, not converted into an empty page
    with SUCCESS status."""
    result = _convert_first_page(api, "Sorry, I cannot read this page.")
    assert result.status == ConversionStatus.PARTIAL_SUCCESS
    assert [error.page_no for error in result.errors] == [1]
    assert result.errors[0].error_message.startswith("Invalid dots response:")


# A DP-Bench page (439.37 x 666.14 pt) rendered at 300 dpi is sent as a
# 1831 x 2776 px image. Qwen2-VL smart_resize brings that to 1260 x 1932 px,
# the frame dots.ocr answers in; a model without that resize, such as a hosted
# OpenAI-compatible VLM, answers in the 1831 px frame it received.
_DP_BENCH_PAGE_SIZE = Size(width=439.37, height=666.14)
_DP_BENCH_SCALE = 300 / 72
_DP_BENCH_IMAGE_SIZE = (1831, 2776)
_DP_BENCH_HEADER = (
    '[{"bbox": [1477, 90, 1620, 140], "category": "Page-header", "text": "YARROW"}]'
)


def _api_engine() -> ApiVlmEngineOptions:
    return ApiVlmEngineOptions(
        engine_type=VlmEngineType.API_OPENAI,
        url="http://localhost:8000/v1/chat/completions",
    )


def _dots_header_bbox(vlm_options: VlmConvertOptions):
    pipeline = VlmPipeline.__new__(VlmPipeline)
    pipeline.pipeline_options = VlmPipelineOptions(vlm_options=vlm_options)
    page = Page(page_no=1, size=_DP_BENCH_PAGE_SIZE)
    page._image_cache[_DP_BENCH_SCALE] = PILImage.new("RGB", _DP_BENCH_IMAGE_SIZE)
    conv_res = MagicMock()
    conv_res.input.file.name = "dp_bench.pdf"

    doc = pipeline._dots_page_document(conv_res, page, _DP_BENCH_HEADER, None)

    assert len(doc.texts) == 1
    return doc.texts[0].prov[0].bbox


@pytest.mark.parametrize("preset_id", ["dots_ocr", "dots_mocr"])
def test_dots_presets_rescale_bboxes_from_qwen2vl_frame(preset_id: str):
    """dots.ocr keeps its smart_resize frame, also when served through an API."""
    vlm_options = VlmConvertOptions.from_preset(
        preset_id, engine_options=_api_engine(), scale=_DP_BENCH_SCALE
    )
    assert vlm_options.model_spec.dots_bbox_frame == DotsBboxFrame.QWEN2VL

    bbox = _dots_header_bbox(vlm_options)

    assert bbox.l == pytest.approx(1477 * 439.37 / 1260)
    assert bbox.r == pytest.approx(1620 * 439.37 / 1260)
    assert bbox.b == pytest.approx(140 * 666.14 / 1932)


def test_dots_input_image_frame_rescales_bboxes_from_sent_image():
    """A model answering in the pixels it received lands on the page."""
    vlm_options = VlmConvertOptions(
        model_spec=VlmModelSpec(
            name="hosted-vlm",
            default_repo_id="hosted-vlm",
            prompt=DOTS_LAYOUT_PROMPT,
            response_format=ResponseFormat.DOTS_JSON,
            dots_bbox_frame=DotsBboxFrame.INPUT_IMAGE,
        ),
        engine_options=_api_engine(),
        scale=_DP_BENCH_SCALE,
    )

    bbox = _dots_header_bbox(vlm_options)

    assert bbox.l == pytest.approx(1477 * 439.37 / 1831)
    assert bbox.r == pytest.approx(1620 * 439.37 / 1831)
    assert bbox.b == pytest.approx(140 * 666.14 / 2776)
    assert bbox.r < _DP_BENCH_PAGE_SIZE.width
