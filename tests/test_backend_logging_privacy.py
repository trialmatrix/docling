# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Backend diagnostics must not expose document paths or exception contents."""

import logging
import plistlib
import zipfile
from io import BytesIO
from pathlib import Path

import pytest
from docling_core.types.doc import DoclingDocument, ImageRef, Size

from docling.backend.iwork.numbers_xml import CHART_SHARE_SUFFIX, read_chart_shares
from docling.backend.json.docling_json_backend import DoclingJSONBackend
from docling.backend.latex.utils.encoding import decode_latex_content
from docling.backend.noop_backend import NoOpBackend
from docling.datamodel.base_models import FailureCategory, InputFormat
from docling.datamodel.document import InputDocument, get_input_rejection_cause


def test_chart_share_failures_keep_valid_data_without_logging_entry_names(caplog):
    valid = {"title": "A chart"}
    payload = plistlib.dumps(valid)
    names = [
        f"case-314-{kind}{CHART_SHARE_SUFFIX}" for kind in ("valid", "large", "bad")
    ]
    archive_bytes = BytesIO()
    with zipfile.ZipFile(archive_bytes, "w") as archive:
        archive.writestr(names[0], payload)
        archive.writestr(names[1], b"x" * (len(payload) + 1))
        archive.writestr(names[2], b"case-314-not-a-property-list")

    with (
        caplog.at_level(logging.DEBUG, logger="docling.backend.iwork.numbers_xml"),
        zipfile.ZipFile(archive_bytes) as archive,
    ):
        shares = read_chart_shares(archive, max_file_bytes=len(payload))

    assert shares == {names[0][: -len(CHART_SHARE_SUFFIX)]: valid}
    assert {record.levelno for record in caplog.records} == {
        logging.WARNING,
        logging.DEBUG,
    }
    assert "case-314" not in caplog.text


def test_json_reference_warning_omits_the_document_name(caplog):
    document = DoclingDocument(name="case-314")
    document.add_picture(
        image=ImageRef(
            mimetype="image/png",
            dpi=72,
            size=Size(width=1, height=1),
            uri=Path("local.png"),
        )
    )
    source = InputDocument(
        BytesIO(document.model_dump_json().encode()),
        filename="case-314.json",
        format=InputFormat.JSON_DOCLING,
        backend=DoclingJSONBackend,
    )
    assert source.valid
    assert source._backend.convert().pictures[0].image is None
    assert "ignored 1 image reference" in caplog.text
    assert "case-314" not in caplog.text


@pytest.mark.parametrize("is_directory", [False, True], ids=["missing", "directory"])
def test_latex_read_failures_do_not_log_paths(tmp_path, caplog, is_directory):
    source = tmp_path / "case-314.tex"
    if is_directory:
        source.mkdir()
    assert decode_latex_content(source) == ""
    assert caplog.records
    assert "case-314" not in caplog.text


def test_noop_initialization_does_not_log_the_document_path(tmp_path, caplog):
    source = tmp_path / "case-314.txt"
    source.write_bytes(b"document")
    with caplog.at_level(logging.DEBUG, logger="docling.backend.noop_backend"):
        document = InputDocument(source, format=InputFormat.AUDIO, backend=NoOpBackend)
    assert document.valid
    assert caplog.records
    assert "case-314" not in caplog.text


@pytest.mark.parametrize("unexpected", [False, True], ids=["missing", "runtime"])
def test_input_failure_logs_preserve_the_cause_without_exposing_it(
    tmp_path, caplog, unexpected
):
    source = tmp_path / "case-314.txt"
    error = RuntimeError("case-314 document detail")

    class FailingBackend(NoOpBackend):
        def __init__(self, in_doc, path_or_stream):
            raise error

    if unexpected:
        source.write_bytes(b"document")
    document = InputDocument(
        source,
        format=InputFormat.AUDIO,
        backend=FailingBackend if unexpected else NoOpBackend,
    )
    assert not document.valid
    if unexpected:
        assert get_input_rejection_cause(document) is error
        assert document._rejection.category == FailureCategory.UNKNOWN
    else:
        assert isinstance(get_input_rejection_cause(document), FileNotFoundError)
        assert document._rejection.category == FailureCategory.SOURCE_UNAVAILABLE
    assert "case-314" in document._rejection.message
    assert caplog.records
    assert "case-314" not in caplog.text
    assert all(record.exc_info is None for record in caplog.records)
