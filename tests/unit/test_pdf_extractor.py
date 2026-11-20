"""Testes unitários de extração PDF (pdfplumber), fontes de documento e OCR opcional."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from rbac_assistant.domain.exceptions import InvalidPdfError, OcrUnavailableError, SourceFetchError
from rbac_assistant.infrastructure.pdf.ocr_adapter import TesseractOcrAdapter
from rbac_assistant.infrastructure.pdf.pdfplumber_extractor import (
    PdfPlumberExtractorAdapter,
    detect_printed_page_number,
)
from rbac_assistant.infrastructure.sources.anac_source_adapter import AnacHttpSourceAdapter
from rbac_assistant.infrastructure.sources.local_file_source_adapter import LocalFileSourceAdapter
from tests.fixtures.synthetic_pdf_factory import create_synthetic_rbac_pdf


@pytest.mark.unit
def test_pdfplumber_extracts_native_text_accents_displaced_pages_and_tables(tmp_path: Path) -> None:
    pdf_path = create_synthetic_rbac_pdf(
        tmp_path / "rbac43_synth.pdf",
        regulation_number="43",
        amendment_number="05",
        include_blank_ocr_page=True,
    )

    pages = PdfPlumberExtractorAdapter().extract_pages(pdf_path, min_native_chars=40)
    assert len(pages) == 4

    # Página física 2 possui página impressa deslocada "10/25"
    p2 = pages[1]
    assert p2.pdf_page_number == 2
    assert p2.printed_page_number == "10/25"
    assert p2.needs_ocr is False
    assert "manutenção preventiva, reconstrução ou alteração" in p2.original_text
    assert "aeronavegabilidade brasileiro;" in p2.original_text
    assert p2.normalized_text != ""
    assert len(p2.content_hash) == 64
    assert len(p2.coordinates_summary) > 0

    # Página física 3 possui tabela extraída com 3 linhas e 3 colunas
    p3 = pages[2]
    assert p3.pdf_page_number == 3
    assert p3.printed_page_number == "11/25"
    assert len(p3.tables) == 1
    assert p3.tables[0].rows[0] == ("Item", "Descrição", "Referência")
    assert p3.tables[0].rows[1][1] == "Inspeção periódica"

    # Página física 4 sem camada de texto nativo sinaliza necessidade de OCR
    p4 = pages[3]
    assert p4.pdf_page_number == 4
    assert p4.needs_ocr is True


@pytest.mark.unit
def test_pdfplumber_rejects_malformed_or_oversized_page_count_pdf(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.pdf"
    corrupt.write_bytes(b"%PDF-1.4\nnot really a valid pdf body")
    with pytest.raises(InvalidPdfError):
        PdfPlumberExtractorAdapter().extract_pages(corrupt)

    valid = create_synthetic_rbac_pdf(tmp_path / "valid.pdf")
    with pytest.raises(InvalidPdfError, match="excedendo o limite"):
        PdfPlumberExtractorAdapter().extract_pages(valid, max_pages=2)


@pytest.mark.unit
def test_detect_printed_page_number_supports_common_formats() -> None:
    assert detect_printed_page_number(["linha", "Origem: SAR 14/32"]) == "14/32"
    assert detect_printed_page_number(["cabeçalho", "Página 5 de 28"]) == "5de28"
    assert detect_printed_page_number(["texto sem rodapé numérico"]) == ""


@pytest.mark.unit
def test_local_and_http_source_adapters_validate_pdf_and_limits(tmp_path: Path) -> None:
    pdf_path = create_synthetic_rbac_pdf(tmp_path / "local.pdf")
    fetched = LocalFileSourceAdapter(max_pdf_size_mb=5).fetch(str(pdf_path))
    assert fetched.filename == "local.pdf"
    assert len(fetched.file_hash) == 64
    assert fetched.pdf_bytes.startswith(b"%PDF-")

    bad_file = tmp_path / "not_a_pdf.txt"
    bad_file.write_text("conteúdo texto simples", encoding="utf-8")
    with pytest.raises(InvalidPdfError):
        LocalFileSourceAdapter().fetch(str(bad_file))

    with pytest.raises(SourceFetchError):
        LocalFileSourceAdapter().fetch(str(tmp_path / "inexistente.pdf"))

    # HTTP adapter com mock transport
    pdf_bytes = pdf_path.read_bytes()

    def ok_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=pdf_bytes, headers={"content-type": "application/pdf"})

    http_client = httpx.Client(transport=httpx.MockTransport(ok_handler))
    remote = AnacHttpSourceAdapter(http_client=http_client).fetch(
        "https://pergamum.anac.gov.br/pergamum/vinculos/RBAC43EMD05.pdf"
    )
    assert remote.file_hash == fetched.file_hash
    assert remote.filename == "RBAC43EMD05.pdf"

    def html_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"<html>erro</html>")

    bad_http = httpx.Client(transport=httpx.MockTransport(html_handler))
    with pytest.raises(InvalidPdfError):
        AnacHttpSourceAdapter(http_client=bad_http).fetch("https://example.gov.br/fake.pdf")


@pytest.mark.unit
def test_optional_ocr_adapter_fails_cleanly_when_tesseract_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("shutil.which", lambda _cmd: None)
    adapter = TesseractOcrAdapter()
    assert adapter.is_available() is False
    with pytest.raises(OcrUnavailableError):
        adapter.extract_page_text(tmp_path / "any.pdf", pdf_page_number=1)
