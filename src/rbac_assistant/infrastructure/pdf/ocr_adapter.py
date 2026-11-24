"""Adaptador OCR opcional e desacoplado (pypdfium2 + pytesseract + Tesseract OCR)."""

from __future__ import annotations

import importlib
import shutil
from pathlib import Path

from rbac_assistant.domain.entities.regulatory_page import compute_page_content_hash
from rbac_assistant.domain.exceptions import OcrUnavailableError
from rbac_assistant.domain.ports.ocr_port import OcrPageResult
from rbac_assistant.infrastructure.pdf.pdfplumber_extractor import normalize_auxiliary_text


class TesseractOcrAdapter:
    """Implementação opcional de `OCRPort` — não exige instalação na imagem Docker base."""

    def __init__(self, *, dpi: int = 300) -> None:
        self._dpi = dpi

    def is_available(self) -> bool:
        """Retorna True somente se `pypdfium2`, `pytesseract` e o binário `tesseract` existirem."""
        if shutil.which("tesseract") is None:
            return False
        try:
            importlib.import_module("pypdfium2")
            importlib.import_module("pytesseract")
        except ImportError:
            return False
        return True

    def extract_page_text(
        self,
        pdf_path: Path,
        *,
        pdf_page_number: int,
        language: str = "por",
    ) -> OcrPageResult:
        if not self.is_available():
            raise OcrUnavailableError(
                "OCR opcional indisponível: instale o extra '[ocr]' (pypdfium2, pytesseract) "
                "e o executável Tesseract OCR com pacote de idioma português ('por')."
            )

        pdfium = importlib.import_module("pypdfium2")
        pytesseract = importlib.import_module("pytesseract")

        document = pdfium.PdfDocument(str(pdf_path))
        try:
            page = document[pdf_page_number - 1]
            bitmap = page.render(scale=self._dpi / 72.0)
            pil_image = bitmap.to_pil()
            original_text = str(pytesseract.image_to_string(pil_image, lang=language) or "")
        finally:
            document.close()

        normalized_text = normalize_auxiliary_text(original_text)
        content_hash = compute_page_content_hash(original_text)
        return OcrPageResult(
            pdf_page_number=pdf_page_number,
            original_text=original_text,
            normalized_text=normalized_text,
            content_hash=content_hash,
            engine_name="tesseract",
            language=language,
        )
