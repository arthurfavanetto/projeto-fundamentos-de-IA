"""Contrato (Port) opcional para OCR seletivo página a página."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True, slots=True)
class OcrPageResult:
    """Texto extraído via OCR em uma página específica do PDF."""

    pdf_page_number: int
    original_text: str
    normalized_text: str
    content_hash: str
    engine_name: str
    language: str


class OCRPort(Protocol):
    """Interface desacoplada para OCR condicional (nunca obrigatório na imagem base)."""

    def is_available(self) -> bool:
        """Indica se as dependências Python e o binário externo de OCR estão disponíveis."""

    def extract_page_text(
        self,
        pdf_path: Path,
        *,
        pdf_page_number: int,
        language: str = "por",
    ) -> OcrPageResult:
        """Executa OCR exclusivamente na página indicada (indexação 1-based)."""
