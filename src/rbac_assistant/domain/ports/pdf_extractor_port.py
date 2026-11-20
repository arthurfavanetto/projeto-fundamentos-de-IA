"""Contrato (Port) para extração página a página de PDFs preservando fidelidade textual."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True, slots=True)
class ExtractedTableData:
    """Tabela bruta extraída de uma página do PDF."""

    pdf_page_number: int
    rows: tuple[tuple[str, ...], ...]


@dataclass(frozen=True, slots=True)
class ExtractedPageData:
    """Resultado da extração nativa de uma única página física do PDF."""

    pdf_page_number: int
    printed_page_number: str
    original_text: str
    normalized_text: str
    content_hash: str
    page_width: float
    page_height: float
    needs_ocr: bool
    tables: tuple[ExtractedTableData, ...] = field(default_factory=tuple)
    image_count: int = 0
    coordinates_summary: tuple[dict[str, float | str], ...] = field(default_factory=tuple)


class PDFExtractorPort(Protocol):
    """Interface para extratores PDF nativos (implementado via pdfplumber na infraestrutura)."""

    def extract_pages(
        self,
        pdf_path: Path,
        *,
        min_native_chars: int = 50,
        max_pages: int = 1000,
    ) -> list[ExtractedPageData]:
        """Extrai o texto página a página sem normalização destrutiva."""
