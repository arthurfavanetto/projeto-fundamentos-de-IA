"""Contrato (Port) para extração página a página de PDFs preservando fidelidade textual."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from rbac_assistant.domain.entities.normative_unit import TableCellGeometry


@dataclass(frozen=True, slots=True)
class GlyphAnomaly:
    """Anomalia de glifo/codificação/vetor detectada na camada nativa de uma página PDF."""

    kind: str
    pdf_page_number: int
    text: str
    context: str
    resolved_by_correction_id: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Serializa anomalia de glifo para persistência e auditoria."""
        return {
            "kind": self.kind,
            "pdf_page_number": self.pdf_page_number,
            "text": self.text,
            "context": self.context,
            "resolved_by_correction_id": self.resolved_by_correction_id,
        }


@dataclass(frozen=True, slots=True)
class ExtractedTableData:
    """Tabela ou formulário estruturado extraído de uma página do PDF."""

    pdf_page_number: int
    rows: tuple[tuple[str, ...], ...]
    caption: str = ""
    bbox: tuple[float, float, float, float] | None = None
    cells: tuple[TableCellGeometry, ...] = ()
    header_row_count: int = 1
    table_kind: str = "data_table"
    has_merged_cells: bool = False
    visual_layout_preserved: bool = True
    layout_fidelity_note: str = ""


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
    source_sha256: str = ""
    applied_corrections: tuple[dict[str, object], ...] = field(default_factory=tuple)
    correction_issues: tuple[str, ...] = field(default_factory=tuple)
    glyph_anomalies: tuple[GlyphAnomaly, ...] = field(default_factory=tuple)


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
