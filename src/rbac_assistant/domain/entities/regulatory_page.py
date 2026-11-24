"""Entidades de página regulatória, métricas de extração e registros de revisão humana."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from rbac_assistant.domain.entities.normative_unit import (
    ExtractionMethod,
    NormativeTable,
    effective_table_ownership_status,
)

CANONICAL_PAGE_HASH_SCHEMA_VERSION = "page_canonical_v2"


def _canonical_number(value: float) -> float:
    num = float(value)
    if not math.isfinite(num):
        raise ValueError(f"Coordenada geométrica inválida para hash canônico: {value!r}")
    rounded = round(num, 2)
    return 0.0 if rounded == 0.0 else rounded


def _canonical_bbox(
    bbox: tuple[float, float, float, float] | Sequence[float] | None,
) -> list[float] | None:
    if bbox is None:
        return None
    if len(bbox) != 4:
        raise ValueError(f"bbox inválido para hash canônico: esperado 4 valores, obtido {bbox!r}")
    return [_canonical_number(v) for v in bbox]


def _canonicalize_table_entry(table: object, order_index: int) -> dict[str, Any]:
    """Serializa uma tabela estruturada ou matriz bruta em representação canônica determinística.

    Exclui explicitamente identificadores aleatórios/internos de banco (`table_id`) e timestamps.
    """
    if hasattr(table, "rows") and not isinstance(table, (str, bytes)):
        raw_rows = getattr(table, "rows", ())
        canonical_rows = [[str(cell) for cell in row] for row in raw_rows]
        row_count = len(canonical_rows)
        col_count = len(canonical_rows[0]) if canonical_rows else 0
        header_row_count = int(getattr(table, "header_row_count", 1))
        table_kind = str(getattr(table, "table_kind", "data_table"))
        caption = str(getattr(table, "caption", ""))
        has_merged_cells = bool(getattr(table, "has_merged_cells", False))
        visual_layout_preserved = bool(getattr(table, "visual_layout_preserved", True))
        layout_fidelity_note = str(getattr(table, "layout_fidelity_note", ""))
        bbox = _canonical_bbox(getattr(table, "bbox", None))

        if isinstance(table, NormativeTable):
            ownership_status = effective_table_ownership_status(table).value
            owner_unit_id = table.unit_id if table.unit_id else None
            owner_section_number = str(table.owner_section_number)
        else:
            raw_uid = getattr(table, "owner_unit_id", getattr(table, "unit_id", None))
            owner_unit_id = str(raw_uid) if raw_uid else None
            raw_own = str(getattr(table, "ownership_status", "")).strip().lower()
            ownership_status = raw_own or ("confirmed" if owner_unit_id else "unassigned")
            owner_section_number = str(getattr(table, "owner_section_number", ""))

        raw_cells = getattr(table, "cells", ()) or ()
        canonical_cells = [
            {
                "bbox": _canonical_bbox(getattr(c, "bbox", None)),
                "col_index": int(getattr(c, "col_index", 0)),
                "col_span": int(getattr(c, "col_span", 1)),
                "is_empty": bool(getattr(c, "is_empty", False)),
                "row_index": int(getattr(c, "row_index", 0)),
                "row_span": int(getattr(c, "row_span", 1)),
                "text": str(getattr(c, "text", "")),
            }
            for c in sorted(
                raw_cells,
                key=lambda item: (
                    int(getattr(item, "row_index", 0)),
                    int(getattr(item, "col_index", 0)),
                ),
            )
        ]
    else:
        # Compatibilidade com matrizes brutas `Sequence[Sequence[str]]`
        canonical_rows = [
            [str(cell) for cell in row]
            for row in table  # type: ignore[attr-defined]
        ]
        row_count = len(canonical_rows)
        col_count = len(canonical_rows[0]) if canonical_rows else 0
        header_row_count = 1 if row_count > 0 else 0
        table_kind = "data_table"
        caption = ""
        has_merged_cells = False
        visual_layout_preserved = True
        layout_fidelity_note = ""
        bbox = None
        ownership_status = "unassigned"
        owner_unit_id = None
        owner_section_number = ""
        canonical_cells = []

    hdr_slice_end = max(0, min(header_row_count, row_count))
    return {
        "bbox": bbox,
        "caption": caption,
        "cells": canonical_cells,
        "col_count": col_count,
        "has_merged_cells": has_merged_cells,
        "header_row_count": header_row_count,
        "header_rows": canonical_rows[:hdr_slice_end],
        "layout_fidelity_note": layout_fidelity_note,
        "order_index": order_index,
        "owner_section_number": owner_section_number,
        "owner_unit_id": owner_unit_id,
        "ownership_status": ownership_status,
        "row_count": row_count,
        "rows": canonical_rows,
        "table_kind": table_kind,
        "visual_layout_preserved": visual_layout_preserved,
    }


def build_canonical_page_hash_payload(
    original_text: str,
    tables: Sequence[object] = (),
) -> dict[str, Any]:
    """Constrói payload JSON canônico (`page_canonical_v2`) para texto e estrutura tabular."""
    canonical_tables = [
        _canonicalize_table_entry(table, order_index=idx) for idx, table in enumerate(tables)
    ]
    return {
        "original_text": original_text,
        "schema_version": CANONICAL_PAGE_HASH_SCHEMA_VERSION,
        "table_count": len(canonical_tables),
        "tables": canonical_tables,
    }


def compute_page_content_hash(
    original_text: str,
    tables: Sequence[object] = (),
) -> str:
    """Calcula fingerprint SHA-256 canônico (`page_canonical_v2`) do texto e estrutura tabular.

    Garantias de integridade:
    - Utiliza exclusivamente `original_text` (nunca `normalized_text`), sem normalização genérica.
    - Inclui versão explícita de esquema (`page_canonical_v2`), contagem/ordem visual de tabelas,
      dimensões, células vazias, fronteiras, cabeçalhos, `header_row_count`, `row_span`,
      `col_span`, células mescladas, `table_kind`, legenda, associação proprietária e geometria.
    - Exclui IDs aleatórios de banco, timestamps e metadados instáveis.
    - Serialização JSON canônica UTF-8 (`ensure_ascii=False`, `sort_keys=True`, `allow_nan=False`).
    """
    payload = build_canonical_page_hash_payload(original_text, tables)
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class PageValidationStatus(StrEnum):
    """Ciclo de vida granular de validação individual por página.

    Nota obrigatória: `NATIVE_QUALITY_CHECKED` indica apenas verificação automatizada
    de integridade — nunca equivale a `MANUALLY_APPROVED` ou
    `APPROVED_FOR_LITERAL_TRANSCRIPTION`.
    """

    PENDING_EXTRACTION = "pending_extraction"
    NATIVE_EXTRACTED = "native_extracted"
    NATIVE_QUALITY_CHECKED = "native_quality_checked"
    PENDING_REVIEW = "pending_review"
    MANUALLY_APPROVED = "manually_approved"
    OCR_EXTRACTED = "ocr_extracted"
    PENDING_MANUAL_REVIEW = "pending_manual_review"
    REJECTED = "rejected"
    APPROVED_FOR_LITERAL_TRANSCRIPTION = "approved_for_literal_transcription"


class ReviewDecision(StrEnum):
    """Decisão registrada em uma revisão humana explícita."""

    APPROVE = "approve"
    REJECT = "reject"
    INVALIDATED_BY_REPROCESSING = "invalidated_by_reprocessing"


@dataclass(frozen=True, slots=True)
class QualityMetrics:
    """Métricas objetivas da análise automatizada de qualidade textual de uma página."""

    char_count: int
    line_count: int
    word_count: int
    has_replacement_chars: bool
    has_mojibake_signals: bool
    has_portuguese_accents: bool
    unbalanced_parentheses: bool
    is_empty_or_incomplete: bool
    passed_automated_checks: bool
    issues: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ExtractionMetadata:
    """Metadados técnicos da extração de uma página ou documento."""

    extraction_method: ExtractionMethod
    extraction_timestamp: datetime
    extraction_version: str
    content_hash: str
    quality_metrics: QualityMetrics
    page_width: float = 0.0
    page_height: float = 0.0
    table_count: int = 0
    image_count: int = 0
    applied_corrections: tuple[dict[str, object], ...] = ()
    glyph_anomalies: tuple[dict[str, object], ...] = ()


@dataclass(frozen=True, slots=True)
class ValidationRecord:
    """Registro auditável de revisão humana ou invalidação de uma página."""

    review_id: str
    page_id: str
    reviewer: str
    reviewed_at: datetime
    decision: ReviewDecision
    source_hash: str
    comments: str = ""

    def __post_init__(self) -> None:
        if not self.review_id.strip():
            raise ValueError("review_id não pode ser vazio.")
        if not self.page_id.strip():
            raise ValueError("page_id não pode ser vazio.")
        if not self.reviewer.strip():
            raise ValueError("reviewer não pode ser vazio.")
        if len(self.source_hash.strip()) != 64:
            raise ValueError("source_hash deve conter 64 caracteres hexadecimais.")


@dataclass(frozen=True, slots=True)
class RegulatoryPage:
    """Página individual extraída de uma versão normativa com texto original e auxiliar."""

    page_id: str
    version_id: str
    pdf_page_number: int
    printed_page_number: str
    original_text: str
    normalized_text: str
    extraction_method: ExtractionMethod
    validation_status: PageValidationStatus
    content_hash: str
    extraction_metadata: ExtractionMetadata | None = None
    coordinates_summary: tuple[dict[str, float | str], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.page_id.strip():
            raise ValueError("page_id não pode ser vazio.")
        if not self.version_id.strip():
            raise ValueError("version_id não pode ser vazio.")
        if self.pdf_page_number < 1:
            raise ValueError("pdf_page_number deve ser >= 1.")
        if len(self.content_hash.strip()) != 64:
            raise ValueError("content_hash deve conter 64 caracteres hexadecimais.")

    @property
    def is_ocr(self) -> bool:
        """Indica se o texto da página dependeu de OCR."""
        return self.extraction_method == ExtractionMethod.OCR_FALLBACK

    @property
    def is_approved_for_literal_transcription(self) -> bool:
        """Retorna True somente quando a página foi aprovada para transcrição literal."""
        return self.validation_status == PageValidationStatus.APPROVED_FOR_LITERAL_TRANSCRIPTION
