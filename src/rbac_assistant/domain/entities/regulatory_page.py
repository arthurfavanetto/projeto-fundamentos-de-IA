"""Entidades de página regulatória, métricas de extração e registros de revisão humana."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from rbac_assistant.domain.entities.normative_unit import ExtractionMethod


def compute_page_content_hash(
    original_text: str,
    tables: Sequence[Sequence[Sequence[str]]] = (),
) -> str:
    """Calcula fingerprint SHA-256 determinístico combinando `original_text` e tabelas.

    Garantias de integridade:
    - Utiliza exclusivamente `original_text` (nunca `normalized_text`) e a estrutura
      exata das tabelas extraídas da página.
    - Preserva Unicode integral (`ensure_ascii=False`), fronteiras de células, ordem de
      tabelas, ordem de linhas, ordem de colunas e células vazias via JSON canônico UTF-8.
    - Independente do `source_hash` binário do PDF.
    """
    canonical_tables = [
        [[str(cell) for cell in row] for row in table_rows]
        for table_rows in tables
    ]
    payload = json.dumps(
        {
            "original_text": original_text,
            "tables": canonical_tables,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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
