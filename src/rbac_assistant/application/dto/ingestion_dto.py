"""Objetos de transferência de dados (DTOs) da ingestão documental e validação híbrida."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

from rbac_assistant.domain.entities.document_version import DocumentVersion
from rbac_assistant.domain.entities.normative_unit import NormativeSection
from rbac_assistant.domain.entities.regulatory_document import (
    DocumentStatus,
    DocumentType,
    RegulatoryDocument,
    ValidationPolicy,
    VersionStatus,
)
from rbac_assistant.domain.entities.regulatory_page import (
    RegulatoryPage,
    ReviewDecision,
    ValidationRecord,
)


@dataclass(frozen=True, slots=True)
class RegisterDocumentDTO:
    """Entrada para cadastro/atualização de um documento regulatório no catálogo."""

    document_id: str
    authority: str
    document_type: DocumentType
    regulation_number: str
    document_title: str
    amendment_number: str = ""
    resolution_number: str = ""
    publication_date: date | None = None
    effective_date: date | None = None
    expiration_date: date | None = None
    document_status: DocumentStatus = DocumentStatus.ACTIVE
    version_status: VersionStatus = VersionStatus.UNKNOWN
    official_url: str = ""
    source_url: str = ""
    pdf_filename: str = ""
    last_verified_at: datetime | None = None
    subject_categories: tuple[str, ...] = ()
    document_description: str = ""


@dataclass(frozen=True, slots=True)
class IngestionRequestDTO:
    """Entrada genérica para ingestão de qualquer documento regulatório (URL ou local)."""

    document: RegisterDocumentDTO
    source_uri: str
    source_kind: str = "local"  # "local" | "http"
    force_reprocess: bool = False
    allow_ocr: bool = False
    validation_policy: ValidationPolicy = ValidationPolicy.REVIEW_ON_DEMAND


@dataclass(frozen=True, slots=True)
class IngestionResultDTO:
    """Resultado consolidado de uma execução de ingestão documental."""

    document: RegulatoryDocument
    version: DocumentVersion
    skipped_idempotent: bool
    pages_extracted: int
    units_created: int
    tables_extracted: int
    ocr_pages: tuple[int, ...] = ()
    invalidated_reviews: tuple[ValidationRecord, ...] = ()
    raw_pdf_path: str = ""
    extracted_snapshot_path: str = ""


@dataclass(frozen=True, slots=True)
class PageReviewRequestDTO:
    """Solicitação explícita de revisão humana de páginas ou seção."""

    document_id: str
    reviewer: str
    decision: ReviewDecision
    confirmed: bool
    version_id: str | None = None
    page_numbers: tuple[int, ...] = ()
    section_number: str | None = None
    comments: str = ""


@dataclass(frozen=True, slots=True)
class PageReviewResultDTO:
    """Resultado auditável de uma operação de revisão humana."""

    document_id: str
    version_id: str
    updated_pages: tuple[RegulatoryPage, ...]
    records: tuple[ValidationRecord, ...]
    exported_review_paths: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class SectionLookupResultDTO:
    """Resultado de consulta determinística a uma seção normativa reconstruída."""

    found: bool
    section: NormativeSection | None = None
    auxiliary_unreviewed_available: bool = False
    message: str = ""
    blocked_by_document_approval: bool = False
    pending_pdf_pages: tuple[int, ...] = ()
    rejected_pdf_pages: tuple[int, ...] = ()
    missing_pdf_pages: tuple[int, ...] = ()
