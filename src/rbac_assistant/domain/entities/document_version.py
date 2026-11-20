"""Entidades de versionamento normativo e histórico de execuções de extração."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

from rbac_assistant.domain.entities.regulatory_document import VersionStatus


@dataclass(frozen=True, slots=True)
class DocumentVersion:
    """Versão/emenda específica de um documento regulatório armazenado no catálogo."""

    version_id: str
    document_id: str
    amendment_number: str
    source_hash: str
    version_status: VersionStatus = VersionStatus.UNKNOWN
    resolution_number: str = ""
    publication_date: date | None = None
    effective_date: date | None = None
    expiration_date: date | None = None
    official_url: str = ""
    source_url: str = ""
    pdf_filename: str = ""
    pdf_storage_path: str = ""
    page_count: int = 0
    imported_at: datetime | None = None
    last_verified_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.version_id.strip():
            raise ValueError("version_id não pode ser vazio.")
        if not self.document_id.strip():
            raise ValueError("document_id não pode ser vazio.")
        if not self.amendment_number.strip():
            raise ValueError("amendment_number não pode ser vazio.")
        if len(self.source_hash.strip()) != 64:
            raise ValueError("source_hash deve conter exatamente 64 caracteres (SHA-256).")
        if self.page_count < 0:
            raise ValueError("page_count não pode ser negativo.")


@dataclass(frozen=True, slots=True)
class ExtractionRun:
    """Registro histórico e auditável de cada tentativa de ingestão/extração."""

    run_id: str
    document_id: str
    version_id: str | None
    started_at: datetime
    finished_at: datetime | None
    status: str  # succeeded | skipped_idempotent | failed
    extraction_version: str
    source_hash: str = ""
    pages_extracted: int = 0
    units_created: int = 0
    error_message: str = ""
