"""Entidade raiz multidocumento do catálogo regulatório (ANAC e outras autoridades)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum


class DocumentType(StrEnum):
    """Tipos de documentos regulatórios suportados pela plataforma."""

    RBAC = "RBAC"
    IS = "IS"
    RESOLUCAO = "RESOLUCAO"
    PORTARIA = "PORTARIA"
    PUBLICACAO_TECNICA = "PUBLICACAO_TECNICA"
    OUTRO = "OUTRO"


class DocumentStatus(StrEnum):
    """Situação geral de um documento no catálogo."""

    ACTIVE = "active"
    ARCHIVED = "archived"
    DRAFT = "draft"
    UNKNOWN = "unknown"


class VersionStatus(StrEnum):
    """Situação normativa oficial de uma versão/emenda específica."""

    CURRENT = "current"
    HISTORICAL = "historical"
    SUPERSEDED = "superseded"
    REVOKED = "revoked"
    PROPOSED = "proposed"
    UNKNOWN = "unknown"


class ValidationPolicy(StrEnum):
    """Políticas configuráveis de validação híbrida por página."""

    STRICT_ALL_PAGES = "strict_all_pages"
    REVIEW_ON_DEMAND = "review_on_demand"
    NATIVE_AUTO_OCR_MANUAL = "native_auto_ocr_manual"


@dataclass(frozen=True, slots=True)
class RegulatoryDocument:
    """Cadastro genérico de um documento regulatório no catálogo centralizado."""

    document_id: str
    authority: str
    document_type: DocumentType
    regulation_number: str
    document_title: str
    document_status: DocumentStatus = DocumentStatus.ACTIVE
    version_status: VersionStatus = VersionStatus.UNKNOWN
    amendment_number: str = ""
    publication_date: date | None = None
    effective_date: date | None = None
    expiration_date: date | None = None
    official_url: str = ""
    source_url: str = ""
    pdf_filename: str = ""
    file_hash: str = ""
    imported_at: datetime | None = None
    last_verified_at: datetime | None = None
    subject_categories: tuple[str, ...] = ()
    document_description: str = ""

    def __post_init__(self) -> None:
        if not self.document_id.strip():
            raise ValueError("document_id não pode ser vazio.")
        if not self.authority.strip():
            raise ValueError("authority não pode ser vazio.")
        if not self.regulation_number.strip():
            raise ValueError("regulation_number não pode ser vazio.")
        if not self.document_title.strip():
            raise ValueError("document_title não pode ser vazio.")
        if self.file_hash and len(self.file_hash.strip()) != 64:
            raise ValueError("file_hash deve conter 64 caracteres hexadecimais (SHA-256).")
