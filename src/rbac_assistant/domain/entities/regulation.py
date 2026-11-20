"""Entidades de domínio para regulamentos e emendas oficiais da ANAC."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum


class VigenciaStatus(StrEnum):
    """Status jurídico de vigência de um regulamento ou emenda da ANAC."""

    VIGENTE = "VIGENTE"
    REVOGADO = "REVOGADO"
    ALTERADO_PARCIALMENTE = "ALTERADO PARCIALMENTE"
    PENDENTE_VERIFICACAO = "PENDING_VERIFICATION"


@dataclass(frozen=True, slots=True)
class DocumentMetadata:
    """Metadados oficiais de rastreabilidade de um documento normativo da ANAC."""

    document_id: str
    regulation_number: str
    regulation_name: str
    document_title: str
    amendment_number: str
    resolution_number: str
    publication_date: date
    effective_date: date
    vigencia_status: VigenciaStatus
    is_current_amendment: bool
    official_url: str
    consulted_at: datetime
    pdf_filename: str
    sha256: str

    def __post_init__(self) -> None:
        if not self.document_id.strip():
            raise ValueError("document_id não pode ser vazio.")
        if not self.regulation_number.strip():
            raise ValueError("regulation_number não pode ser vazio.")
        if not self.amendment_number.strip():
            raise ValueError("amendment_number não pode ser vazio.")
        if len(self.sha256.strip()) != 64:
            raise ValueError("sha256 deve conter exatamente 64 caracteres hexadecimais.")


@dataclass(frozen=True, slots=True)
class Amendment:
    """Representa uma emenda específica de um RBAC da ANAC."""

    amendment_number: str
    resolution_number: str
    publication_date: date
    effective_date: date
    vigencia_status: VigenciaStatus
    is_current: bool


@dataclass(frozen=True, slots=True)
class Regulation:
    """Entidade raiz representando um Regulamento Brasileiro da Aviação Civil (RBAC)."""

    regulation_number: str
    regulation_name: str
    document_title: str
    amendments: tuple[Amendment, ...] = ()

    @property
    def current_amendment(self) -> Amendment | None:
        """Retorna a emenda marcada como vigente atual, se houver."""
        current = [a for a in self.amendments if a.is_current]
        if len(current) > 1:
            raise ValueError(
                f"Múltiplas emendas marcadas como vigentes em {self.regulation_number}."
            )
        return current[0] if current else None
