"""Entidades de citação e rastreabilidade de dupla paginação (PDF vs. impressa ANAC)."""

from __future__ import annotations

from dataclasses import dataclass

from rbac_assistant.domain.entities.normative_unit import ExtractionMethod, ValidationStatus
from rbac_assistant.domain.entities.regulation import VigenciaStatus


@dataclass(frozen=True, slots=True)
class PageReference:
    """Referência de página contendo numeração física do PDF e numeração impressa da ANAC."""

    pdf_page_number: int
    printed_page_number: str
    extraction_method: ExtractionMethod
    validation_status: ValidationStatus

    def __post_init__(self) -> None:
        if self.pdf_page_number < 1:
            raise ValueError("pdf_page_number deve ser >= 1.")
        if not self.printed_page_number.strip():
            raise ValueError("printed_page_number não pode ser vazio.")


@dataclass(frozen=True, slots=True)
class Citation:
    """Fundamentação completa e rastreável de um dispositivo normativo recuperado."""

    regulation_number: str
    regulation_name: str
    amendment_number: str
    vigencia_status: VigenciaStatus
    section_number: str
    pages: tuple[PageReference, ...]
    official_url: str
    paragraph_number: str = ""
    item_number: str = ""

    @property
    def has_unvalidated_pages(self) -> bool:
        """Indica se alguma página citada não está validada para transcrição literal."""
        return any(page.validation_status != ValidationStatus.VALIDATED for page in self.pages)
