"""Entidades normativas hierárquicas da ANAC e governança de validação por página."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from rbac_assistant.domain.exceptions import UnvalidatedPageBlockedError


class ExtractionMethod(StrEnum):
    """Método utilizado para extrair o conteúdo textual de uma página do PDF."""

    NATIVE_PDF = "native_pdf"
    OCR_FALLBACK = "ocr_fallback"
    OCR_REQUIRED_UNAVAILABLE = "ocr_required_unavailable"


class ValidationStatus(StrEnum):
    """Estado de validação textual individual (compatível com Etapa 1)."""

    VALIDATED = "validated"
    BLOCKED_UNREVIEWED = "blocked_unreviewed"
    PENDING_REVIEW = "pending_review"
    FAILED = "failed"


class NormativeUnitType(StrEnum):
    """Níveis estruturais da hierarquia normativa oficial da ANAC e documentos regulatórios."""

    PREAMBLE = "preamble"
    CHAPTER = "chapter"
    SUBPART = "subpart"
    SECTION = "section"
    SUBSECTION = "subsection"
    PARAGRAPH = "paragraph"
    INCISO = "inciso"
    ALINEA = "alinea"
    SUBITEM = "subitem"
    APPENDIX = "appendix"
    TABLE = "table"
    NOTE = "note"


class StructureStatus(StrEnum):
    """Confiança estrutural na delimitação automática de uma unidade normativa."""

    PARSED = "parsed"
    PENDING_VALIDATION = "pending_validation"


@dataclass(frozen=True, slots=True)
class PageExtractionMetadata:
    """Metadados individuais por página garantindo a política `block_until_reviewed`."""

    pdf_page_number: int
    printed_page_number: str
    extraction_method: ExtractionMethod
    validation_status: ValidationStatus
    integrity_checks_passed: bool
    text_sha256: str = ""
    reviewed_by: str | None = None
    reviewed_at: datetime | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if self.pdf_page_number < 1:
            raise ValueError("pdf_page_number deve ser >= 1 (indexação 1-based).")
        if not self.printed_page_number.strip():
            raise ValueError("printed_page_number não pode ser vazio.")

        if self.validation_status == ValidationStatus.VALIDATED:
            if not self.integrity_checks_passed:
                raise ValueError(
                    "Página não pode ter status 'validated' sem aprovação nas verificações "
                    "de integridade textual."
                )
            if self.extraction_method == ExtractionMethod.OCR_FALLBACK and (
                not self.reviewed_by or self.reviewed_at is None
            ):
                raise ValueError(
                    "Política block_until_reviewed: página extraída via OCR exige revisão "
                    "manual (reviewed_by e reviewed_at) antes de receber status 'validated'."
                )

    @classmethod
    def create_ocr_page(
        cls,
        *,
        pdf_page_number: int,
        printed_page_number: str,
        integrity_checks_passed: bool = False,
        text_sha256: str = "",
        notes: str = "",
    ) -> PageExtractionMetadata:
        """Cria metadados de página OCR bloqueada por padrão até revisão manual."""
        return cls(
            pdf_page_number=pdf_page_number,
            printed_page_number=printed_page_number,
            extraction_method=ExtractionMethod.OCR_FALLBACK,
            validation_status=ValidationStatus.BLOCKED_UNREVIEWED,
            integrity_checks_passed=integrity_checks_passed,
            text_sha256=text_sha256,
            notes=notes,
        )

    @property
    def is_literal_transcription_allowed(self) -> bool:
        """Indica se a página está apta para transcrição normativa literal."""
        return (
            self.validation_status == ValidationStatus.VALIDATED
            and self.integrity_checks_passed
        )


@dataclass(frozen=True, slots=True)
class NormativeTable:
    """Tabela normativa extraída preservando linhas, colunas e página de origem."""

    table_id: str
    version_id: str
    pdf_page_number: int
    rows: tuple[tuple[str, ...], ...]
    unit_id: str | None = None
    caption: str = ""


@dataclass(frozen=True, slots=True)
class NormativeReference:
    """Referência cruzada identificada no texto normativo (ex.: RBAC 145, seção 43.9)."""

    reference_id: str
    unit_id: str
    target_regulation: str
    target_section: str = ""
    raw_citation: str = ""


@dataclass(frozen=True, slots=True)
class NormativeUnit:
    """Unidade normativa estruturada de um regulamento (Seção, Parágrafo, Inciso, Apêndice)."""

    document_id: str
    regulation_number: str
    amendment_number: str
    unit_type: NormativeUnitType
    section_number: str
    title: str
    exact_text: str
    pages: tuple[PageExtractionMetadata, ...]
    subpart: str = ""
    subsection_number: str = ""
    paragraph_number: str = ""
    item_number: str = ""
    content_hash: str = ""
    unit_id: str = ""
    version_id: str = ""
    parent_unit_id: str | None = None
    normalized_text: str = ""
    start_page: int = 1
    end_page: int = 1
    order_index: int = 0
    structure_status: StructureStatus = StructureStatus.PARSED
    validation_status: str = "pending_review"

    def __post_init__(self) -> None:
        if not self.section_number.strip():
            raise ValueError("section_number não pode ser vazio.")
        if not self.pages:
            raise ValueError("NormativeUnit deve referenciar ao menos uma página de origem.")

    @property
    def unvalidated_pages(self) -> tuple[PageExtractionMetadata, ...]:
        """Retorna todas as páginas abrangidas pela seção que ainda não estão validadas."""
        return tuple(page for page in self.pages if not page.is_literal_transcription_allowed)

    @property
    def can_guarantee_literal_transcription(self) -> bool:
        """Retorna True somente se todas as páginas abrangidas estiverem validadas."""
        return (
            self.structure_status == StructureStatus.PARSED
            and len(self.unvalidated_pages) == 0
        )

    @property
    def literal_transcription_warning(self) -> str | None:
        """Gera aviso explícito quando a seção abranger páginas não validadas ou bloqueadas."""
        if self.structure_status != StructureStatus.PARSED:
            return (
                f"AVISO DE ESTRUTURA: A delimitação automática da seção {self.section_number} "
                "está marcada como pendente de validação. O sistema não garante a transcrição "
                "literal completa desta seção."
            )
        unvalidated = self.unvalidated_pages
        if not unvalidated:
            return None

        details = ", ".join(
            f"PDF p.{p.pdf_page_number} (impressa {p.printed_page_number}, "
            f"{p.extraction_method.value}/{p.validation_status.value})"
            for p in unvalidated
        )
        return (
            f"AVISO DE FIDELIDADE: A seção {self.section_number} abrange página(s) ainda "
            f"não validadas [{details}]. Sob a política block_until_reviewed, o sistema "
            "não pode garantir a transcrição literal completa desta seção até revisão manual."
        )

    def require_literal_transcription(self) -> str:
        """Retorna o texto literal exato ou bloqueia se houver páginas não validadas."""
        unvalidated = self.unvalidated_pages
        if unvalidated:
            blocked_pages = tuple(p.pdf_page_number for p in unvalidated)
            raise UnvalidatedPageBlockedError(self.section_number, blocked_pages)
        return self.exact_text


@dataclass(frozen=True, slots=True)
class NormativeSection:
    """Seção normativa reconstruída em ordem (incluindo seções que atravessam páginas)."""

    document_id: str
    version_id: str
    authority: str
    document_type: str
    regulation_number: str
    amendment_number: str
    version_status: str
    section_number: str
    title: str
    original_text: str
    normalized_text: str
    start_page: int
    end_page: int
    pdf_pages: tuple[int, ...]
    printed_pages: tuple[str, ...]
    validation_status: str
    structure_status: StructureStatus
    literal_transcription_allowed: bool
    pending_pdf_pages: tuple[int, ...] = ()
    ocr_blocked_pdf_pages: tuple[int, ...] = ()
    warning_message: str | None = None
    tables: tuple[NormativeTable, ...] = ()
    references: tuple[NormativeReference, ...] = ()
