"""Caso de uso para reconstrução multi-página de seções normativas com validação híbrida."""

from __future__ import annotations

from rbac_assistant.application.dto.ingestion_dto import SectionLookupResultDTO
from rbac_assistant.application.services.catalog_service import CatalogService
from rbac_assistant.application.services.validation_service import ValidationService
from rbac_assistant.domain.entities.normative_unit import (
    NormativeSection,
    NormativeUnitType,
    StructureStatus,
    resolve_section_tables,
)
from rbac_assistant.domain.entities.regulatory_document import ValidationPolicy
from rbac_assistant.domain.ports.document_repository_port import DocumentRepositoryPort


class GetNormativeSectionUseCase:
    """Recupera e reconstrói uma seção completa respeitando a política de validação vigente."""

    def __init__(
        self,
        *,
        repository: DocumentRepositoryPort,
        catalog_service: CatalogService,
        validation_service: ValidationService,
    ) -> None:
        self._repository = repository
        self._catalog_service = catalog_service
        self._validation_service = validation_service

    def execute(
        self,
        *,
        document_id: str,
        section_number: str,
        version_id: str | None = None,
        amendment_number: str | None = None,
        policy: ValidationPolicy | None = None,
    ) -> SectionLookupResultDTO:
        document = self._catalog_service.get_document(document_id)
        version = self._catalog_service.resolve_version(
            document_id=document_id,
            version_id=version_id,
            amendment_number=amendment_number,
        )

        # Bloqueio global obrigatório antes de consultar unidades ou expor qualquer texto
        all_version_pages = self._repository.get_pages(version.version_id)
        doc_assessment = self._validation_service.evaluate_document_version(
            version=version,
            pages=all_version_pages,
        )
        if not doc_assessment.literal_transcription_allowed:
            return SectionLookupResultDTO(
                found=False,
                section=None,
                auxiliary_unreviewed_available=False,
                blocked_by_document_approval=True,
                pending_pdf_pages=doc_assessment.pending_pdf_pages,
                rejected_pdf_pages=doc_assessment.rejected_pdf_pages,
                missing_pdf_pages=doc_assessment.missing_pdf_pages,
                message=(
                    doc_assessment.block_reason
                    or (
                        f"A versão '{version.version_id}' de {document.document_type.value} "
                        f"{document.regulation_number} ainda não possui todas as páginas "
                        "revisadas e aprovadas para transcrição literal."
                    )
                ),
            )

        units = self._repository.get_units_by_section(
            version_id=version.version_id,
            section_number=section_number,
        )
        if not units:
            return SectionLookupResultDTO(
                found=False,
                message=(
                    f"Seção '{section_number}' não localizada em {document.document_type.value} "
                    f"{document.regulation_number} (Emenda {version.amendment_number})."
                ),
            )

        # Seleciona exclusivamente a unidade-raiz efetiva (`parent_unit_id=None`);
        # jamais utiliza uma subunidade parcial como substituta da seção completa.
        all_roots = [u for u in units if u.parent_unit_id is None]
        typed_roots = [
            u
            for u in all_roots
            if u.unit_type in (NormativeUnitType.SECTION, NormativeUnitType.APPENDIX)
        ]
        root_units = typed_roots or all_roots
        if not root_units:
            return SectionLookupResultDTO(
                found=False,
                section=None,
                auxiliary_unreviewed_available=False,
                message=(
                    f"Seção '{section_number}' bloqueada por política fail-closed: "
                    "unidade-raiz (parent_unit_id=None) ausente no SQLite; não é permitido "
                    "retornar uma subunidade parcial como se fosse a seção completa."
                ),
            )
        primary = root_units[0]

        page_numbers = sorted({p for u in units for p in u.page_numbers})
        pages_by_num = {p.pdf_page_number: p for p in all_version_pages}
        pages = [pages_by_num[p_num] for p_num in page_numbers if p_num in pages_by_num]
        raw_page_tables = self._repository.list_tables(
            version.version_id, page_numbers=page_numbers
        )
        tables, ownership_issues, integrity_issues = resolve_section_tables(
            page_tables=raw_page_tables,
            unit_id=primary.unit_id,
            section_number=primary.section_number,
        )
        if ownership_issues or integrity_issues:
            block_details = "; ".join([*ownership_issues, *integrity_issues][:3])
            return SectionLookupResultDTO(
                found=False,
                section=None,
                auxiliary_unreviewed_available=False,
                message=(
                    f"Seção '{primary.section_number}' bloqueada por política fail-closed de "
                    f"tabelas: {block_details}"
                ),
            )
        references = tuple(primary.references)

        structure_status = (
            StructureStatus.PARSED
            if all(u.structure_status == StructureStatus.PARSED for u in (root_units or [primary]))
            else StructureStatus.PENDING_VALIDATION
        )

        assessment = self._validation_service.evaluate_section(
            section_number=primary.section_number,
            pages=pages,
            structure_status=structure_status,
            policy=policy,
            document_assessment=doc_assessment,
        )
        if not assessment.literal_transcription_allowed:
            return SectionLookupResultDTO(
                found=False,
                section=None,
                auxiliary_unreviewed_available=False,
                blocked_by_document_approval=True,
                pending_pdf_pages=assessment.pending_pdf_pages,
                message=(
                    assessment.warning_message
                    or f"Seção '{primary.section_number}' bloqueada pela política de aprovação."
                ),
            )

        printed_pages = tuple(
            p.printed_page_number or f"PDF {p.pdf_page_number}" for p in pages
        )
        section = NormativeSection(
            document_id=document.document_id,
            version_id=version.version_id,
            authority=document.authority,
            document_type=document.document_type.value,
            regulation_number=document.regulation_number,
            amendment_number=version.amendment_number,
            version_status=version.version_status.value,
            section_number=primary.section_number,
            title=primary.title,
            original_text=primary.original_text,
            normalized_text=primary.normalized_text,
            start_page=min(page_numbers),
            end_page=max(page_numbers),
            pdf_pages=tuple(page_numbers),
            printed_pages=printed_pages,
            validation_status=assessment.aggregate_status.value,
            structure_status=structure_status,
            literal_transcription_allowed=assessment.literal_transcription_allowed,
            pending_pdf_pages=assessment.pending_pdf_pages,
            ocr_blocked_pdf_pages=assessment.ocr_blocked_pdf_pages,
            warning_message=assessment.warning_message,
            tables=tables,
            references=references,
        )

        return SectionLookupResultDTO(
            found=True,
            section=section,
            auxiliary_unreviewed_available=False,
            message="Transcrição literal aprovada.",
        )
