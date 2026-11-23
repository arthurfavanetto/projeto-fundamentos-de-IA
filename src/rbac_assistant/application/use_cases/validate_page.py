"""Caso de uso para revisão humana explícita de páginas individuais, intervalos ou seções."""

from __future__ import annotations

from datetime import UTC, datetime

from rbac_assistant.application.dto.ingestion_dto import (
    PageReviewRequestDTO,
    PageReviewResultDTO,
)
from rbac_assistant.application.services.catalog_service import CatalogService
from rbac_assistant.application.services.validation_service import ValidationService
from rbac_assistant.domain.entities.regulatory_page import RegulatoryPage, ValidationRecord
from rbac_assistant.domain.exceptions import SectionNotFoundError, ValidationPolicyError
from rbac_assistant.domain.ports.audit_port import AuditEvent, AuditPort
from rbac_assistant.domain.ports.document_repository_port import DocumentRepositoryPort
from rbac_assistant.domain.ports.file_store_port import FileStorePort


class ValidatePageUseCase:
    """Aprova ou reprova explicitamente páginas (por número, intervalo ou seção) com auditoria."""

    def __init__(
        self,
        *,
        repository: DocumentRepositoryPort,
        catalog_service: CatalogService,
        validation_service: ValidationService,
        file_store: FileStorePort | None = None,
        audit_port: AuditPort | None = None,
    ) -> None:
        self._repository = repository
        self._catalog_service = catalog_service
        self._validation_service = validation_service
        self._file_store = file_store
        self._audit_port = audit_port

    def execute(self, request: PageReviewRequestDTO) -> PageReviewResultDTO:
        """Aplica revisão humana explícita às páginas indicadas e exporta registro auditável."""
        version = self._catalog_service.resolve_version(
            document_id=request.document_id,
            version_id=request.version_id,
        )

        target_page_numbers: list[int] = list(request.page_numbers)
        if request.section_number:
            units = self._repository.get_units_by_section(
                version_id=version.version_id,
                section_number=request.section_number,
            )
            if not units:
                raise SectionNotFoundError(
                    f"Seção '{request.section_number}' não encontrada na versão "
                    f"'{version.version_id}'."
                )
            section_pages = sorted({p for u in units for p in u.page_numbers})
            target_page_numbers.extend(section_pages)

        unique_pages = sorted(set(target_page_numbers))
        if not unique_pages:
            raise ValidationPolicyError(
                "Informe explicitamente ao menos uma página, intervalo ou seção para revisão. "
                "Aprovação silenciosa de todas as páginas não é permitida."
            )

        pages = self._repository.get_pages(version.version_id, page_numbers=unique_pages)
        found_numbers = {p.pdf_page_number for p in pages}
        missing = sorted(set(unique_pages) - found_numbers)
        if missing:
            raise ValidationPolicyError(
                f"Página(s) inexistente(s) na versão '{version.version_id}': {missing}."
            )

        updated_pages: list[RegulatoryPage] = []
        records: list[ValidationRecord] = []
        exported_paths: list[str] = []

        for page in pages:
            new_status, record = self._validation_service.build_review_transition(
                page=page,
                decision=request.decision,
                reviewer=request.reviewer,
                confirmed=request.confirmed,
                comments=request.comments,
            )
            updated = self._repository.update_page_validation(
                page_id=page.page_id,
                new_status=new_status,
                review_record=record,
            )
            updated_pages.append(updated)
            records.append(record)

            if self._file_store is not None:
                review_path = self._file_store.save_review_record(
                    document_id=request.document_id,
                    review_id=record.review_id,
                    payload={
                        "review_id": record.review_id,
                        "document_id": request.document_id,
                        "version_id": version.version_id,
                        "page_id": page.page_id,
                        "pdf_page_number": page.pdf_page_number,
                        "printed_page_number": page.printed_page_number,
                        "reviewer": record.reviewer,
                        "reviewed_at": record.reviewed_at.isoformat(),
                        "decision": record.decision.value,
                        "new_status": new_status.value,
                        "source_hash": record.source_hash,
                        "comments": record.comments,
                    },
                )
                exported_paths.append(str(review_path))

        if self._audit_port is not None:
            self._audit_port.record(
                AuditEvent(
                    timestamp=datetime.now(UTC),
                    action="page_review_recorded",
                    status=request.decision.value,
                    actor_id=request.reviewer,
                    details={
                        "document_id": request.document_id,
                        "version_id": version.version_id,
                        "pages_count": len(updated_pages),
                        "section_number": request.section_number or "",
                    },
                )
            )

        return PageReviewResultDTO(
            document_id=request.document_id,
            version_id=version.version_id,
            updated_pages=tuple(updated_pages),
            records=tuple(records),
            exported_review_paths=tuple(exported_paths),
        )
