"""Serviço de gestão e pesquisa no catálogo centralizado multidocumento."""

from __future__ import annotations

from rbac_assistant.application.dto.ingestion_dto import RegisterDocumentDTO
from rbac_assistant.domain.entities.document_version import DocumentVersion
from rbac_assistant.domain.entities.regulatory_document import (
    DocumentType,
    RegulatoryDocument,
    VersionStatus,
)
from rbac_assistant.domain.exceptions import DocumentNotFoundError
from rbac_assistant.domain.ports.document_repository_port import DocumentRepositoryPort


class CatalogService:
    """Orquestra cadastro, listagem e busca no catálogo sem acoplamento a um regulamento."""

    def __init__(self, repository: DocumentRepositoryPort) -> None:
        self._repository = repository

    def register_document(self, dto: RegisterDocumentDTO) -> RegulatoryDocument:
        """Cadastra ou atualiza um documento no catálogo SQLite."""
        entity = RegulatoryDocument(
            document_id=dto.document_id,
            authority=dto.authority,
            document_type=dto.document_type,
            regulation_number=dto.regulation_number,
            document_title=dto.document_title,
            document_status=dto.document_status,
            version_status=dto.version_status,
            amendment_number=dto.amendment_number,
            publication_date=dto.publication_date,
            effective_date=dto.effective_date,
            expiration_date=dto.expiration_date,
            official_url=dto.official_url,
            source_url=dto.source_url,
            pdf_filename=dto.pdf_filename,
            last_verified_at=dto.last_verified_at,
            subject_categories=dto.subject_categories,
            document_description=dto.document_description,
        )
        return self._repository.upsert_document(entity)

    def get_document(self, document_id: str) -> RegulatoryDocument:
        doc = self._repository.get_document(document_id)
        if doc is None:
            raise DocumentNotFoundError(f"Documento não encontrado no catálogo: '{document_id}'.")
        return doc

    def search_documents(
        self,
        *,
        authority: str | None = None,
        document_type: DocumentType | None = None,
        regulation_number: str | None = None,
        subject: str | None = None,
        amendment_number: str | None = None,
        version_status: VersionStatus | None = None,
    ) -> list[RegulatoryDocument]:
        return self._repository.search_documents(
            authority=authority,
            document_type=document_type,
            regulation_number=regulation_number,
            subject=subject,
            amendment_number=amendment_number,
            version_status=version_status,
        )

    def list_versions(self, document_id: str) -> list[DocumentVersion]:
        self.get_document(document_id)
        return self._repository.list_versions(document_id)

    def resolve_version(
        self,
        *,
        document_id: str,
        version_id: str | None = None,
        amendment_number: str | None = None,
    ) -> DocumentVersion:
        """Resolve versão explícita ou a versão vigente (`current`) sem misturar versões."""
        if version_id:
            version = self._repository.get_version(version_id)
            if version is None or version.document_id != document_id:
                raise DocumentNotFoundError(
                    f"Versão '{version_id}' não encontrada para o documento '{document_id}'."
                )
            return version

        versions = self._repository.list_versions(document_id)
        if not versions:
            raise DocumentNotFoundError(
                f"Nenhuma versão ingerida para o documento '{document_id}'."
            )

        if amendment_number:
            for ver in versions:
                if ver.amendment_number.lower() == amendment_number.strip().lower():
                    return ver
            raise DocumentNotFoundError(
                f"Emenda '{amendment_number}' não encontrada para '{document_id}'."
            )

        current = self._repository.get_current_version(document_id)
        if current is not None:
            return current
        return versions[0]
