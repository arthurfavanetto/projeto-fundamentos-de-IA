"""Caso de uso para cadastro/atualização de documento no catálogo regulatório."""

from __future__ import annotations

from rbac_assistant.application.dto.ingestion_dto import RegisterDocumentDTO
from rbac_assistant.application.services.catalog_service import CatalogService
from rbac_assistant.domain.entities.regulatory_document import RegulatoryDocument


class RegisterDocumentUseCase:
    """Registra ou atualiza metadados de qualquer documento regulatório no catálogo."""

    def __init__(self, catalog_service: CatalogService) -> None:
        self._catalog_service = catalog_service

    def execute(self, dto: RegisterDocumentDTO) -> RegulatoryDocument:
        return self._catalog_service.register_document(dto)
