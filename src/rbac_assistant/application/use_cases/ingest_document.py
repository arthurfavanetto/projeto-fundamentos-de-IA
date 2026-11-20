"""Caso de uso para execução do pipeline genérico de ingestão documental."""

from __future__ import annotations

from rbac_assistant.application.dto.ingestion_dto import (
    IngestionRequestDTO,
    IngestionResultDTO,
)
from rbac_assistant.application.services.ingestion_service import IngestionService


class IngestDocumentUseCase:
    """Executa ingestão ponta a ponta (obtenção, hash, extração, estruturação e SQLite)."""

    def __init__(self, ingestion_service: IngestionService) -> None:
        self._ingestion_service = ingestion_service

    def execute(self, request: IngestionRequestDTO) -> IngestionResultDTO:
        return self._ingestion_service.ingest(request)
