"""Contrato (Port) para armazenamento canônico (SSOT) de documentos normativos."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from rbac_assistant.domain.entities.normative_unit import NormativeUnit
from rbac_assistant.domain.entities.regulation import DocumentMetadata


class DocumentStorePort(Protocol):
    """Interface de persistência e leitura determinística do repositório canônico (SSOT)."""

    def save_document(
        self,
        metadata: DocumentMetadata,
        units: Sequence[NormativeUnit],
    ) -> None:
        """Persiste metadados oficiais e unidades normativas canônicas de forma atômica."""

    def get_document_metadata(self, document_id: str) -> DocumentMetadata | None:
        """Recupera os metadados oficiais de um documento pelo identificador."""

    def list_documents(self) -> list[DocumentMetadata]:
        """Lista todos os documentos normativos registrados no repositório canônico."""

    def get_section(
        self,
        *,
        regulation_number: str,
        section_number: str,
        amendment_number: str | None = None,
    ) -> NormativeUnit | None:
        """Busca determinística de uma seção/dispositivo exato no armazenamento canônico."""
