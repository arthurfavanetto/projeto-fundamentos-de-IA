"""Contratos (Ports) de chunking estrutural e reexportação dos ports RAG."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from rbac_assistant.domain.entities.document_version import DocumentVersion
from rbac_assistant.domain.entities.rag import NormativeChunk
from rbac_assistant.domain.entities.regulatory_document import RegulatoryDocument
from rbac_assistant.domain.entities.regulatory_page import RegulatoryPage
from rbac_assistant.domain.ports.embedding_port import EmbeddingPort
from rbac_assistant.domain.ports.llm_port import LLMPort
from rbac_assistant.domain.ports.validation_port import ParsedUnitRecord


class ChunkingPort(Protocol):
    """Interface para transformação determinística de unidades normativas em chunks."""

    def build_chunks(
        self,
        *,
        document: RegulatoryDocument,
        version: DocumentVersion,
        units: Sequence[ParsedUnitRecord],
        pages: Sequence[RegulatoryPage],
    ) -> list[NormativeChunk]:
        """Produz chunks hierárquicos determinísticos preservando texto original e metadados."""


__all__ = ["ChunkingPort", "EmbeddingPort", "LLMPort"]
