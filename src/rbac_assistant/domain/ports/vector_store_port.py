"""Contrato (Port) para operações de banco vetorial e verificação de saúde."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

MetadataValue = str | int | float | bool


@dataclass(frozen=True, slots=True)
class VectorRecord:
    """Registro vetorial determinístico armazenado em uma coleção do banco vetorial."""

    record_id: str
    document: str
    embedding: tuple[float, ...]
    metadata: Mapping[str, MetadataValue]


@dataclass(frozen=True, slots=True)
class VectorSearchHit:
    """Resultado individual retornado por busca semântica vetorial."""

    record_id: str
    document: str
    metadata: Mapping[str, MetadataValue]
    distance: float


class VectorStorePort(Protocol):
    """Interface de comunicação com o servidor vetorial (ex.: ChromaDB HTTP)."""

    def check_heartbeat(self) -> int:
        """Verifica conectividade com o servidor vetorial e retorna o timestamp de heartbeat."""

    def upsert_records(
        self,
        *,
        collection_name: str,
        records: Sequence[VectorRecord],
    ) -> None:
        """Insere ou atualiza registros vetoriais de forma idempotente."""

    def get_records(
        self,
        *,
        collection_name: str,
        record_ids: Sequence[str],
    ) -> list[VectorRecord]:
        """Recupera registros vetoriais diretamente pelos seus IDs determinísticos."""

    def query_similar(
        self,
        *,
        collection_name: str,
        query_embedding: Sequence[float],
        top_k: int = 5,
        where: Mapping[str, MetadataValue] | None = None,
    ) -> list[VectorSearchHit]:
        """Executa busca vetorial filtrada por metadados normativos."""

    def delete_collection(self, collection_name: str) -> None:
        """Remove uma coleção completa (usado em limpeza de testes ou reindexação)."""
