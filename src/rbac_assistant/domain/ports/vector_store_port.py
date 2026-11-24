"""Contrato (Port) para operações de banco vetorial e verificação de saúde."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

MetadataValue = str | int | float | bool
MetadataFilterValue = MetadataValue | Sequence[MetadataValue]


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


@dataclass(frozen=True, slots=True)
class CollectionMetadata:
    """Metadados de controle de modelo, dimensão e estratégia associados a uma coleção vetorial."""

    name: str
    embedding_model: str | None = None
    embedding_dimension: int | None = None
    instruction_strategy_version: str | None = None
    created_at: str | None = None
    count: int = 0


class VectorStorePort(Protocol):
    """Interface de comunicação com o servidor vetorial (ex.: ChromaDB HTTP)."""

    def check_heartbeat(self) -> int:
        """Verifica conectividade com o servidor vetorial e retorna o timestamp de heartbeat."""

    def ensure_collection_compatible(
        self,
        *,
        collection_name: str,
        embedding_model: str,
        embedding_dimension: int,
        instruction_strategy_version: str | None = None,
    ) -> CollectionMetadata:
        """Cria ou valida coleção garantindo compatibilidade de modelo, dimensão e estratégia."""

    def get_collection_metadata(self, collection_name: str) -> CollectionMetadata | None:
        """Retorna metadados e contagem de uma coleção existente ou `None` se não existir."""

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

    def delete_records(
        self,
        *,
        collection_name: str,
        record_ids: Sequence[str],
    ) -> None:
        """Remove registros obsoletos específicos pelos seus identificadores."""

    def delete_by_where(
        self,
        *,
        collection_name: str,
        where: Mapping[str, MetadataFilterValue],
    ) -> None:
        """Remove todos os registros que satisfaçam um filtro de metadados."""

    def count_records(self, collection_name: str) -> int:
        """Retorna a quantidade de vetores persistidos na coleção (0 se inexistente)."""

    def list_record_ids(
        self,
        *,
        collection_name: str,
        where: Mapping[str, MetadataFilterValue] | None = None,
    ) -> list[str]:
        """Lista os IDs de registros persistidos na coleção (opcionalmente filtrados)."""

    def query_similar(
        self,
        *,
        collection_name: str,
        query_embedding: Sequence[float],
        top_k: int = 5,
        where: Mapping[str, MetadataFilterValue] | None = None,
    ) -> list[VectorSearchHit]:
        """Executa busca vetorial filtrada por metadados normativos."""

    def delete_collection(self, collection_name: str) -> None:
        """Remove uma coleção completa (usado em limpeza de testes ou reindexação)."""

