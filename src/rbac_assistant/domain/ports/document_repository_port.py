"""Contrato (Port) do repositório centralizado SSOT (catálogo, versões, páginas e revisões)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from rbac_assistant.domain.entities.document_version import DocumentVersion, ExtractionRun
from rbac_assistant.domain.entities.normative_unit import (
    NormativeReference,
    NormativeTable,
)
from rbac_assistant.domain.entities.rag import (
    IndexRegistryEntry,
    NormativeChunk,
    SearchFilter,
)
from rbac_assistant.domain.entities.regulatory_document import (
    DocumentType,
    RegulatoryDocument,
    VersionStatus,
)
from rbac_assistant.domain.entities.regulatory_page import (
    PageValidationStatus,
    RegulatoryPage,
    ValidationRecord,
)
from rbac_assistant.domain.ports.validation_port import ParsedUnitRecord


class DocumentRepositoryPort(Protocol):
    """Interface SSOT implementada em SQLite na infraestrutura."""

    def initialize(self) -> None:
        """Aplica migrações versionadas de esquema de forma idempotente."""

    def list_applied_migrations(self) -> list[str]:
        """Lista as versões de migração já aplicadas em `schema_migrations`."""

    def upsert_document(self, document: RegulatoryDocument) -> RegulatoryDocument:
        """Cadastra ou atualiza um documento regulatório no catálogo."""

    def get_document(self, document_id: str) -> RegulatoryDocument | None:
        """Recupera um documento pelo seu `document_id`."""

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
        """Pesquisa documentos cadastrados no catálogo por metadados."""

    def get_version(self, version_id: str) -> DocumentVersion | None:
        """Recupera uma versão específica pelo `version_id`."""

    def get_version_by_hash(
        self,
        *,
        document_id: str,
        amendment_number: str,
        source_hash: str,
    ) -> DocumentVersion | None:
        """Busca versão existente pela tríade (document_id, amendment_number, source_hash)."""

    def list_versions(self, document_id: str) -> list[DocumentVersion]:
        """Lista todas as versões (vigentes, históricas, propostas) de um documento."""

    def list_all_versions(self) -> list[DocumentVersion]:
        """Lista todas as versões ingeridas em todo o catálogo."""

    def get_current_version(self, document_id: str) -> DocumentVersion | None:
        """Retorna a versão marcada como `current` (vigente) de um documento."""

    def save_version_ingestion(
        self,
        *,
        document: RegulatoryDocument,
        version: DocumentVersion,
        pages: Sequence[RegulatoryPage],
        units: Sequence[ParsedUnitRecord],
        tables: Sequence[NormativeTable],
        extraction_run: ExtractionRun,
    ) -> list[ValidationRecord]:
        """Persiste atomicamente versão, páginas, unidades, tabelas e histórico.

        Retorna eventuais registros de invalidação gerados caso páginas já aprovadas
        tenham sofrido alteração de `content_hash` durante reprocessamento.
        """

    def record_extraction_run(self, run: ExtractionRun) -> None:
        """Registra tentativa de extração (inclusive idempotência ou erro)."""

    def list_extraction_runs(self, document_id: str) -> list[ExtractionRun]:
        """Lista o histórico de execuções de extração de um documento."""

    def get_pages(
        self,
        version_id: str,
        *,
        page_numbers: Sequence[int] | None = None,
    ) -> list[RegulatoryPage]:
        """Recupera páginas ordenadas por `pdf_page_number`."""

    def get_page(self, page_id: str) -> RegulatoryPage | None:
        """Recupera uma página pelo seu `page_id`."""

    def update_page_validation(
        self,
        *,
        page_id: str,
        new_status: PageValidationStatus,
        review_record: ValidationRecord,
    ) -> RegulatoryPage:
        """Atualiza o estado de validação da página e persiste o registro auditável de revisão."""

    def list_validation_records(self, page_id: str) -> list[ValidationRecord]:
        """Lista o histórico de revisões de uma página."""

    def get_units_by_section(
        self,
        *,
        version_id: str,
        section_number: str,
    ) -> list[ParsedUnitRecord]:
        """Recupera unidades normativas pertencentes a uma seção (em ordem)."""

    def list_units(self, version_id: str) -> list[ParsedUnitRecord]:
        """Lista todas as unidades normativas estruturadas de uma versão."""

    def list_tables(
        self,
        version_id: str,
        *,
        page_numbers: Sequence[int] | None = None,
    ) -> list[NormativeTable]:
        """Lista tabelas normativas de uma versão (opcionalmente filtradas por páginas)."""

    def list_references(self, unit_id: str) -> list[NormativeReference]:
        """Lista referências cruzadas vinculadas a uma unidade normativa."""

    def save_chunks(
        self,
        *,
        version_id: str,
        chunks: Sequence[NormativeChunk],
    ) -> list[str]:
        """Persiste chunks de uma versão, sincroniza FTS5 e retorna chunk_ids obsoletos."""

    def list_chunks(self, version_id: str) -> list[NormativeChunk]:
        """Lista todos os chunks estruturais de uma versão em ordem determinística."""

    def get_chunks_by_ids(self, chunk_ids: Sequence[str]) -> list[NormativeChunk]:
        """Recupera chunks estruturais diretamente pelos seus `chunk_id`s."""

    def count_chunks(self, version_id: str | None = None) -> int:
        """Conta chunks persistidos (de uma versão específica ou do banco inteiro)."""

    def find_chunks_by_structure(
        self,
        *,
        section_number: str,
        regulation_number: str | None = None,
        paragraph_number: str | None = None,
        item_number: str | None = None,
        subitem_number: str | None = None,
        leaf_number: str | None = None,
        filters: SearchFilter | None = None,
        limit: int = 10,
    ) -> list[NormativeChunk]:
        """Busca exata determinística por referência estrutural (ex.: RBAC 43.3, 43.9(a)(1)(i))."""

    def search_chunks_fts(
        self,
        *,
        query: str,
        filters: SearchFilter | None = None,
        limit: int = 10,
    ) -> list[tuple[str, float]]:
        """Executa busca textual FTS5 (BM25, tolerante a acentos) retornando (chunk_id, score)."""

    def rebuild_fts_index(self, *, version_id: str | None = None) -> int:
        """Reconstrói o índice derivado FTS5 a partir dos chunks persistidos no SQLite."""

    def upsert_index_registry(self, entry: IndexRegistryEntry) -> None:
        """Registra/atualiza o estado de indexação vetorial de uma versão em uma coleção."""

    def list_index_registry(
        self,
        *,
        collection_name: str | None = None,
    ) -> list[IndexRegistryEntry]:
        """Lista os registros de versões indexadas (opcionalmente filtrados por coleção)."""

    def delete_index_registry(
        self,
        *,
        collection_name: str,
        version_id: str | None = None,
    ) -> None:
        """Remove registro(s) de indexação de uma coleção (ex.: após rebuild)."""

    def checkpoint_wal(self) -> None:
        """Executa checkpoint WAL explícito para consistência antes de backups."""

    def backup_to(self, destination_path: Path) -> Path:
        """Gera backup consistente online do banco SQLite usando API nativa."""
