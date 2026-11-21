"""Serviço de indexação lexical (SQLite FTS5) e vetorial (ChromaDB) com idempotência.

Garantias:
- Idempotente: reindexar o mesmo documento/versão sem alterações não duplica chunks nem vetores.
- Limpeza determinística: remove chunks/vetores obsoletos em reindexações ou atualizações.
- Rastreabilidade completa de metadados (documento, versão, emenda, seção, páginas, hashes e
  estado de revisão).
- Compatibilidade vetorial: impede mistura de embeddings de modelos ou dimensões distintos e
  permite reconstrução controlada do índice vetorial a partir do SQLite (SSOT).
- Invariante de governança: NUNCA aprova páginas automaticamente nem altera `validation_records`.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

import structlog

from rbac_assistant.application.services.catalog_service import CatalogService
from rbac_assistant.domain.entities.rag import (
    IndexingOutcome,
    IndexRegistryEntry,
    IndexStatusReport,
    IndexVersionSummary,
    NormativeChunk,
)
from rbac_assistant.domain.exceptions import (
    EmbeddingCompatibilityError,
    IndexingError,
    StorageConnectionError,
)
from rbac_assistant.domain.ports.document_repository_port import DocumentRepositoryPort
from rbac_assistant.domain.ports.rag_ports import ChunkingPort, EmbeddingPort
from rbac_assistant.domain.ports.vector_store_port import (
    MetadataValue,
    VectorRecord,
    VectorStorePort,
)

logger = structlog.get_logger(__name__)


class IndexingService:
    """Orquestra chunking estrutural, sincronização FTS5 e indexação vetorial no ChromaDB."""

    def __init__(
        self,
        *,
        repository: DocumentRepositoryPort,
        catalog_service: CatalogService,
        chunker: ChunkingPort,
        vector_store: VectorStorePort,
        embedding_port: EmbeddingPort,
        default_collection_name: str = "regulatory_chunks_ge2_768_v1",
    ) -> None:
        self._repository = repository
        self._catalog_service = catalog_service
        self._chunker = chunker
        self._vector_store = vector_store
        self._embedding_port = embedding_port
        self._default_collection_name = default_collection_name

    def index_document(
        self,
        *,
        document_id: str,
        version_id: str | None = None,
        amendment_number: str | None = None,
        collection_name: str | None = None,
        force_reindex: bool = False,
    ) -> IndexingOutcome:
        """Indexa ou reindexa uma versão documental específica no SQLite FTS5 e no ChromaDB."""
        target_collection = collection_name or self._default_collection_name
        document = self._catalog_service.get_document(document_id)
        version = self._catalog_service.resolve_version(
            document_id=document_id,
            version_id=version_id,
            amendment_number=amendment_number,
        )

        # Verifica compatibilidade de modelo, dimensão e estratégia da coleção antes de gravar
        self._vector_store.ensure_collection_compatible(
            collection_name=target_collection,
            embedding_model=self._embedding_port.model_name,
            embedding_dimension=self._embedding_port.dimensions,
            instruction_strategy_version=self._embedding_port.instruction_strategy_version,
        )

        # Verifica idempotência quando não for reindexação forçada
        existing_registry = {
            entry.version_id: entry
            for entry in self._repository.list_index_registry(collection_name=target_collection)
        }
        reg_entry = existing_registry.get(version.version_id)
        existing_sqlite_chunks = self._repository.list_chunks(version.version_id)
        if (
            not force_reindex
            and reg_entry is not None
            and reg_entry.source_hash == version.source_hash
            and reg_entry.embedding_model == self._embedding_port.model_name
            and reg_entry.embedding_dimension == self._embedding_port.dimensions
            and reg_entry.instruction_strategy_version
            == self._embedding_port.instruction_strategy_version
            and len(existing_sqlite_chunks) == reg_entry.chunk_count
            and reg_entry.chunk_count > 0
        ):
            existing_chroma_ids = self._vector_store.list_record_ids(
                collection_name=target_collection,
                where={"version_id": version.version_id},
            )
            if len(existing_chroma_ids) == reg_entry.chunk_count:
                logger.info(
                    "indexing_skipped_up_to_date",
                    document_id=document.document_id,
                    version_id=version.version_id,
                    collection=target_collection,
                    chunks=reg_entry.chunk_count,
                )
                return IndexingOutcome(
                    document_id=document.document_id,
                    version_id=version.version_id,
                    collection_name=target_collection,
                    embedding_model=self._embedding_port.model_name,
                    embedding_dimension=self._embedding_port.dimensions,
                    chunks_created=reg_entry.chunk_count,
                    stale_chunks_removed=0,
                    skipped_up_to_date=True,
                )

        units = self._repository.list_units(version.version_id)
        pages = self._repository.get_pages(version.version_id)
        if not units:
            raise IndexingError(
                f"Versão '{version.version_id}' do documento '{document.document_id}' "
                "não possui unidades normativas extraídas para indexação."
            )

        raw_chunks = self._chunker.build_chunks(
            document=document,
            version=version,
            units=units,
            pages=pages,
        )
        if not raw_chunks:
            raise IndexingError(
                f"Nenhum chunk estrutural gerado para '{document.document_id}' "
                f"(versão '{version.version_id}')."
            )

        # Gera 1 embedding independente por chunk usando título estrutural separado
        # (nunca altera `original_text` nem `normalized_text` persistidos no SQLite)
        chunk_titles = [self._build_embedding_title(c) for c in raw_chunks]
        vectors = self._embedding_port.embed_documents(
            [c.search_text for c in raw_chunks],
            titles=chunk_titles,
        )
        if len(vectors) != len(raw_chunks):
            raise IndexingError(
                f"Divergência entre chunks ({len(raw_chunks)}) e embeddings gerados "
                f"({len(vectors)})."
            )

        enriched_chunks = [
            replace(
                chunk,
                embedding_model=self._embedding_port.model_name,
                embedding_dimension=len(vec),
            )
            for chunk, vec in zip(raw_chunks, vectors, strict=True)
        ]

        # Identifica vetores obsoletos no ChromaDB antes do upsert
        new_chunk_ids = {chunk.chunk_id for chunk in enriched_chunks}
        previous_chroma_ids = self._vector_store.list_record_ids(
            collection_name=target_collection,
            where={"version_id": version.version_id},
        )
        stale_chroma_ids = [cid for cid in previous_chroma_ids if cid not in new_chunk_ids]

        # 1. Grava vetores no ChromaDB com metadados normativos rastreáveis
        records = [
            VectorRecord(
                record_id=chunk.chunk_id,
                document=chunk.search_text,
                embedding=vec,
                metadata=self._build_vector_metadata(chunk),
            )
            for chunk, vec in zip(enriched_chunks, vectors, strict=True)
        ]
        self._vector_store.upsert_records(collection_name=target_collection, records=records)

        # 2. Remove vetores obsoletos da versão no ChromaDB
        if stale_chroma_ids:
            self._vector_store.delete_records(
                collection_name=target_collection,
                record_ids=stale_chroma_ids,
            )

        # 3. Persiste chunks + sincroniza FTS5 no SQLite (sem alterar estados de revisão de páginas)
        removed_sqlite_ids = self._repository.save_chunks(
            version_id=version.version_id,
            chunks=enriched_chunks,
        )
        extra_stale = [cid for cid in removed_sqlite_ids if cid not in stale_chroma_ids]
        if extra_stale:
            self._vector_store.delete_records(
                collection_name=target_collection,
                record_ids=extra_stale,
            )

        # 4. Atualiza registro de indexação da coleção
        self._repository.upsert_index_registry(
            IndexRegistryEntry(
                collection_name=target_collection,
                version_id=version.version_id,
                document_id=document.document_id,
                source_hash=version.source_hash,
                embedding_model=self._embedding_port.model_name,
                embedding_dimension=self._embedding_port.dimensions,
                chunk_count=len(enriched_chunks),
                indexed_at=datetime.now(UTC),
                instruction_strategy_version=self._embedding_port.instruction_strategy_version,
            )
        )

        total_removed = len(set(stale_chroma_ids) | set(removed_sqlite_ids))
        logger.info(
            "document_version_indexed",
            document_id=document.document_id,
            version_id=version.version_id,
            collection=target_collection,
            chunks_created=len(enriched_chunks),
            stale_chunks_removed=total_removed,
            embedding_model=self._embedding_port.model_name,
            embedding_dimension=self._embedding_port.dimensions,
        )
        return IndexingOutcome(
            document_id=document.document_id,
            version_id=version.version_id,
            collection_name=target_collection,
            embedding_model=self._embedding_port.model_name,
            embedding_dimension=self._embedding_port.dimensions,
            chunks_created=len(enriched_chunks),
            stale_chunks_removed=total_removed,
            skipped_up_to_date=False,
        )

    def rebuild_vector_index(
        self,
        *,
        collection_name: str | None = None,
        document_id: str | None = None,
    ) -> list[IndexingOutcome]:
        """Reconstrói a coleção vetorial integralmente a partir do SQLite (SSOT).

        Uso típico: após trocar `GEMINI_EMBEDDING_MODEL` ou `GEMINI_EMBEDDING_DIMENSIONS`.
        Preserva 100% do SQLite, PDFs originais e aprovações humanas de revisão.
        """
        target_collection = collection_name or self._default_collection_name
        self._vector_store.delete_collection(target_collection)
        self._repository.delete_index_registry(collection_name=target_collection)

        if document_id:
            versions = self._repository.list_versions(document_id)
        else:
            versions = self._repository.list_all_versions()

        outcomes: list[IndexingOutcome] = []
        for version in versions:
            if not self._repository.list_units(version.version_id):
                continue
            outcomes.append(
                self.index_document(
                    document_id=version.document_id,
                    version_id=version.version_id,
                    collection_name=target_collection,
                    force_reindex=True,
                )
            )
        return outcomes

    def get_index_status(
        self,
        *,
        collection_name: str | None = None,
    ) -> IndexStatusReport:
        """Inspeciona o estado consolidado de SQLite FTS5, ChromaDB e defasagens."""
        target_collection = collection_name or self._default_collection_name
        documents = self._repository.search_documents()
        doc_by_id = {doc.document_id: doc for doc in documents}
        versions = self._repository.list_all_versions()
        registry_by_version = {
            entry.version_id: entry
            for entry in self._repository.list_index_registry(collection_name=target_collection)
        }

        chroma_available = True
        chroma_vector_count = 0
        collection_model: str | None = None
        collection_dim: int | None = None
        compatibility_warning: str | None = None

        try:
            col_meta = self._vector_store.get_collection_metadata(target_collection)
            if col_meta is not None:
                chroma_vector_count = col_meta.count
                collection_model = col_meta.embedding_model
                collection_dim = col_meta.embedding_dimension
                if col_meta.count > 0 and (
                    (
                        collection_model is not None
                        and collection_model != self._embedding_port.model_name
                    )
                    or (
                        collection_dim is not None
                        and collection_dim != self._embedding_port.dimensions
                    )
                ):
                    compatibility_warning = (
                        f"Coleção '{target_collection}' usa {collection_model} "
                        f"({collection_dim}d), divergente da configuração atual "
                        f"{self._embedding_port.model_name} ({self._embedding_port.dimensions}d). "
                        "Execute rebuild controlado."
                    )
        except (StorageConnectionError, EmbeddingCompatibilityError) as exc:
            chroma_available = False
            compatibility_warning = str(exc)

        summaries: list[IndexVersionSummary] = []
        unindexed: list[str] = []
        stale: list[str] = []

        for ver in versions:
            doc = doc_by_id.get(ver.document_id)
            sqlite_count = self._repository.count_chunks(ver.version_id)
            entry = registry_by_version.get(ver.version_id)
            is_indexed = entry is not None and sqlite_count > 0
            is_stale = False
            stale_reasons: list[str] = []

            if not is_indexed:
                unindexed.append(ver.version_id)
            elif entry is not None:
                if entry.source_hash != ver.source_hash:
                    is_stale = True
                    stale_reasons.append("source_hash alterado")
                if entry.embedding_model != self._embedding_port.model_name:
                    is_stale = True
                    stale_reasons.append(
                        f"modelo alterado ({entry.embedding_model} -> "
                        f"{self._embedding_port.model_name})"
                    )
                if entry.embedding_dimension != self._embedding_port.dimensions:
                    is_stale = True
                    stale_reasons.append(
                        f"dimensão alterada ({entry.embedding_dimension} -> "
                        f"{self._embedding_port.dimensions})"
                    )
                if (
                    entry.instruction_strategy_version
                    and entry.instruction_strategy_version
                    != self._embedding_port.instruction_strategy_version
                ):
                    is_stale = True
                    stale_reasons.append(
                        f"estratégia alterada ({entry.instruction_strategy_version} -> "
                        f"{self._embedding_port.instruction_strategy_version})"
                    )
                if entry.chunk_count != sqlite_count:
                    is_stale = True
                    stale_reasons.append(
                        f"contagem divergente (registro={entry.chunk_count}, sqlite={sqlite_count})"
                    )
                if is_stale:
                    stale.append(ver.version_id)

            summaries.append(
                IndexVersionSummary(
                    document_id=ver.document_id,
                    version_id=ver.version_id,
                    regulation_number=doc.regulation_number if doc else "",
                    amendment_number=ver.amendment_number,
                    version_status=ver.version_status.value,
                    sqlite_chunk_count=sqlite_count,
                    indexed_chunk_count=entry.chunk_count if entry else 0,
                    is_indexed=is_indexed,
                    is_stale=is_stale,
                    stale_reason="; ".join(stale_reasons),
                )
            )

        return IndexStatusReport(
            collection_name=target_collection,
            configured_embedding_model=self._embedding_port.model_name,
            configured_embedding_dimension=self._embedding_port.dimensions,
            collection_embedding_model=collection_model,
            collection_embedding_dimension=collection_dim,
            chroma_available=chroma_available,
            chroma_vector_count=chroma_vector_count,
            total_sqlite_chunks=self._repository.count_chunks(),
            total_documents=len(documents),
            total_versions=len(versions),
            indexed_versions_count=len(versions) - len(unindexed),
            unindexed_versions=tuple(unindexed),
            stale_versions=tuple(stale),
            version_summaries=tuple(summaries),
            compatibility_warning=compatibility_warning,
        )

    @staticmethod
    def _build_embedding_title(chunk: NormativeChunk) -> str | None:
        parts = [
            f"{chunk.document_type} {chunk.regulation_number}".strip(),
            f"Emenda {chunk.amendment_number}" if chunk.amendment_number else "",
            f"Seção {chunk.section_number}" if chunk.section_number else "",
            chunk.title.strip(),
        ]
        joined = " — ".join(part for part in parts if part)
        return joined or None

    def _build_vector_metadata(self, chunk: NormativeChunk) -> dict[str, MetadataValue]:
        return {
            "chunk_id": chunk.chunk_id,
            "unit_id": chunk.unit_id,
            "document_id": chunk.document_id,
            "version_id": chunk.version_id,
            "authority": chunk.authority,
            "document_type": chunk.document_type,
            "regulation_number": chunk.regulation_number,
            "amendment_number": chunk.amendment_number,
            "version_status": chunk.version_status,
            "section_number": chunk.section_number,
            "subsection_number": chunk.subsection_number,
            "paragraph_number": chunk.paragraph_number,
            "item_number": chunk.item_number,
            "title": chunk.title,
            "start_page": chunk.start_page,
            "end_page": chunk.end_page,
            "printed_pages": ",".join(chunk.printed_pages),
            "source_hash": chunk.source_hash,
            "content_hash": chunk.content_hash,
            "validation_status": chunk.validation_status,
            "embedding_model": chunk.embedding_model,
            "embedding_dimension": chunk.embedding_dimension,
            "instruction_strategy_version": self._embedding_port.instruction_strategy_version,
        }
