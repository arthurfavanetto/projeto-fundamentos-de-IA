"""CLI administrativo para indexação lexical (SQLite FTS5) e vetorial (ChromaDB).

Exemplos:
    uv run python scripts/index_document.py --status
    uv run python scripts/index_document.py --document-id anac_rbac_43
    uv run python scripts/index_document.py --document-id anac_rbac_43 --reindex
    uv run python scripts/index_document.py --rebuild-collection --confirm

IMPORTANTE: Este script nunca altera o status de validação humana de páginas nem
executa indexação implícita sem argumentos explícitos.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from rbac_assistant.application.services.catalog_service import CatalogService
from rbac_assistant.application.services.indexing_service import IndexingService
from rbac_assistant.config.settings import AppSettings, load_settings
from rbac_assistant.domain.ports.rag_ports import EmbeddingPort
from rbac_assistant.infrastructure.llm.gemini_client import (
    GeminiClientAdapter,
    GeminiEmbeddingAdapter,
)
from rbac_assistant.infrastructure.storage.sqlite_document_repository import (
    SqliteDocumentRepository,
)
from rbac_assistant.infrastructure.vector_db.chroma_vector_store import ChromaVectorStore
from rbac_assistant.rag.ingestion.structural_chunker import StructuralNormativeChunker


class _MetadataOnlyEmbeddingDescriptor:
    """Descritor somente leitura usado por `--status` sem exigir `GEMINI_API_KEY`."""

    def __init__(self, *, model_name: str, dimensions: int) -> None:
        self._model_name = model_name
        self._dimensions = dimensions

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed_documents(self, texts: Any) -> list[tuple[float, ...]]:
        raise RuntimeError("Operação de embedding indisponível no modo somente leitura (--status).")

    def embed_query(self, text: str) -> tuple[float, ...]:
        raise RuntimeError("Operação de embedding indisponível no modo somente leitura (--status).")


def _build_indexing_service(
    *,
    settings: AppSettings,
    documents_dir: Path | None,
    collection_name: str | None,
    require_api_key: bool,
) -> tuple[IndexingService, GeminiClientAdapter | None]:
    base_dir = documents_dir or settings.storage.documents_dir
    repository = SqliteDocumentRepository(
        base_dir / settings.storage.sqlite_filename,
        busy_timeout_ms=settings.storage.sqlite_busy_timeout_ms,
    )
    repository.initialize()
    catalog = CatalogService(repository)
    chunker = StructuralNormativeChunker(max_chunk_tokens=settings.chunking.max_chunk_tokens)
    vector_store = ChromaVectorStore(
        host=settings.chroma_host,
        port=settings.chroma_port,
        heartbeat_path=settings.vector_db.heartbeat_path,
        timeout_seconds=settings.vector_db.request_timeout_seconds,
    )

    gemini_client: GeminiClientAdapter | None = None
    embedding_port: EmbeddingPort
    if require_api_key:
        api_key = settings.require_gemini_api_key()
        gemini_client = GeminiClientAdapter(
            api_key=api_key,
            generation_model=settings.gemini_generation_model,
            embedding_model=settings.gemini_embedding_model,
            embedding_dimensions=settings.effective_embedding_dimensions,
            timeout_seconds=settings.gemini.timeout_seconds,
            max_retries=settings.gemini.max_retries,
            retry_min_wait_seconds=settings.gemini.retry_min_wait_seconds,
            retry_max_wait_seconds=settings.gemini.retry_max_wait_seconds,
            embedding_batch_size=settings.gemini.embedding_batch_size,
        )
        embedding_port = GeminiEmbeddingAdapter(gemini_client)
    else:
        embedding_port = _MetadataOnlyEmbeddingDescriptor(
            model_name=settings.gemini_embedding_model,
            dimensions=settings.effective_embedding_dimensions,
        )

    service = IndexingService(
        repository=repository,
        catalog_service=catalog,
        chunker=chunker,
        vector_store=vector_store,
        embedding_port=embedding_port,
        default_collection_name=collection_name or settings.vector_db.collection_name,
    )
    return service, gemini_client


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Indexação controlada (FTS5 + ChromaDB) e diagnóstico de índices normativos."
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Exibe status detalhado dos índices (chunks, vetores, não indexados, desatualizados).",
    )
    parser.add_argument("--document-id", help="Documento específico a indexar/reindexar.")
    parser.add_argument("--version-id", help="Versão específica do documento.")
    parser.add_argument("--amendment", help="Emenda específica do documento.")
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Força reindexação controlada do documento especificado.",
    )
    parser.add_argument(
        "--rebuild-collection",
        action="store_true",
        help="Reconstrói a coleção vetorial inteira a partir do SQLite (exige --confirm).",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Confirmação explícita exigida para --rebuild-collection.",
    )
    parser.add_argument("--collection", help="Sobrescreve o nome da coleção no ChromaDB.")
    parser.add_argument("--documents-dir", type=Path, help="Sobrescreve o diretório de dados.")
    args = parser.parse_args(argv)

    if not args.status and not args.document_id and not args.rebuild_collection:
        parser.error("Informe --status, --document-id <id> ou --rebuild-collection --confirm.")

    if args.rebuild_collection and not args.confirm:
        parser.error("A reconstrução completa da coleção exige --confirm explícito.")

    settings = load_settings()
    require_api_key = bool(args.document_id or args.rebuild_collection)
    service, gemini_client = _build_indexing_service(
        settings=settings,
        documents_dir=args.documents_dir,
        collection_name=args.collection,
        require_api_key=require_api_key,
    )

    try:
        if args.status:
            report = service.get_index_status(collection_name=args.collection)
            print(
                json.dumps(
                    {
                        "collection_name": report.collection_name,
                        "configured_embedding_model": report.configured_embedding_model,
                        "configured_embedding_dimension": report.configured_embedding_dimension,
                        "collection_embedding_model": report.collection_embedding_model,
                        "collection_embedding_dimension": report.collection_embedding_dimension,
                        "chroma_available": report.chroma_available,
                        "chroma_vector_count": report.chroma_vector_count,
                        "total_sqlite_chunks": report.total_sqlite_chunks,
                        "total_documents": report.total_documents,
                        "total_versions": report.total_versions,
                        "indexed_versions_count": report.indexed_versions_count,
                        "unindexed_versions": list(report.unindexed_versions),
                        "stale_versions": list(report.stale_versions),
                        "compatibility_warning": report.compatibility_warning,
                        "versions": [
                            {
                                "document_id": v.document_id,
                                "version_id": v.version_id,
                                "regulation_number": v.regulation_number,
                                "amendment_number": v.amendment_number,
                                "version_status": v.version_status,
                                "sqlite_chunk_count": v.sqlite_chunk_count,
                                "indexed_chunk_count": v.indexed_chunk_count,
                                "is_indexed": v.is_indexed,
                                "is_stale": v.is_stale,
                                "stale_reason": v.stale_reason,
                            }
                            for v in report.version_summaries
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0

        if args.rebuild_collection:
            outcomes = service.rebuild_vector_index(
                collection_name=args.collection,
                document_id=args.document_id,
            )
            print(
                json.dumps(
                    {
                        "rebuilt_versions": len(outcomes),
                        "outcomes": [
                            {
                                "document_id": o.document_id,
                                "version_id": o.version_id,
                                "collection_name": o.collection_name,
                                "embedding_model": o.embedding_model,
                                "embedding_dimension": o.embedding_dimension,
                                "chunks_created": o.chunks_created,
                                "stale_chunks_removed": o.stale_chunks_removed,
                            }
                            for o in outcomes
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0

        outcome = service.index_document(
            document_id=args.document_id,
            version_id=args.version_id,
            amendment_number=args.amendment,
            collection_name=args.collection,
            force_reindex=args.reindex,
        )
        print(
            json.dumps(
                {
                    "document_id": outcome.document_id,
                    "version_id": outcome.version_id,
                    "collection_name": outcome.collection_name,
                    "embedding_model": outcome.embedding_model,
                    "embedding_dimension": outcome.embedding_dimension,
                    "chunks_created": outcome.chunks_created,
                    "stale_chunks_removed": outcome.stale_chunks_removed,
                    "skipped_up_to_date": outcome.skipped_up_to_date,
                    "warnings": list(outcome.warnings),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    finally:
        if gemini_client is not None:
            gemini_client.close()


if __name__ == "__main__":
    sys.exit(main())
