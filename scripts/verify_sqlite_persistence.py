"""Smoke test operacional de persistência SQLite + PDF original + revisão após restart Docker.

Executado dentro do contêiner `rbac-bot` (UID/GID 10001) sobre `/app/data/documents`:
    python scripts/verify_sqlite_persistence.py --phase write
    docker compose restart rbac-bot
    python scripts/verify_sqlite_persistence.py --phase verify
    python scripts/verify_sqlite_persistence.py --phase cleanup
"""

from __future__ import annotations

import argparse
import hashlib
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path

from rbac_assistant.application.dto.ingestion_dto import (
    PageReviewRequestDTO,
    RegisterDocumentDTO,
)
from rbac_assistant.application.services.catalog_service import CatalogService
from rbac_assistant.application.services.validation_service import ValidationService
from rbac_assistant.application.use_cases.validate_page import ValidatePageUseCase
from rbac_assistant.config.settings import load_settings
from rbac_assistant.domain.entities.document_version import DocumentVersion, ExtractionRun
from rbac_assistant.domain.entities.normative_unit import (
    ExtractionMethod,
    NormativeUnitType,
    StructureStatus,
)
from rbac_assistant.domain.entities.rag import NormativeChunk, SearchFilter
from rbac_assistant.domain.entities.regulatory_document import (
    DocumentStatus,
    DocumentType,
    RegulatoryDocument,
    ValidationPolicy,
    VersionStatus,
)
from rbac_assistant.domain.entities.regulatory_page import (
    PageValidationStatus,
    RegulatoryPage,
    ReviewDecision,
)
from rbac_assistant.domain.ports.validation_port import ParsedUnitRecord
from rbac_assistant.infrastructure.storage.file_document_store import LocalFileDocumentStore
from rbac_assistant.infrastructure.storage.sqlite_document_repository import (
    SqliteDocumentRepository,
)

SMOKE_DOC_ID = "smoke_test_rbac_999"
SMOKE_PDF_BYTES = b"%PDF-1.4\n% Smoke test fixture (not an official regulation)\n%%EOF\n"


def _build_components() -> tuple[SqliteDocumentRepository, LocalFileDocumentStore]:
    settings = load_settings()
    base_dir = settings.storage.documents_dir
    expected_db = base_dir / settings.storage.sqlite_filename
    repo = SqliteDocumentRepository(
        expected_db,
        busy_timeout_ms=settings.storage.sqlite_busy_timeout_ms,
    )
    repo.initialize()
    store = LocalFileDocumentStore(documents_dir=base_dir)
    store.ensure_directories()
    return repo, store


def run_phase(phase: str) -> int:
    repo, store = _build_components()
    file_hash = hashlib.sha256(SMOKE_PDF_BYTES).hexdigest()
    version_id = f"{SMOKE_DOC_ID}_emd_00_{file_hash[:8]}"

    if phase == "write":
        raw_path = store.save_raw_pdf(
            authority="TEST",
            document_type="RBAC",
            regulation_number="999",
            file_hash=file_hash,
            pdf_bytes=SMOKE_PDF_BYTES,
        )
        now = datetime.now(UTC)
        doc = RegulatoryDocument(
            document_id=SMOKE_DOC_ID,
            authority="TEST",
            document_type=DocumentType.RBAC,
            regulation_number="999",
            document_title="Documento Fictício de Smoke Test SQLite",
            document_status=DocumentStatus.DRAFT,
            version_status=VersionStatus.PROPOSED,
            amendment_number="00",
            pdf_filename="smoke_999.pdf",
            file_hash=file_hash,
            imported_at=now,
        )
        version = DocumentVersion(
            version_id=version_id,
            document_id=SMOKE_DOC_ID,
            amendment_number="00",
            source_hash=file_hash,
            version_status=VersionStatus.PROPOSED,
            pdf_filename="smoke_999.pdf",
            pdf_storage_path=str(raw_path),
            page_count=1,
            imported_at=now,
        )
        page_text = "999.1 Aplicabilidade de manutenção.\n(a) Regra fictícia para smoke test."
        page_hash = hashlib.sha256(page_text.encode("utf-8")).hexdigest()
        page = RegulatoryPage(
            page_id=f"{version_id}::p0001",
            version_id=version_id,
            pdf_page_number=1,
            printed_page_number="1/1",
            original_text=page_text,
            normalized_text=page_text,
            extraction_method=ExtractionMethod.NATIVE_PDF,
            validation_status=PageValidationStatus.NATIVE_QUALITY_CHECKED,
            content_hash=page_hash,
        )
        unit = ParsedUnitRecord(
            unit_id=f"{version_id}::999.1",
            version_id=version_id,
            parent_unit_id=None,
            unit_type=NormativeUnitType.SECTION,
            section_number="999.1",
            subsection_number="",
            paragraph_number="",
            item_number="",
            title="Aplicabilidade de manutenção.",
            subpart="",
            original_text=page_text,
            normalized_text=page_text,
            start_page=1,
            end_page=1,
            page_numbers=(1,),
            order_index=0,
            structure_status=StructureStatus.PARSED,
            content_hash=page_hash,
        )
        run = ExtractionRun(
            run_id=f"smoke_{file_hash[:8]}",
            document_id=SMOKE_DOC_ID,
            version_id=version_id,
            started_at=now,
            finished_at=now,
            status="succeeded",
            extraction_version="2.0.0",
            source_hash=file_hash,
            pages_extracted=1,
            units_created=1,
        )
        repo.save_version_ingestion(
            document=doc,
            version=version,
            pages=[page],
            units=[unit],
            tables=[],
            extraction_run=run,
        )

        # Persiste chunk estrutural + entrada FTS5 para validar persistência e reconstrução
        smoke_chunk = NormativeChunk(
            chunk_id=f"{SMOKE_DOC_ID}__{version_id}__999.1__0000__c000",
            document_id=SMOKE_DOC_ID,
            version_id=version_id,
            unit_id=unit.unit_id,
            authority="TEST",
            document_type="RBAC",
            regulation_number="999",
            amendment_number="00",
            version_status=VersionStatus.PROPOSED.value,
            section_number="999.1",
            subsection_number="",
            paragraph_number="",
            item_number="",
            title="Aplicabilidade de manutenção.",
            subpart="",
            start_page=1,
            end_page=1,
            pdf_pages=(1,),
            printed_pages=("1/1",),
            chunk_index=0,
            original_text=page_text,
            search_text=f"TEST RBAC 999 | Seção 999.1\n{page_text}",
            source_hash=file_hash,
            content_hash=page_hash,
            validation_status=PageValidationStatus.NATIVE_QUALITY_CHECKED.value,
        )
        repo.save_chunks(version_id=version_id, chunks=[smoke_chunk])

        # Registra aprovação explícita de página para validar persistência de revisão
        catalog = CatalogService(repo)
        catalog.register_document(
            RegisterDocumentDTO(
                document_id=SMOKE_DOC_ID,
                authority="TEST",
                document_type=DocumentType.RBAC,
                regulation_number="999",
                document_title="Documento Fictício de Smoke Test SQLite",
                amendment_number="00",
                version_status=VersionStatus.PROPOSED,
            )
        )
        ValidatePageUseCase(
            repository=repo,
            catalog_service=catalog,
            validation_service=ValidationService(ValidationPolicy.REVIEW_ON_DEMAND),
            file_store=store,
        ).execute(
            PageReviewRequestDTO(
                document_id=SMOKE_DOC_ID,
                version_id=version_id,
                reviewer="ci_smoke_tester",
                decision=ReviewDecision.APPROVE,
                confirmed=True,
                page_numbers=(1,),
                comments="Verificação de persistência Docker non-root.",
            )
        )
        repo.checkpoint_wal()
        print(f"[OK] SQLite + FTS5 + PDF + revisão gravados com sucesso em {repo.db_path}.")
        return 0

    if phase == "verify":
        if str(repo.db_path) != "/app/data/documents/catalog.sqlite3":
            raise AssertionError(
                f"Caminho inesperado do SQLite no contêiner: {repo.db_path!s}"
            )
        migrations = repo.list_applied_migrations()
        for required_mig in (
            "v001_initial_catalog",
            "v002_fts5_and_chunks",
            "v003_index_strategy_metadata",
        ):
            if required_mig not in migrations:
                raise AssertionError(
                    f"Migração obrigatória ausente em schema_migrations: {required_mig} "
                    f"(aplicadas: {migrations})"
                )

        doc = repo.get_document(SMOKE_DOC_ID)
        if doc is None:
            raise AssertionError("Documento de smoke test não persistiu após reinício.")
        ver = repo.get_version(version_id)
        if ver is None:
            raise AssertionError("Versão de smoke test não persistiu após reinício.")
        raw_pdf = store.get_raw_pdf_path(ver.pdf_storage_path)
        if raw_pdf.read_bytes() != SMOKE_PDF_BYTES:
            raise AssertionError("Conteúdo do PDF original divergiu após reinício.")
        pages = repo.get_pages(version_id)
        if len(pages) != 1 or not pages[0].is_approved_for_literal_transcription:
            raise AssertionError(f"Estado de validação não persistiu: {pages!r}")
        reviews = repo.list_validation_records(pages[0].page_id)
        if not reviews or reviews[0].reviewer != "ci_smoke_tester":
            raise AssertionError(f"Registro de revisão não persistiu: {reviews!r}")

        # Verifica busca FTS5 sem acentos e reconstrução do índice derivado a partir do SQLite
        smoke_filter = SearchFilter(document_id=SMOKE_DOC_ID, only_current_version=False)
        hits_before = repo.search_chunks_fts(query="manutencao", filters=smoke_filter)
        if not hits_before:
            raise AssertionError("Índice FTS5 não retornou o chunk de smoke test após restart.")

        rebuilt_count = repo.rebuild_fts_index(version_id=version_id)
        if rebuilt_count != 1:
            raise AssertionError(f"Reconstrução FTS5 esperava 1 chunk, retornou {rebuilt_count}.")
        hits_after = repo.search_chunks_fts(query="manutencao", filters=smoke_filter)
        if not hits_after:
            raise AssertionError("Busca FTS5 falhou após reconstrução a partir do SQLite.")

        print(
            "[OK] Persistência SQLite (/app/data/documents/catalog.sqlite3), PDF original, "
            "revisão e reconstrução FTS5 verificados pós-restart."
        )
        return 0

    if phase == "cleanup":
        ver = repo.get_version(version_id)
        if ver is not None and ver.pdf_storage_path:
            Path(ver.pdf_storage_path).unlink(missing_ok=True)
        with sqlite3.connect(repo.db_path) as conn:
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.execute(
                "DELETE FROM normative_chunks_fts WHERE document_id = ?;",
                (SMOKE_DOC_ID,),
            )
            conn.execute(
                "DELETE FROM regulatory_documents WHERE document_id = ?;",
                (SMOKE_DOC_ID,),
            )
            conn.commit()
        repo.checkpoint_wal()
        print("[OK] Artefatos de smoke test SQLite e FTS5 removidos.")
        return 0

    raise ValueError(f"Fase inválida: {phase}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test de persistência SQLite no Docker.")
    parser.add_argument("--phase", choices=("write", "verify", "cleanup"), required=True)
    args = parser.parse_args()
    sys.exit(run_phase(args.phase))


if __name__ == "__main__":
    main()
