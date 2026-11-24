"""Serviço genérico e idempotente de ingestão documental regulatória."""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime

from rbac_assistant.application.dto.ingestion_dto import (
    IngestionRequestDTO,
    IngestionResultDTO,
)
from rbac_assistant.application.services.validation_service import ValidationService
from rbac_assistant.domain.entities.document_version import DocumentVersion, ExtractionRun
from rbac_assistant.domain.entities.normative_unit import (
    ExtractionMethod,
    NormativeTable,
)
from rbac_assistant.domain.entities.regulatory_document import RegulatoryDocument
from rbac_assistant.domain.entities.regulatory_page import (
    ExtractionMetadata,
    RegulatoryPage,
    compute_page_content_hash,
)
from rbac_assistant.domain.exceptions import (
    AuditWriteError,
    OcrUnavailableError,
    SourceFetchError,
)
from rbac_assistant.domain.ports.audit_port import AuditEvent, AuditPort
from rbac_assistant.domain.ports.document_repository_port import DocumentRepositoryPort
from rbac_assistant.domain.ports.document_source_port import DocumentSourcePort
from rbac_assistant.domain.ports.file_store_port import FileStorePort
from rbac_assistant.domain.ports.ocr_port import OCRPort
from rbac_assistant.domain.ports.pdf_extractor_port import PDFExtractorPort
from rbac_assistant.domain.ports.validation_port import (
    StructureParserStrategy,
    TextQualityCheckerPort,
)


def build_version_id(document_id: str, amendment_number: str, source_hash: str) -> str:
    """Gera identificador determinístico para a versão normativa sem colisão entre hashes."""
    clean_amend = amendment_number.strip().lower().replace(" ", "_") or "00"
    return f"{document_id}_emd_{clean_amend}_{source_hash[:8]}"


class IngestionService:
    """Pipeline documental agnóstico a regulamento com persistência ACID em SQLite."""

    def __init__(
        self,
        *,
        repository: DocumentRepositoryPort,
        file_store: FileStorePort,
        pdf_extractor: PDFExtractorPort,
        quality_checker: TextQualityCheckerPort,
        validation_service: ValidationService,
        structure_strategies: Sequence[StructureParserStrategy],
        local_source: DocumentSourcePort,
        http_source: DocumentSourcePort | None = None,
        ocr_adapter: OCRPort | None = None,
        audit_port: AuditPort | None = None,
        min_native_chars: int = 50,
        max_pdf_pages: int = 1000,
        ocr_language: str = "por",
        extraction_version: str = "2.0.0",
    ) -> None:
        self._repository = repository
        self._file_store = file_store
        self._pdf_extractor = pdf_extractor
        self._quality_checker = quality_checker
        self._validation_service = validation_service
        self._structure_strategies = tuple(structure_strategies)
        self._local_source = local_source
        self._http_source = http_source
        self._ocr_adapter = ocr_adapter
        self._audit_port = audit_port
        self._min_native_chars = min_native_chars
        self._max_pdf_pages = max_pdf_pages
        self._ocr_language = ocr_language
        self._extraction_version = extraction_version


    def _select_source(self, source_kind: str) -> DocumentSourcePort:
        if source_kind == "local":
            return self._local_source
        if source_kind == "http":
            if self._http_source is None:
                raise SourceFetchError("Adaptador HTTP não configurado para download remoto.")
            return self._http_source
        raise SourceFetchError(f"Tipo de fonte desconhecido: {source_kind!r}")

    def _select_strategy(self, document: RegulatoryDocument) -> StructureParserStrategy:
        for strategy in self._structure_strategies:
            if strategy.supports(
                authority=document.authority,
                document_type=document.document_type,
            ):
                return strategy
        raise ValueError(
            f"Nenhuma estratégia estrutural disponível para "
            f"{document.authority}/{document.document_type.value}."
        )

    def ingest(self, request: IngestionRequestDTO) -> IngestionResultDTO:
        """Executa o pipeline completo de ingestão idempotente de um documento regulatório."""
        started_at = datetime.now(UTC)
        run_id = f"run_{uuid.uuid4().hex[:12]}"
        doc_dto = request.document

        source = self._select_source(request.source_kind)
        fetched = source.fetch(request.source_uri, filename=doc_dto.pdf_filename or None)

        amendment = doc_dto.amendment_number.strip() or "00"
        existing_version = self._repository.get_version_by_hash(
            document_id=doc_dto.document_id,
            amendment_number=amendment,
            source_hash=fetched.file_hash,
        )

        # Idempotência: evita reextração desnecessária quando mesmo hash já foi ingerido
        if existing_version is not None and not request.force_reprocess:
            existing_doc = self._repository.get_document(doc_dto.document_id)
            assert existing_doc is not None
            pages_existing = self._repository.get_pages(existing_version.version_id)
            units_existing = self._repository.list_units(existing_version.version_id)
            tables_existing = self._repository.list_tables(existing_version.version_id)
            self._repository.record_extraction_run(
                ExtractionRun(
                    run_id=run_id,
                    document_id=doc_dto.document_id,
                    version_id=existing_version.version_id,
                    started_at=started_at,
                    finished_at=datetime.now(UTC),
                    status="skipped_idempotent",
                    extraction_version=self._extraction_version,
                    source_hash=fetched.file_hash,
                    pages_extracted=len(pages_existing),
                    units_created=len(units_existing),
                )
            )
            if self._audit_port is not None:
                self._audit_port.record(
                    AuditEvent(
                        timestamp=datetime.now(UTC),
                        action="document_ingested",
                        status="skipped_idempotent",
                        actor_id="system",
                        details={
                            "document_id": existing_doc.document_id,
                            "version_id": existing_version.version_id,
                            "amendment_number": existing_version.amendment_number,
                            "pages_extracted": len(pages_existing),
                            "units_created": len(units_existing),
                            "source_hash": fetched.file_hash,
                        },
                    )
                )
            return IngestionResultDTO(
                document=existing_doc,
                version=existing_version,
                skipped_idempotent=True,
                pages_extracted=len(pages_existing),
                units_created=len(units_existing),
                tables_extracted=len(tables_existing),
                ocr_pages=tuple(p.pdf_page_number for p in pages_existing if p.is_ocr),
                raw_pdf_path=existing_version.pdf_storage_path,
            )

        version_id = (
            existing_version.version_id
            if existing_version is not None
            else build_version_id(doc_dto.document_id, amendment, fetched.file_hash)
        )

        try:
            raw_pdf_path = self._file_store.save_raw_pdf(
                authority=doc_dto.authority,
                document_type=doc_dto.document_type.value,
                regulation_number=doc_dto.regulation_number,
                file_hash=fetched.file_hash,
                pdf_bytes=fetched.pdf_bytes,
            )

            extracted_pages = self._pdf_extractor.extract_pages(
                raw_pdf_path,
                min_native_chars=self._min_native_chars,
                max_pages=self._max_pdf_pages,
            )

            now = datetime.now(UTC)
            regulatory_pages: list[RegulatoryPage] = []
            all_tables: list[NormativeTable] = []
            ocr_page_numbers: list[int] = []

            for ext_page in extracted_pages:
                orig_text = ext_page.original_text
                norm_text = ext_page.normalized_text
                method = ExtractionMethod.NATIVE_PDF

                if ext_page.needs_ocr:
                    ocr_page_numbers.append(ext_page.pdf_page_number)
                    if (
                        request.allow_ocr
                        and self._ocr_adapter is not None
                        and self._ocr_adapter.is_available()
                    ):
                        try:
                            ocr_res = self._ocr_adapter.extract_page_text(
                                raw_pdf_path,
                                pdf_page_number=ext_page.pdf_page_number,
                                language=self._ocr_language,
                            )
                            orig_text = ocr_res.original_text
                            norm_text = ocr_res.normalized_text
                            method = ExtractionMethod.OCR_FALLBACK
                        except OcrUnavailableError:
                            method = ExtractionMethod.OCR_REQUIRED_UNAVAILABLE
                    else:
                        method = ExtractionMethod.OCR_REQUIRED_UNAVAILABLE

                content_hash = compute_page_content_hash(
                    orig_text,
                    tuple(table.rows for table in ext_page.tables),
                )

                quality = self._quality_checker.evaluate(
                    original_text=orig_text,
                    normalized_text=norm_text,
                    min_chars=self._min_native_chars,
                )
                initial_status = self._validation_service.determine_initial_page_status(
                    extraction_method=method,
                    quality_metrics=quality,
                )

                page_id = f"{version_id}::p{ext_page.pdf_page_number:04d}"
                metadata = ExtractionMetadata(
                    extraction_method=method,
                    extraction_timestamp=now,
                    extraction_version=self._extraction_version,
                    content_hash=content_hash,
                    quality_metrics=quality,
                    page_width=ext_page.page_width,
                    page_height=ext_page.page_height,
                    table_count=len(ext_page.tables),
                    image_count=ext_page.image_count,
                )
                regulatory_pages.append(
                    RegulatoryPage(
                        page_id=page_id,
                        version_id=version_id,
                        pdf_page_number=ext_page.pdf_page_number,
                        printed_page_number=ext_page.printed_page_number,
                        original_text=orig_text,
                        normalized_text=norm_text,
                        extraction_method=method,
                        validation_status=initial_status,
                        content_hash=content_hash,
                        extraction_metadata=metadata,
                        coordinates_summary=ext_page.coordinates_summary,
                    )
                )

                for t_idx, ext_table in enumerate(ext_page.tables):
                    t_hash = hashlib.sha256(
                        f"{version_id}:{ext_page.pdf_page_number}:{t_idx}".encode()
                    ).hexdigest()[:12]
                    all_tables.append(
                        NormativeTable(
                            table_id=f"tbl_{t_hash}",
                            version_id=version_id,
                            pdf_page_number=ext_page.pdf_page_number,
                            rows=ext_table.rows,
                        )
                    )

            document = RegulatoryDocument(
                document_id=doc_dto.document_id,
                authority=doc_dto.authority,
                document_type=doc_dto.document_type,
                regulation_number=doc_dto.regulation_number,
                document_title=doc_dto.document_title,
                document_status=doc_dto.document_status,
                version_status=doc_dto.version_status,
                amendment_number=amendment,
                publication_date=doc_dto.publication_date,
                effective_date=doc_dto.effective_date,
                expiration_date=doc_dto.expiration_date,
                official_url=doc_dto.official_url,
                source_url=doc_dto.source_url or fetched.source_uri,
                pdf_filename=fetched.filename,
                file_hash=fetched.file_hash,
                imported_at=now,
                last_verified_at=doc_dto.last_verified_at or now,
                subject_categories=doc_dto.subject_categories,
                document_description=doc_dto.document_description,
            )
            version = DocumentVersion(
                version_id=version_id,
                document_id=document.document_id,
                amendment_number=amendment,
                resolution_number=doc_dto.resolution_number,
                publication_date=doc_dto.publication_date,
                effective_date=doc_dto.effective_date,
                expiration_date=doc_dto.expiration_date,
                version_status=doc_dto.version_status,
                official_url=doc_dto.official_url,
                source_url=doc_dto.source_url or fetched.source_uri,
                pdf_filename=fetched.filename,
                pdf_storage_path=str(raw_pdf_path),
                source_hash=fetched.file_hash,
                page_count=len(regulatory_pages),
                imported_at=now,
                last_verified_at=doc_dto.last_verified_at or now,
            )

            strategy = self._select_strategy(document)
            parsed_units = strategy.parse(
                version_id=version_id,
                regulation_number=document.regulation_number,
                pages=regulatory_pages,
                tables=all_tables,
            )

            extraction_run = ExtractionRun(
                run_id=run_id,
                document_id=document.document_id,
                version_id=version_id,
                started_at=started_at,
                finished_at=datetime.now(UTC),
                status="succeeded",
                extraction_version=self._extraction_version,
                source_hash=fetched.file_hash,
                pages_extracted=len(regulatory_pages),
                units_created=len(parsed_units),
            )

            invalidated = self._repository.save_version_ingestion(
                document=document,
                version=version,
                pages=regulatory_pages,
                units=parsed_units,
                tables=all_tables,
                extraction_run=extraction_run,
            )

            snapshot_path = self._file_store.save_extracted_snapshot(
                version_id=version_id,
                payload={
                    "document_id": document.document_id,
                    "version_id": version_id,
                    "amendment_number": amendment,
                    "source_hash": fetched.file_hash,
                    "extraction_version": self._extraction_version,
                    "extracted_at": now.isoformat(),
                    "page_count": len(regulatory_pages),
                    "unit_count": len(parsed_units),
                    "table_count": len(all_tables),
                    "pages": [
                        {
                            "pdf_page_number": p.pdf_page_number,
                            "printed_page_number": p.printed_page_number,
                            "extraction_method": p.extraction_method.value,
                            "validation_status": p.validation_status.value,
                            "content_hash": p.content_hash,
                        }
                        for p in regulatory_pages
                    ],
                },
            )

            if self._audit_port is not None:
                self._audit_port.record(
                    AuditEvent(
                        timestamp=datetime.now(UTC),
                        action="document_ingested",
                        status="success",
                        actor_id="system",
                        details={
                            "document_id": document.document_id,
                            "version_id": version.version_id,
                            "amendment_number": amendment,
                            "pages_extracted": len(regulatory_pages),
                            "units_created": len(parsed_units),
                            "tables_extracted": len(all_tables),
                            "ocr_pages_count": len(ocr_page_numbers),
                            "source_hash": fetched.file_hash,
                        },
                    )
                )

            return IngestionResultDTO(
                document=document,
                version=version,
                skipped_idempotent=False,
                pages_extracted=len(regulatory_pages),
                units_created=len(parsed_units),
                tables_extracted=len(all_tables),
                ocr_pages=tuple(ocr_page_numbers),
                invalidated_reviews=tuple(invalidated),
                raw_pdf_path=str(raw_pdf_path),
                extracted_snapshot_path=str(snapshot_path),
            )
        except AuditWriteError:
            raise
        except Exception as exc:
            # Registra falha no histórico se o documento já existir no catálogo
            if self._repository.get_document(doc_dto.document_id) is not None:
                self._repository.record_extraction_run(
                    ExtractionRun(
                        run_id=run_id,
                        document_id=doc_dto.document_id,
                        version_id=None,
                        started_at=started_at,
                        finished_at=datetime.now(UTC),
                        status="failed",
                        extraction_version=self._extraction_version,
                        source_hash=fetched.file_hash,
                        error_message=str(exc),
                    )
                )
            if self._audit_port is not None:
                self._audit_port.record(
                    AuditEvent(
                        timestamp=datetime.now(UTC),
                        action="document_ingested",
                        status="failed",
                        actor_id="system",
                        details={
                            "document_id": doc_dto.document_id,
                            "amendment_number": amendment,
                            "error_type": type(exc).__name__,
                        },
                    )
                )
            raise
