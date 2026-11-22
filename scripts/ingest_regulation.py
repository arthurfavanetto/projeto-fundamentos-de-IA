"""CLI administrativo genérico para cadastro e ingestão de documentos regulatórios."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml

from rbac_assistant.application.dto.ingestion_dto import (
    IngestionRequestDTO,
    RegisterDocumentDTO,
)
from rbac_assistant.application.services.ingestion_service import IngestionService
from rbac_assistant.application.services.validation_service import ValidationService
from rbac_assistant.application.use_cases.ingest_document import IngestDocumentUseCase
from rbac_assistant.config.logging_config import configure_cli_logging
from rbac_assistant.config.settings import load_settings
from rbac_assistant.domain.entities.regulatory_document import (
    DocumentStatus,
    DocumentType,
    ValidationPolicy,
    VersionStatus,
)
from rbac_assistant.infrastructure.pdf.anac_structure_strategy import AnacRbacStructureStrategy
from rbac_assistant.infrastructure.pdf.generic_structure_strategy import GenericStructureStrategy
from rbac_assistant.infrastructure.pdf.ocr_adapter import TesseractOcrAdapter
from rbac_assistant.infrastructure.pdf.pdfplumber_extractor import PdfPlumberExtractorAdapter
from rbac_assistant.infrastructure.pdf.text_quality_checker import TextQualityChecker
from rbac_assistant.infrastructure.sources.anac_source_adapter import AnacHttpSourceAdapter
from rbac_assistant.infrastructure.sources.local_file_source_adapter import LocalFileSourceAdapter
from rbac_assistant.infrastructure.storage.file_document_store import LocalFileDocumentStore
from rbac_assistant.infrastructure.storage.sqlite_document_repository import (
    SqliteDocumentRepository,
)


def _parse_date(val: Any) -> date | None:
    if not val or str(val).upper().startswith("PENDING"):
        return None
    return date.fromisoformat(str(val))


def _parse_datetime(val: Any) -> datetime | None:
    if not val or str(val).upper().startswith("PENDING"):
        return None
    return datetime.fromisoformat(str(val).replace("Z", "+00:00"))


def _load_catalog_entry(catalog_yaml: Path, document_id: str) -> dict[str, Any]:
    data = yaml.safe_load(catalog_yaml.read_text(encoding="utf-8")) or {}
    for entry in data.get("regulations", []):
        if str(entry.get("document_id")) == document_id:
            return dict(entry)
    raise ValueError(f"Documento '{document_id}' não encontrado em {catalog_yaml}.")


def build_ingestion_use_case(documents_dir: Path | None = None) -> IngestDocumentUseCase:
    """Monta o caso de uso de ingestão com adaptadores concretos."""
    settings = load_settings()
    configure_cli_logging(
        log_level=settings.log_level,
        redact_secrets=settings.security.audit.redact_secrets,
    )
    base_dir = documents_dir or settings.storage.documents_dir
    sqlite_path = base_dir / settings.storage.sqlite_filename

    repository = SqliteDocumentRepository(
        sqlite_path,
        busy_timeout_ms=settings.storage.sqlite_busy_timeout_ms,
    )
    repository.initialize()

    file_store = LocalFileDocumentStore(
        documents_dir=base_dir,
        raw_pdfs_subdir=settings.storage.raw_pdfs_subdir,
        extracted_subdir=settings.storage.extracted_subdir,
        reviews_subdir=settings.storage.reviews_subdir,
    )
    file_store.ensure_directories()

    validation_service = ValidationService(
        policy=ValidationPolicy(settings.extraction.validation_policy)
    )
    service = IngestionService(
        repository=repository,
        file_store=file_store,
        pdf_extractor=PdfPlumberExtractorAdapter(),
        quality_checker=TextQualityChecker(),
        validation_service=validation_service,
        structure_strategies=(AnacRbacStructureStrategy(), GenericStructureStrategy()),
        local_source=LocalFileSourceAdapter(max_pdf_size_mb=settings.extraction.max_pdf_size_mb),
        http_source=AnacHttpSourceAdapter(max_pdf_size_mb=settings.extraction.max_pdf_size_mb),
        ocr_adapter=TesseractOcrAdapter(),
        min_native_chars=settings.extraction.min_native_chars_per_page,
        max_pdf_pages=settings.extraction.max_pdf_pages,
        ocr_language=settings.extraction.ocr_language,
        extraction_version=settings.extraction.extraction_version,
    )
    return IngestDocumentUseCase(service)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ingestão genérica de documentos regulatórios no catálogo SQLite."
    )
    parser.add_argument("--document-id", required=True, help="Identificador único do documento.")
    parser.add_argument(
        "--catalog-yaml",
        type=Path,
        default=Path("config/regulations_catalog.yaml"),
        help="Manifesto YAML de metadados (opcional se informar flags explícitas).",
    )
    parser.add_argument("--pdf-path", type=Path, help="Importação manual de PDF local.")
    parser.add_argument("--source-url", help="Download direto por URL oficial HTTPS.")
    parser.add_argument("--authority", help="Autoridade emissora (ex.: ANAC).")
    parser.add_argument("--document-type", help="Tipo (RBAC, IS, RESOLUCAO, etc.).")
    parser.add_argument("--regulation-number", help="Número do regulamento (ex.: 43, 145).")
    parser.add_argument("--title", help="Título oficial do documento.")
    parser.add_argument("--amendment", help="Número da emenda/versão.")
    parser.add_argument("--version-status", default="unknown", help="Status da versão.")
    parser.add_argument(
        "--documents-dir",
        type=Path,
        help="Sobrescreve o diretório de dados (padrão: config/settings.yaml).",
    )
    parser.add_argument("--force", action="store_true", help="Força reprocessamento controlado.")
    parser.add_argument(
        "--allow-ocr", action="store_true", help="Habilita OCR opcional se presente."
    )
    args = parser.parse_args(argv)

    entry: dict[str, Any] = {}
    if args.catalog_yaml and args.catalog_yaml.is_file():
        try:
            entry = _load_catalog_entry(args.catalog_yaml, args.document_id)
        except ValueError:
            entry = {}

    authority = args.authority or str(entry.get("authority", ""))
    doc_type = DocumentType(args.document_type or str(entry.get("document_type", "RBAC")))
    reg_num = args.regulation_number or str(entry.get("regulation_number", ""))
    title = args.title or str(entry.get("document_title", ""))
    amendment = args.amendment or str(entry.get("amendment_number", "00"))
    raw_ver_status = (
        args.version_status
        if args.version_status != "unknown"
        else str(entry.get("version_status", "unknown"))
    )
    ver_status = VersionStatus(raw_ver_status)

    if not authority or not reg_num or not title:
        parser.error(
            "Metadados obrigatórios ausentes: forneça entrada em --catalog-yaml ou informe "
            "--authority, --regulation-number e --title."
        )

    if args.pdf_path:
        source_kind = "local"
        source_uri = str(args.pdf_path)
    else:
        source_kind = "http"
        source_uri = args.source_url or str(entry.get("source_url", ""))
        if not source_uri:
            parser.error("Informe --pdf-path ou --source-url (ou source_url no catálogo YAML).")

    dto = RegisterDocumentDTO(
        document_id=args.document_id,
        authority=authority,
        document_type=doc_type,
        regulation_number=reg_num,
        document_title=title,
        amendment_number=amendment,
        resolution_number=str(entry.get("resolution_number", "")),
        publication_date=_parse_date(entry.get("publication_date")),
        effective_date=_parse_date(entry.get("effective_date")),
        expiration_date=_parse_date(entry.get("expiration_date")),
        document_status=DocumentStatus(str(entry.get("document_status", "active"))),
        version_status=ver_status,
        official_url=str(entry.get("official_url", "")),
        source_url=args.source_url or str(entry.get("source_url", "")),
        pdf_filename=str(entry.get("pdf_filename", "")),
        last_verified_at=_parse_datetime(entry.get("consulted_at")),
        subject_categories=tuple(str(c) for c in entry.get("subject_categories", [])),
        document_description=str(entry.get("document_description", "")).strip(),
    )

    use_case = build_ingestion_use_case(documents_dir=args.documents_dir)
    result = use_case.execute(
        IngestionRequestDTO(
            document=dto,
            source_uri=source_uri,
            source_kind=source_kind,
            force_reprocess=args.force,
            allow_ocr=args.allow_ocr,
        )
    )

    print(
        json.dumps(
            {
                "document_id": result.document.document_id,
                "version_id": result.version.version_id,
                "amendment_number": result.version.amendment_number,
                "version_status": result.version.version_status.value,
                "source_hash": result.version.source_hash,
                "skipped_idempotent": result.skipped_idempotent,
                "pages_extracted": result.pages_extracted,
                "units_created": result.units_created,
                "tables_extracted": result.tables_extracted,
                "ocr_pages": list(result.ocr_pages),
                "invalidated_reviews": len(result.invalidated_reviews),
                "raw_pdf_path": result.raw_pdf_path,
                "extracted_snapshot_path": result.extracted_snapshot_path,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
