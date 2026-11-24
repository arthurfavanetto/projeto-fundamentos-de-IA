"""CLI de inspeção do catálogo SQLite, versões, páginas e reconstrução de seções."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from rbac_assistant.application.services.catalog_service import CatalogService
from rbac_assistant.application.services.validation_service import ValidationService
from rbac_assistant.application.use_cases.get_normative_section import GetNormativeSectionUseCase
from rbac_assistant.config.logging_config import configure_cli_logging
from rbac_assistant.config.settings import load_settings
from rbac_assistant.domain.entities.regulatory_document import ValidationPolicy
from rbac_assistant.infrastructure.storage.sqlite_document_repository import (
    SqliteDocumentRepository,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspeciona documentos, versões, estados de validação e seções reconstruídas."
    )
    parser.add_argument("--document-id", help="Inspeciona um documento específico.")
    parser.add_argument("--version-id", help="Versão específica do documento.")
    parser.add_argument("--section", help="Reconstrói e exibe uma seção normativa (ex.: 43.3).")
    parser.add_argument(
        "--policy",
        choices=("strict_all_pages", "review_on_demand", "native_auto_ocr_manual"),
        help="Sobrescreve a política de validação para inspeção.",
    )
    parser.add_argument("--documents-dir", type=Path, help="Diretório do catálogo SQLite.")
    args = parser.parse_args(argv)

    settings = load_settings()
    configure_cli_logging(
        log_level=settings.log_level,
        redact_secrets=settings.security.audit.redact_secrets,
    )
    base_dir = args.documents_dir or settings.storage.documents_dir
    repository = SqliteDocumentRepository(
        base_dir / settings.storage.sqlite_filename,
        busy_timeout_ms=settings.storage.sqlite_busy_timeout_ms,
    )
    repository.initialize()
    catalog = CatalogService(repository)
    active_policy = ValidationPolicy(args.policy or settings.extraction.validation_policy)
    validation = ValidationService(policy=active_policy)

    if not args.document_id:
        docs = catalog.search_documents()
        payload = [
            {
                "document_id": d.document_id,
                "authority": d.authority,
                "document_type": d.document_type.value,
                "regulation_number": d.regulation_number,
                "amendment_number": d.amendment_number,
                "version_status": d.version_status.value,
                "document_title": d.document_title,
            }
            for d in docs
        ]
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.section:
        lookup = GetNormativeSectionUseCase(
            repository=repository,
            catalog_service=catalog,
            validation_service=validation,
        ).execute(
            document_id=args.document_id,
            section_number=args.section,
            version_id=args.version_id,
            policy=active_policy,
        )
        if not lookup.found or lookup.section is None:
            print(
                json.dumps(
                    {
                        "found": False,
                        "blocked_by_document_approval": lookup.blocked_by_document_approval,
                        "pending_pdf_pages": list(lookup.pending_pdf_pages),
                        "rejected_pdf_pages": list(lookup.rejected_pdf_pages),
                        "missing_pdf_pages": list(lookup.missing_pdf_pages),
                        "message": lookup.message,
                    },
                    ensure_ascii=False,
                )
            )
            return 1
        sec = lookup.section
        print(
            json.dumps(
                {
                    "found": True,
                    "document_id": sec.document_id,
                    "version_id": sec.version_id,
                    "regulation_number": sec.regulation_number,
                    "amendment_number": sec.amendment_number,
                    "version_status": sec.version_status,
                    "section_number": sec.section_number,
                    "title": sec.title,
                    "pdf_pages": list(sec.pdf_pages),
                    "printed_pages": list(sec.printed_pages),
                    "validation_status": sec.validation_status,
                    "structure_status": sec.structure_status.value,
                    "literal_transcription_allowed": sec.literal_transcription_allowed,
                    "auxiliary_unreviewed_available": lookup.auxiliary_unreviewed_available,
                    "pending_pdf_pages": list(sec.pending_pdf_pages),
                    "ocr_blocked_pdf_pages": list(sec.ocr_blocked_pdf_pages),
                    "warning_message": sec.warning_message,
                    "original_text": sec.original_text,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    doc = catalog.get_document(args.document_id)
    versions = catalog.list_versions(args.document_id)
    version_summaries = []
    for ver in versions:
        pages = repository.get_pages(ver.version_id)
        units = repository.list_units(ver.version_id)
        status_counts = Counter(p.validation_status.value for p in pages)
        version_summaries.append(
            {
                "version_id": ver.version_id,
                "amendment_number": ver.amendment_number,
                "version_status": ver.version_status.value,
                "source_hash": ver.source_hash,
                "page_count": len(pages),
                "unit_count": len(units),
                "validation_breakdown": dict(status_counts),
            }
        )

    print(
        json.dumps(
            {
                "document_id": doc.document_id,
                "authority": doc.authority,
                "document_type": doc.document_type.value,
                "regulation_number": doc.regulation_number,
                "document_title": doc.document_title,
                "versions": version_summaries,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
