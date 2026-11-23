"""CLI de revisão humana explícita de páginas, intervalos ou seções extraídas.

Exemplos:
    uv run python scripts/validate_extraction.py --document-id anac_rbac_43 --page 3
    uv run python scripts/validate_extraction.py --document-id anac_rbac_43 --pages 3-8
    uv run python scripts/validate_extraction.py --document-id anac_rbac_43 --section 43.3
    uv run python scripts/validate_extraction.py --document-id anac_rbac_43 --section 43.3 \
        --decision approve --reviewer "Arthur Fava" --comments "Conferido com PDF ANAC" --confirm

IMPORTANTE: Não existe opção de aprovação silenciosa de todas as páginas.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rbac_assistant.application.dto.ingestion_dto import PageReviewRequestDTO
from rbac_assistant.application.services.catalog_service import CatalogService
from rbac_assistant.application.services.validation_service import ValidationService
from rbac_assistant.application.use_cases.validate_page import ValidatePageUseCase
from rbac_assistant.config.settings import load_settings
from rbac_assistant.domain.entities.regulatory_document import ValidationPolicy
from rbac_assistant.domain.entities.regulatory_page import ReviewDecision
from rbac_assistant.infrastructure.audit.jsonl_audit_adapter import JsonlAuditAdapter
from rbac_assistant.infrastructure.storage.file_document_store import LocalFileDocumentStore
from rbac_assistant.infrastructure.storage.sqlite_document_repository import (
    SqliteDocumentRepository,
)


def parse_page_range(spec: str) -> list[int]:
    """Interpreta especificações como '3', '3-8' ou '2,4-6' em lista ordenada de inteiros."""
    numbers: set[int] = set()
    for part in spec.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_s, end_s = token.split("-", maxsplit=1)
            start, end = int(start_s.strip()), int(end_s.strip())
            if start < 1 or end < start:
                raise ValueError(f"Intervalo de páginas inválido: {spec!r}")
            numbers.update(range(start, end + 1))
        else:
            page_num = int(token)
            if page_num < 1:
                raise ValueError(f"Número de página inválido: {page_num}")
            numbers.add(page_num)
    return sorted(numbers)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspeção e revisão humana explícita de extração normativa."
    )
    parser.add_argument("--document-id", required=True, help="Identificador do documento.")
    parser.add_argument("--version-id", help="Versão específica (padrão: versão vigente).")
    parser.add_argument("--page", type=int, help="Número físico da página do PDF (ex.: 3).")
    parser.add_argument("--pages", help="Intervalo de páginas físicas (ex.: 3-8 ou 2,4-6).")
    parser.add_argument("--section", help="Número da seção normativa (ex.: 43.3).")
    parser.add_argument(
        "--decision",
        choices=("approve", "reject"),
        help="Registra aprovação ou reprovação explícita após inspeção.",
    )
    parser.add_argument("--reviewer", help="Nome/identificador do responsável autenticado.")
    parser.add_argument("--comments", default="", help="Observações da conferência com o PDF.")
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Confirma explicitamente a decisão de revisão (obrigatório em modo não-interativo).",
    )
    parser.add_argument(
        "--documents-dir",
        type=Path,
        help="Sobrescreve o diretório do catálogo SQLite.",
    )
    parser.add_argument(
        "--audit-logs-dir",
        type=Path,
        help="Sobrescreve o diretório de logs JSONL de auditoria.",
    )
    args = parser.parse_args(argv)

    target_pages: list[int] = []
    if args.page is not None:
        target_pages.append(args.page)
    if args.pages:
        target_pages.extend(parse_page_range(args.pages))

    if not target_pages and not args.section:
        parser.error("Informe --page, --pages ou --section. Aprovação em massa cega é proibida.")

    settings = load_settings()
    base_dir = args.documents_dir or settings.storage.documents_dir
    resolved_audit_dir = args.audit_logs_dir or (
        args.documents_dir / "audit_logs"
        if args.documents_dir is not None
        else settings.storage.audit_logs_dir
    )
    repository = SqliteDocumentRepository(
        base_dir / settings.storage.sqlite_filename,
        busy_timeout_ms=settings.storage.sqlite_busy_timeout_ms,
    )
    repository.initialize()
    file_store = LocalFileDocumentStore(documents_dir=base_dir)
    catalog_service = CatalogService(repository)
    validation_service = ValidationService(
        policy=ValidationPolicy(settings.extraction.validation_policy)
    )

    version = catalog_service.resolve_version(
        document_id=args.document_id,
        version_id=args.version_id,
    )

    display_pages = list(target_pages)
    if args.section:
        units = repository.get_units_by_section(
            version_id=version.version_id,
            section_number=args.section,
        )
        if not units:
            print(f"[ERRO] Seção '{args.section}' não encontrada em {version.version_id}.")
            return 1
        display_pages.extend(p for u in units for p in u.page_numbers)

    unique_display = sorted(set(display_pages))
    pages = repository.get_pages(version.version_id, page_numbers=unique_display)

    print(f"Documento: {args.document_id} | Versão: {version.version_id}")
    print(f"PDF original armazenado: {version.pdf_storage_path}")
    print("=" * 80)
    for page in pages:
        print(
            f"[PDF p.{page.pdf_page_number} | Impressa: {page.printed_page_number or 'N/I'} | "
            f"Método: {page.extraction_method.value} | Status: {page.validation_status.value} | "
            f"Hash: {page.content_hash[:16]}]"
        )
        print("-" * 80)
        print(page.original_text)
        print("=" * 80)

    if args.decision is None:
        print(
            "Modo somente inspeção. Para registrar decisão humana, informe "
            "--decision approve|reject --reviewer <nome> --confirm."
        )
        return 0

    if not args.reviewer or not args.reviewer.strip():
        parser.error("Informe --reviewer <responsável> para registrar a decisão.")

    confirmed = bool(args.confirm)
    if not confirmed:
        answer = input(
            f"Confirma '{args.decision}' para {len(pages)} página(s) "
            f"{unique_display} como '{args.reviewer}'? [s/N]: "
        ).strip()
        confirmed = answer.lower() in ("s", "sim", "y", "yes")
        if not confirmed:
            print("Operação cancelada pelo revisor.")
            return 1

    audit_adapter = JsonlAuditAdapter.from_settings(settings, logs_dir=resolved_audit_dir)
    use_case = ValidatePageUseCase(
        repository=repository,
        catalog_service=catalog_service,
        validation_service=validation_service,
        file_store=file_store,
        audit_port=audit_adapter,
    )
    result = use_case.execute(
        PageReviewRequestDTO(
            document_id=args.document_id,
            version_id=version.version_id,
            reviewer=args.reviewer,
            decision=ReviewDecision(args.decision),
            confirmed=confirmed,
            page_numbers=tuple(unique_display),
            comments=args.comments,
        )
    )
    print(
        f"[OK] Decisão '{args.decision}' registrada por '{args.reviewer}' em "
        f"{len(result.updated_pages)} página(s) (versão {result.version_id})."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
