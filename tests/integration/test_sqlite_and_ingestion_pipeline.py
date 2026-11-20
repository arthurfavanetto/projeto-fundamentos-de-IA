"""Testes de integração de SQLite (WAL, ACID, versões, invalidação), ingestão e extensibilidade."""

from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path

import pytest

from rbac_assistant.application.dto.ingestion_dto import (
    IngestionRequestDTO,
    PageReviewRequestDTO,
    RegisterDocumentDTO,
)
from rbac_assistant.application.services.catalog_service import CatalogService
from rbac_assistant.application.services.ingestion_service import IngestionService
from rbac_assistant.application.services.validation_service import ValidationService
from rbac_assistant.application.use_cases.get_normative_section import GetNormativeSectionUseCase
from rbac_assistant.application.use_cases.ingest_document import IngestDocumentUseCase
from rbac_assistant.application.use_cases.validate_page import ValidatePageUseCase
from rbac_assistant.domain.entities.regulatory_document import (
    DocumentType,
    ValidationPolicy,
    VersionStatus,
)
from rbac_assistant.domain.entities.regulatory_page import (
    PageValidationStatus,
    ReviewDecision,
)
from rbac_assistant.infrastructure.pdf.anac_structure_strategy import AnacRbacStructureStrategy
from rbac_assistant.infrastructure.pdf.generic_structure_strategy import GenericStructureStrategy
from rbac_assistant.infrastructure.pdf.pdfplumber_extractor import PdfPlumberExtractorAdapter
from rbac_assistant.infrastructure.pdf.text_quality_checker import TextQualityChecker
from rbac_assistant.infrastructure.sources.local_file_source_adapter import LocalFileSourceAdapter
from rbac_assistant.infrastructure.storage.file_document_store import LocalFileDocumentStore
from rbac_assistant.infrastructure.storage.sqlite_document_repository import (
    SqliteDocumentRepository,
)
from tests.fixtures.synthetic_pdf_factory import (
    create_synthetic_rbac_pdf,
    create_synthetic_resolution_pdf,
)


def _build_stack(base_dir: Path) -> tuple[
    SqliteDocumentRepository,
    LocalFileDocumentStore,
    CatalogService,
    ValidationService,
    IngestDocumentUseCase,
    ValidatePageUseCase,
    GetNormativeSectionUseCase,
]:
    repo = SqliteDocumentRepository(base_dir / "catalog.sqlite3")
    repo.initialize()
    store = LocalFileDocumentStore(documents_dir=base_dir)
    store.ensure_directories()
    catalog = CatalogService(repo)
    validation = ValidationService(policy=ValidationPolicy.REVIEW_ON_DEMAND)
    ingestion_svc = IngestionService(
        repository=repo,
        file_store=store,
        pdf_extractor=PdfPlumberExtractorAdapter(),
        quality_checker=TextQualityChecker(),
        validation_service=validation,
        structure_strategies=(AnacRbacStructureStrategy(), GenericStructureStrategy()),
        local_source=LocalFileSourceAdapter(),
    )
    return (
        repo,
        store,
        catalog,
        validation,
        IngestDocumentUseCase(ingestion_svc),
        ValidatePageUseCase(
            repository=repo,
            catalog_service=catalog,
            validation_service=validation,
            file_store=store,
        ),
        GetNormativeSectionUseCase(
            repository=repo,
            catalog_service=catalog,
            validation_service=validation,
        ),
    )


@pytest.mark.integration
def test_end_to_end_ingestion_idempotency_review_reopen_and_invalidation(tmp_path: Path) -> None:
    base_dir = tmp_path / "documents"
    repo, _store, _catalog, _val, ingest_uc, validate_uc, get_sec_uc = _build_stack(base_dir)

    pdf_v1 = create_synthetic_rbac_pdf(
        tmp_path / "rbac43_emd05.pdf",
        regulation_number="43",
        amendment_number="05",
        include_blank_ocr_page=True,
    )
    doc_dto = RegisterDocumentDTO(
        document_id="anac_rbac_43",
        authority="ANAC",
        document_type=DocumentType.RBAC,
        regulation_number="43",
        document_title="Manutenção, Manutenção Preventiva, Reconstrução e Alteração",
        amendment_number="05",
        effective_date=date(2021, 5, 26),
        version_status=VersionStatus.CURRENT,
        subject_categories=("manutencao", "aeronavegabilidade"),
    )

    # 1. Primeira ingestão
    res1 = ingest_uc.execute(IngestionRequestDTO(document=doc_dto, source_uri=str(pdf_v1)))
    assert res1.skipped_idempotent is False
    assert res1.pages_extracted == 4
    assert res1.tables_extracted == 1
    assert res1.ocr_pages == (4,)
    assert Path(res1.raw_pdf_path).is_file()
    assert Path(res1.extracted_snapshot_path).is_file()

    # 2. Segunda execução idempotente (não reextrai nem duplica)
    res2 = ingest_uc.execute(IngestionRequestDTO(document=doc_dto, source_uri=str(pdf_v1)))
    assert res2.skipped_idempotent is True
    assert len(repo.list_versions("anac_rbac_43")) == 1

    # 3. Antes da revisão humana, seção multi-página 43.3 (p.2-3) está localizada porém pendente
    before_review = get_sec_uc.execute(document_id="anac_rbac_43", section_number="43.3")
    assert before_review.found is True
    assert before_review.section is not None
    assert before_review.section.pdf_pages == (2, 3)
    assert before_review.section.printed_pages == ("10/25", "11/25")
    assert before_review.section.literal_transcription_allowed is False
    assert before_review.section.pending_pdf_pages == (2, 3)

    # 4. Aprova somente página 2 -> seção 43.3 continua bloqueada pois página 3 ainda pendente
    validate_uc.execute(
        PageReviewRequestDTO(
            document_id="anac_rbac_43",
            reviewer="Arthur Fava",
            decision=ReviewDecision.APPROVE,
            confirmed=True,
            page_numbers=(2,),
            comments="Página 2 conferida.",
        )
    )
    partial_review = get_sec_uc.execute(document_id="anac_rbac_43", section_number="43.3")
    assert partial_review.section is not None
    assert partial_review.section.literal_transcription_allowed is False
    assert partial_review.section.pending_pdf_pages == (3,)

    # 5. Aprova seção 43.3 completa (cobre páginas 2 e 3) -> liberada para transcrição literal!
    rev_result = validate_uc.execute(
        PageReviewRequestDTO(
            document_id="anac_rbac_43",
            reviewer="Arthur Fava",
            decision=ReviewDecision.APPROVE,
            confirmed=True,
            section_number="43.3",
            comments="Seção 43.3 conferida contra PDF oficial.",
        )
    )
    assert len(rev_result.exported_review_paths) == 2
    assert all(Path(p).is_file() for p in rev_result.exported_review_paths)

    after_review = get_sec_uc.execute(document_id="anac_rbac_43", section_number="43.3")
    assert after_review.section is not None
    assert after_review.section.literal_transcription_allowed is True
    assert after_review.section.pending_pdf_pages == ()

    # 6. Verifica modo WAL, integridade referencial, backup online e persistência pós-reabertura
    with sqlite3.connect(repo.db_path) as conn:
        journal = conn.execute("PRAGMA journal_mode;").fetchone()[0]
        assert str(journal).lower() == "wal"

    backup_file = tmp_path / "backups" / "catalog_backup.sqlite3"
    repo.backup_to(backup_file)
    assert backup_file.is_file()

    repo_reopened = SqliteDocumentRepository(base_dir / "catalog.sqlite3")
    reopened_pages = repo_reopened.get_pages(res1.version.version_id, page_numbers=[2, 3, 4])
    assert reopened_pages[0].validation_status == (
        PageValidationStatus.APPROVED_FOR_LITERAL_TRANSCRIPTION
    )
    assert reopened_pages[2].validation_status == PageValidationStatus.PENDING_MANUAL_REVIEW

    # 7. Alteração no texto de uma página na mesma versão invalida a aprovação anterior!
    original_pages = repo.get_pages(res1.version.version_id)
    altered_p2 = original_pages[1]
    from rbac_assistant.domain.entities.regulatory_page import RegulatoryPage

    new_text = altered_p2.original_text + "\n(Alteração simulada no texto da página 2)"
    import hashlib

    modified_pages = list(original_pages)
    modified_pages[1] = RegulatoryPage(
        page_id=altered_p2.page_id,
        version_id=altered_p2.version_id,
        pdf_page_number=altered_p2.pdf_page_number,
        printed_page_number=altered_p2.printed_page_number,
        original_text=new_text,
        normalized_text=new_text,
        extraction_method=altered_p2.extraction_method,
        validation_status=PageValidationStatus.NATIVE_QUALITY_CHECKED,
        content_hash=hashlib.sha256(new_text.encode("utf-8")).hexdigest(),
    )
    invalidations = repo.save_version_ingestion(
        document=res1.document,
        version=res1.version,
        pages=modified_pages,
        units=repo.list_units(res1.version.version_id),
        tables=repo.list_tables(res1.version.version_id),
        extraction_run=repo.list_extraction_runs("anac_rbac_43")[0].__class__(
            run_id="run_reprocess_inv",
            document_id="anac_rbac_43",
            version_id=res1.version.version_id,
            started_at=repo.list_extraction_runs("anac_rbac_43")[0].started_at,
            finished_at=repo.list_extraction_runs("anac_rbac_43")[0].finished_at,
            status="succeeded",
            extraction_version="2.0.0",
            source_hash=res1.version.source_hash,
            pages_extracted=len(modified_pages),
            units_created=res1.units_created,
        ),
    )
    assert len(invalidations) == 1
    assert invalidations[0].decision == ReviewDecision.INVALIDATED_BY_REPROCESSING
    p2_after_inv = repo.get_page(altered_p2.page_id)
    assert p2_after_inv is not None
    assert p2_after_inv.validation_status == PageValidationStatus.NATIVE_QUALITY_CHECKED


@pytest.mark.integration
def test_multidocument_extensibility_second_fictional_regulation_and_versions(
    tmp_path: Path,
) -> None:
    """Comprova que um 2º documento fictício e uma Resolução são ingeridos sem alterar o núcleo."""
    base_dir = tmp_path / "documents"
    repo, _store, catalog, _val, ingest_uc, validate_uc, get_sec_uc = _build_stack(base_dir)

    # Documento 1: RBAC 43 (Emenda 04 histórica + Emenda 05 vigente)
    pdf_43_v4 = create_synthetic_rbac_pdf(
        tmp_path / "rbac43_emd04.pdf",
        regulation_number="43",
        amendment_number="04",
        section_text_override="(a) Redação histórica da Emenda 04 do RBAC 43.",
    )
    pdf_43_v5 = create_synthetic_rbac_pdf(
        tmp_path / "rbac43_emd05.pdf",
        regulation_number="43",
        amendment_number="05",
    )
    ingest_uc.execute(
        IngestionRequestDTO(
            document=RegisterDocumentDTO(
                document_id="anac_rbac_43",
                authority="ANAC",
                document_type=DocumentType.RBAC,
                regulation_number="43",
                document_title="Manutenção Aeronáutica",
                amendment_number="04",
                effective_date=date(2019, 8, 7),
                version_status=VersionStatus.HISTORICAL,
            ),
            source_uri=str(pdf_43_v4),
        )
    )
    ingest_uc.execute(
        IngestionRequestDTO(
            document=RegisterDocumentDTO(
                document_id="anac_rbac_43",
                authority="ANAC",
                document_type=DocumentType.RBAC,
                regulation_number="43",
                document_title="Manutenção Aeronáutica",
                amendment_number="05",
                effective_date=date(2021, 5, 26),
                version_status=VersionStatus.CURRENT,
            ),
            source_uri=str(pdf_43_v5),
        )
    )

    # Documento 2 (FICTÍCIO DE TESTE — não é norma oficial): RBAC 999
    pdf_999 = create_synthetic_rbac_pdf(
        tmp_path / "ficticio_rbac_999.pdf",
        regulation_number="999",
        amendment_number="01",
    )
    res_999 = ingest_uc.execute(
        IngestionRequestDTO(
            document=RegisterDocumentDTO(
                document_id="test_fictional_rbac_999",
                authority="ANAC",
                document_type=DocumentType.RBAC,
                regulation_number="999",
                document_title="Fixture Fictícia de Extensibilidade (Não Oficial)",
                amendment_number="01",
                version_status=VersionStatus.PROPOSED,
                subject_categories=("teste_extensibilidade",),
            ),
            source_uri=str(pdf_999),
        )
    )

    # Documento 3 (Outro tipo documental — Resolução): usa GenericStructureStrategy automaticamente
    pdf_res = create_synthetic_resolution_pdf(tmp_path / "ficticia_res_700.pdf")
    res_700 = ingest_uc.execute(
        IngestionRequestDTO(
            document=RegisterDocumentDTO(
                document_id="test_fictional_res_700",
                authority="ANAC",
                document_type=DocumentType.RESOLUCAO,
                regulation_number="700",
                document_title="Resolução Fictícia de Teste",
                amendment_number="00",
                version_status=VersionStatus.PROPOSED,
            ),
            source_uri=str(pdf_res),
        )
    )

    # Verifica isolamento completo no catálogo
    assert len(catalog.search_documents()) == 3
    assert len(repo.list_versions("anac_rbac_43")) == 2
    assert len(repo.list_versions("test_fictional_rbac_999")) == 1
    assert res_999.units_created > 0
    assert res_700.units_created == 2

    # Aprova seção 999.3 somente no documento fictício — não afeta RBAC 43!
    validate_uc.execute(
        PageReviewRequestDTO(
            document_id="test_fictional_rbac_999",
            reviewer="Engenheiro QA",
            decision=ReviewDecision.APPROVE,
            confirmed=True,
            section_number="999.3",
        )
    )
    sec_999 = get_sec_uc.execute(document_id="test_fictional_rbac_999", section_number="999.3")
    sec_43_current = get_sec_uc.execute(document_id="anac_rbac_43", section_number="43.3")
    sec_43_hist = get_sec_uc.execute(
        document_id="anac_rbac_43", section_number="43.3", amendment_number="04"
    )
    sec_res = get_sec_uc.execute(document_id="test_fictional_res_700", section_number="Art. 1º")

    assert sec_999.section is not None and sec_999.section.literal_transcription_allowed is True
    assert sec_43_current.section is not None
    assert sec_43_current.section.amendment_number == "05"
    assert sec_43_current.section.literal_transcription_allowed is False
    assert sec_43_hist.section is not None
    assert sec_43_hist.section.amendment_number == "04"
    assert "Emenda 04" in sec_43_hist.section.original_text
    assert sec_res.section is not None
    assert "Art. 1º" in sec_res.section.original_text
