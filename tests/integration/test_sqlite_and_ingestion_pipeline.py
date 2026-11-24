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

    # 3. Antes da revisão humana global, seção 43.3 permanece bloqueada sem expor texto
    before_review = get_sec_uc.execute(document_id="anac_rbac_43", section_number="43.3")
    assert before_review.found is False
    assert before_review.section is None
    assert before_review.blocked_by_document_approval is True
    assert before_review.pending_pdf_pages == (1, 2, 3, 4)

    # 4. Aprova somente página 2 -> continua bloqueado globalmente
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
    assert partial_review.found is False
    assert partial_review.section is None
    assert partial_review.blocked_by_document_approval is True
    assert partial_review.pending_pdf_pages == (1, 3, 4)

    # 5. Aprovar apenas a seção 43.3 (p. 2 e 3) AINDA mantém a versão bloqueada (p. 1 e p. 4 OCR)!
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

    still_blocked = get_sec_uc.execute(document_id="anac_rbac_43", section_number="43.3")
    assert still_blocked.found is False
    assert still_blocked.section is None
    assert still_blocked.blocked_by_document_approval is True
    assert still_blocked.pending_pdf_pages == (1, 4)

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
    from rbac_assistant.domain.entities.regulatory_page import (
        RegulatoryPage,
        compute_page_content_hash,
    )

    new_text = altered_p2.original_text + "\n(Alteração simulada no texto da página 2)"
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
        content_hash=compute_page_content_hash(new_text),
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
def test_reprocessing_table_only_change_invalidates_human_approval_and_unchanged_preserves_it(
    tmp_path: Path,
) -> None:
    """Alterar apenas uma célula de tabela (mantendo original_text idêntico) invalida aprovação."""
    from dataclasses import replace

    from rbac_assistant.domain.ports.pdf_extractor_port import (
        ExtractedPageData,
        ExtractedTableData,
    )

    base_dir = tmp_path / "documents_table_inv"
    repo = SqliteDocumentRepository(base_dir / "catalog.sqlite3")
    repo.initialize()
    store = LocalFileDocumentStore(documents_dir=base_dir)
    store.ensure_directories()
    catalog = CatalogService(repo)
    validation = ValidationService(policy=ValidationPolicy.REVIEW_ON_DEMAND)

    pdf_path = create_synthetic_rbac_pdf(tmp_path / "rbac43_table_inv.pdf")
    base_extracted = PdfPlumberExtractorAdapter().extract_pages(pdf_path, min_native_chars=40)
    assert len(base_extracted) == 3
    assert len(base_extracted[2].tables) == 1

    class _ControllablePdfExtractor:
        def __init__(self, pages: list[ExtractedPageData]) -> None:
            self.pages = pages

        def extract_pages(
            self,
            pdf_path: Path,
            *,
            min_native_chars: int = 50,
            max_pages: int = 1000,
        ) -> list[ExtractedPageData]:
            del pdf_path, min_native_chars, max_pages
            return list(self.pages)

    extractor = _ControllablePdfExtractor(base_extracted)
    ingestion_svc = IngestionService(
        repository=repo,
        file_store=store,
        pdf_extractor=extractor,
        quality_checker=TextQualityChecker(),
        validation_service=validation,
        structure_strategies=(AnacRbacStructureStrategy(), GenericStructureStrategy()),
        local_source=LocalFileSourceAdapter(),
    )
    validate_uc = ValidatePageUseCase(
        repository=repo,
        catalog_service=catalog,
        validation_service=validation,
        file_store=store,
    )
    get_sec_uc = GetNormativeSectionUseCase(
        repository=repo,
        catalog_service=catalog,
        validation_service=validation,
    )
    doc_dto = RegisterDocumentDTO(
        document_id="anac_rbac_43_tbl",
        authority="ANAC",
        document_type=DocumentType.RBAC,
        regulation_number="43",
        document_title="RBAC 43 Sintético (Teste de Hash de Tabela)",
        amendment_number="05",
        version_status=VersionStatus.CURRENT,
    )

    # 1. Ingestão inicial e aprovação global de todas as 3 páginas
    first = ingestion_svc.ingest(IngestionRequestDTO(document=doc_dto, source_uri=str(pdf_path)))
    validate_uc.execute(
        PageReviewRequestDTO(
            document_id="anac_rbac_43_tbl",
            version_id=first.version.version_id,
            reviewer="revisor_tabelas",
            decision=ReviewDecision.APPROVE,
            confirmed=True,
            page_numbers=(1, 2, 3),
            comments="Todas as páginas e tabelas conferidas.",
        )
    )
    ok_before = get_sec_uc.execute(document_id="anac_rbac_43_tbl", section_number="43.9")
    assert ok_before.found is True and ok_before.section is not None
    assert ok_before.section.literal_transcription_allowed is True

    # 2. Reprocessamento sem alteração de texto nem tabela -> preserva 100% das aprovações!
    unchanged = ingestion_svc.ingest(
        IngestionRequestDTO(
            document=doc_dto,
            source_uri=str(pdf_path),
            force_reprocess=True,
        )
    )
    assert unchanged.invalidated_reviews == ()
    assert all(
        p.validation_status == PageValidationStatus.APPROVED_FOR_LITERAL_TRANSCRIPTION
        for p in repo.get_pages(first.version.version_id)
    )

    # 3. Altera APENAS 1 célula da tabela da página 3, mantendo original_text 100% idêntico!
    p3_orig = base_extracted[2]
    orig_rows = p3_orig.tables[0].rows
    edited_rows = (
        orig_rows[0],
        ("01", "Inspeção extraordinária (editada)", orig_rows[1][2]),
        *orig_rows[2:],
    )
    altered_p3 = replace(
        p3_orig,
        tables=(ExtractedTableData(pdf_page_number=3, rows=edited_rows),),
    )
    extractor.pages = [base_extracted[0], base_extracted[1], altered_p3]

    reprocessed = ingestion_svc.ingest(
        IngestionRequestDTO(
            document=doc_dto,
            source_uri=str(pdf_path),
            force_reprocess=True,
        )
    )
    assert len(reprocessed.invalidated_reviews) == 1
    assert reprocessed.invalidated_reviews[0].decision == ReviewDecision.INVALIDATED_BY_REPROCESSING
    assert reprocessed.invalidated_reviews[0].page_id.endswith("::p0003")

    pages_after = {p.pdf_page_number: p for p in repo.get_pages(first.version.version_id)}
    assert pages_after[1].validation_status == (
        PageValidationStatus.APPROVED_FOR_LITERAL_TRANSCRIPTION
    )
    assert pages_after[2].validation_status == (
        PageValidationStatus.APPROVED_FOR_LITERAL_TRANSCRIPTION
    )
    assert pages_after[3].validation_status != (
        PageValidationStatus.APPROVED_FOR_LITERAL_TRANSCRIPTION
    )

    # Como a p. 3 perdeu aprovação, TODA a versão (inclusive 43.1 na p. 2) volta ao bloqueio!
    blocked_after_table_edit = get_sec_uc.execute(
        document_id="anac_rbac_43_tbl",
        section_number="43.1",
    )
    assert blocked_after_table_edit.found is False
    assert blocked_after_table_edit.section is None
    assert blocked_after_table_edit.blocked_by_document_approval is True
    assert blocked_after_table_edit.pending_pdf_pages == (3,)


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

    # Aprova TODAS as páginas do documento fictício 999 e da Resolução 700 — não afeta RBAC 43!
    validate_uc.execute(
        PageReviewRequestDTO(
            document_id="test_fictional_rbac_999",
            reviewer="Engenheiro QA",
            decision=ReviewDecision.APPROVE,
            confirmed=True,
            page_numbers=(1, 2, 3),
        )
    )
    validate_uc.execute(
        PageReviewRequestDTO(
            document_id="test_fictional_res_700",
            reviewer="Engenheiro QA",
            decision=ReviewDecision.APPROVE,
            confirmed=True,
            page_numbers=(1,),
        )
    )
    # Aprova integralmente apenas a Emenda 04 do RBAC 43 (Emenda 05 permanece pendente!)
    validate_uc.execute(
        PageReviewRequestDTO(
            document_id="anac_rbac_43",
            version_id=repo.list_versions("anac_rbac_43")[0].version_id
            if repo.list_versions("anac_rbac_43")[0].amendment_number == "04"
            else repo.list_versions("anac_rbac_43")[1].version_id,
            reviewer="Engenheiro QA",
            decision=ReviewDecision.APPROVE,
            confirmed=True,
            page_numbers=(1, 2, 3),
        )
    )

    sec_999 = get_sec_uc.execute(document_id="test_fictional_rbac_999", section_number="999.3")
    sec_43_current = get_sec_uc.execute(document_id="anac_rbac_43", section_number="43.3")
    sec_43_hist = get_sec_uc.execute(
        document_id="anac_rbac_43", section_number="43.3", amendment_number="04"
    )
    sec_res = get_sec_uc.execute(document_id="test_fictional_res_700", section_number="Art. 1º")

    assert sec_999.found is True
    assert sec_999.section is not None and sec_999.section.literal_transcription_allowed is True
    assert sec_43_current.found is False
    assert sec_43_current.section is None
    assert sec_43_current.blocked_by_document_approval is True
    assert sec_43_hist.found is True and sec_43_hist.section is not None
    assert sec_43_hist.section.amendment_number == "04"
    assert "Emenda 04" in sec_43_hist.section.original_text
    assert sec_res.found is True and sec_res.section is not None
    assert "Art. 1º" in sec_res.section.original_text


@pytest.mark.integration
def test_reprocessing_with_altered_table_metadata_or_ownership_invalidates_review(
    tmp_path: Path,
) -> None:
    """Blocker 1 (#13): alterar metadado de tabela (sem mudar texto/rows) invalida revisão."""
    from dataclasses import replace

    from rbac_assistant.domain.entities.normative_unit import TableCellGeometry
    from rbac_assistant.domain.ports.pdf_extractor_port import ExtractedPageData

    pdf_path = create_synthetic_rbac_pdf(tmp_path / "rbac43_meta_hash.pdf")
    base_dir = tmp_path / "documents_meta_hash"
    base_extracted = PdfPlumberExtractorAdapter().extract_pages(pdf_path, min_native_chars=40)

    class _MutableMetaExtractor:
        def __init__(self, pages: list[ExtractedPageData]) -> None:
            self.pages = pages

        def extract_pages(
            self,
            pdf_path_arg: Path,
            *,
            min_native_chars: int = 80,
            max_pages: int | None = None,
        ) -> list[ExtractedPageData]:
            del pdf_path_arg, min_native_chars, max_pages
            return list(self.pages)

    extractor = _MutableMetaExtractor(base_extracted)
    repo = SqliteDocumentRepository(base_dir / "catalog.sqlite3")
    repo.initialize()
    store = LocalFileDocumentStore(documents_dir=base_dir)
    store.ensure_directories()
    catalog = CatalogService(repo)
    validation = ValidationService(policy=ValidationPolicy.STRICT_ALL_PAGES)
    ingestion_svc = IngestionService(
        repository=repo,
        file_store=store,
        pdf_extractor=extractor,
        quality_checker=TextQualityChecker(),
        validation_service=validation,
        structure_strategies=(AnacRbacStructureStrategy(), GenericStructureStrategy()),
        local_source=LocalFileSourceAdapter(),
    )
    validate_uc = ValidatePageUseCase(
        repository=repo,
        catalog_service=catalog,
        validation_service=validation,
        file_store=store,
    )
    get_sec_uc = GetNormativeSectionUseCase(
        repository=repo,
        catalog_service=catalog,
        validation_service=validation,
    )
    doc_dto = RegisterDocumentDTO(
        document_id="anac_rbac_43_meta_hash",
        authority="ANAC",
        document_type=DocumentType.RBAC,
        regulation_number="43",
        document_title="Manutenção Aeronáutica",
        amendment_number="05",
        version_status=VersionStatus.CURRENT,
    )

    try:
        first = ingestion_svc.ingest(
            IngestionRequestDTO(document=doc_dto, source_uri=str(pdf_path))
        )
        validate_uc.execute(
            PageReviewRequestDTO(
                document_id=doc_dto.document_id,
                version_id=first.version.version_id,
                reviewer="revisor_meta",
                decision=ReviewDecision.APPROVE,
                confirmed=True,
                page_numbers=(1, 2, 3),
                comments="Aprovação completa inicial.",
            )
        )
        assert get_sec_uc.execute(document_id=doc_dto.document_id, section_number="43.9").found

        # Altera SOMENTE metadados estruturais na p. 3 (caption, header_row_count, col_span),
        # mantendo original_text e rows 100% IDÊNTICOS!
        p3_orig = base_extracted[2]
        t0 = p3_orig.tables[0]
        altered_cells = (
            TableCellGeometry(
                row_index=0, col_index=0, row_span=1, col_span=2, text=t0.rows[0][0]
            ),
            *t0.cells[1:],
        )
        altered_tbl = replace(
            t0,
            caption="TABELA ALTERADA SEM MUDAR ROWS",
            header_row_count=2,
            cells=altered_cells,
            has_merged_cells=True,
        )
        extractor.pages = [
            base_extracted[0],
            base_extracted[1],
            replace(p3_orig, tables=(altered_tbl,)),
        ]

        reproc = ingestion_svc.ingest(
            IngestionRequestDTO(
                document=doc_dto,
                source_uri=str(pdf_path),
                force_reprocess=True,
            )
        )
        assert len(reproc.invalidated_reviews) == 1
        assert reproc.invalidated_reviews[0].decision == ReviewDecision.INVALIDATED_BY_REPROCESSING
        assert reproc.invalidated_reviews[0].page_id.endswith("::p0003")

        # Como p. 3 foi invalidada, o bloqueio global do PR #19 bloqueia TODAS as seções
        blocked = get_sec_uc.execute(document_id=doc_dto.document_id, section_number="43.1")
        assert blocked.found is False
        assert blocked.blocked_by_document_approval is True
        assert blocked.pending_pdf_pages == (3,)
    finally:
        repo.close()
