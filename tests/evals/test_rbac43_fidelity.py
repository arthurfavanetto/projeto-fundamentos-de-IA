"""Testes de fidelidade normativa contra amostras de referência conferidas manualmente."""

from __future__ import annotations

import json
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
from rbac_assistant.domain.entities.normative_unit import NormativeUnitType
from rbac_assistant.domain.entities.regulatory_document import (
    DocumentType,
    ValidationPolicy,
    VersionStatus,
)
from rbac_assistant.domain.entities.regulatory_page import ReviewDecision
from rbac_assistant.infrastructure.pdf.anac_structure_strategy import AnacRbacStructureStrategy
from rbac_assistant.infrastructure.pdf.generic_structure_strategy import GenericStructureStrategy
from rbac_assistant.infrastructure.pdf.pdfplumber_extractor import PdfPlumberExtractorAdapter
from rbac_assistant.infrastructure.pdf.text_quality_checker import TextQualityChecker
from rbac_assistant.infrastructure.sources.local_file_source_adapter import LocalFileSourceAdapter
from rbac_assistant.infrastructure.storage.file_document_store import LocalFileDocumentStore
from rbac_assistant.infrastructure.storage.sqlite_document_repository import (
    SqliteDocumentRepository,
)
from tests.fixtures.synthetic_pdf_factory import create_synthetic_rbac_pdf


def _run_ingestion(base_dir: Path, pdf_path: Path) -> tuple[
    SqliteDocumentRepository,
    GetNormativeSectionUseCase,
    ValidatePageUseCase,
    str,
]:
    repo = SqliteDocumentRepository(base_dir / "catalog.sqlite3")
    repo.initialize()
    store = LocalFileDocumentStore(documents_dir=base_dir)
    store.ensure_directories()
    catalog = CatalogService(repo)
    validation = ValidationService(ValidationPolicy.REVIEW_ON_DEMAND)
    svc = IngestionService(
        repository=repo,
        file_store=store,
        pdf_extractor=PdfPlumberExtractorAdapter(),
        quality_checker=TextQualityChecker(),
        validation_service=validation,
        structure_strategies=(AnacRbacStructureStrategy(), GenericStructureStrategy()),
        local_source=LocalFileSourceAdapter(),
    )
    res = IngestDocumentUseCase(svc).execute(
        IngestionRequestDTO(
            document=RegisterDocumentDTO(
                document_id="anac_rbac_43",
                authority="ANAC",
                document_type=DocumentType.RBAC,
                regulation_number="43",
                document_title="Manutenção, Manutenção Preventiva, Reconstrução e Alteração",
                amendment_number="05",
                version_status=VersionStatus.CURRENT,
            ),
            source_uri=str(pdf_path),
        )
    )
    lookup = GetNormativeSectionUseCase(
        repository=repo,
        catalog_service=catalog,
        validation_service=validation,
    )
    validate_uc = ValidatePageUseCase(
        repository=repo,
        catalog_service=catalog,
        validation_service=validation,
        file_store=store,
    )
    return repo, lookup, validate_uc, res.version.source_hash


def test_synthetic_verbatim_fidelity_preserves_punctuation_accents_and_hierarchy(
    tmp_path: Path,
) -> None:
    """Valida bloqueio pré-aprovação e preservação literal integral após aprovação global em CI."""
    pdf_path = create_synthetic_rbac_pdf(tmp_path / "rbac43_fidelity.pdf")
    repo, lookup, validate_uc, _hash = _run_ingestion(tmp_path / "docs", pdf_path)

    # Antes da aprovação global de 100% das páginas, a consulta de seção deve ser bloqueada
    blocked = lookup.execute(document_id="anac_rbac_43", section_number="43.1")
    assert blocked.found is False
    assert blocked.section is None
    assert blocked.blocked_by_document_approval is True
    assert blocked.pending_pdf_pages == (1, 2, 3)

    # Aprova todas as 3 páginas do PDF sintético em tmp_path
    ver = repo.get_current_version("anac_rbac_43")
    assert ver is not None
    validate_uc.execute(
        PageReviewRequestDTO(
            document_id="anac_rbac_43",
            version_id=ver.version_id,
            reviewer="revisor_teste",
            decision=ReviewDecision.APPROVE,
            confirmed=True,
            page_numbers=tuple(range(1, ver.page_count + 1)),
            comments="Aprovação integral do PDF sintético em tmp_path.",
        )
    )

    sec_1 = lookup.execute(document_id="anac_rbac_43", section_number="43.1")
    assert sec_1.found and sec_1.section is not None
    assert sec_1.section.literal_transcription_allowed is True
    expected_lines = [
        "43.1 Aplicabilidade",
        "(a) Este regulamento estabelece regras de manutenção aeronáutica e inspeção.",
        "(1) aeronave com certificado de aeronavegabilidade brasileiro;",
        "(i) célula, motor, hélice, acessório e componentes.",
    ]
    for line in expected_lines:
        assert line in sec_1.section.original_text


def test_official_rbac43_emd05_verbatim_fidelity_when_pdf_available(
    repo_root: Path,
    tmp_path: Path,
) -> None:
    """Valida fidelidade literal da extração do RBAC 43 Emenda 05 sem aprovar páginas reais."""
    official_pdf = repo_root / "data" / "raw_pdfs" / "RBAC43EMD05.pdf"
    if not official_pdf.is_file():
        pytest.skip(
            "PDF oficial RBAC43EMD05.pdf não está em data/raw_pdfs/ neste ambiente (gitignored)."
        )

    fixture_path = repo_root / "tests" / "fixtures" / "rbac43_reference_samples.json"
    reference = json.loads(fixture_path.read_text(encoding="utf-8"))

    repo, lookup, _validate_uc, file_hash = _run_ingestion(tmp_path / "official_eval", official_pdf)
    assert file_hash == reference["expected_sha256"]

    ver = repo.get_current_version("anac_rbac_43")
    assert ver is not None
    assert ver.page_count == reference["expected_total_pages"]

    roots = [u for u in repo.list_units(ver.version_id) if u.parent_unit_id is None]
    assert len(roots) == reference["expected_root_sections_count"]

    pages_by_num = {p.pdf_page_number: p for p in repo.get_pages(ver.version_id)}

    for sample in reference["samples"]:
        # 1. Sem aprovação humana das 32 páginas, GetNormativeSectionUseCase permanece bloqueado
        blocked_lookup = lookup.execute(
            document_id="anac_rbac_43",
            section_number=sample["section_number"],
        )
        assert blocked_lookup.found is False
        assert blocked_lookup.section is None
        assert blocked_lookup.blocked_by_document_approval is True
        assert blocked_lookup.pending_pdf_pages == tuple(range(1, ver.page_count + 1))

        # 2. Confere fidelidade literal integral diretamente nas unidades extraídas em tmp_path
        units = repo.get_units_by_section(
            version_id=ver.version_id,
            section_number=sample["section_number"],
        )
        assert units
        root_units = [
            u
            for u in units
            if u.parent_unit_id is None
            and u.unit_type in (NormativeUnitType.SECTION, NormativeUnitType.APPENDIX)
        ]
        primary = root_units[0] if root_units else units[0]
        pdf_pages = sorted({p for u in units for p in u.page_numbers})
        printed_pages = [
            pages_by_num[p].printed_page_number or f"PDF {p}"
            for p in pdf_pages
            if p in pages_by_num
        ]
        assert pdf_pages == sample["expected_pdf_pages"]
        assert printed_pages == sample["expected_printed_pages"]
        for fragment in sample["expected_verbatim_fragments"]:
            assert fragment in primary.original_text
