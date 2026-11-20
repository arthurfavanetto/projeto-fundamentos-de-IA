"""Testes de fidelidade normativa contra amostras de referência conferidas manualmente."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rbac_assistant.application.dto.ingestion_dto import (
    IngestionRequestDTO,
    RegisterDocumentDTO,
)
from rbac_assistant.application.services.catalog_service import CatalogService
from rbac_assistant.application.services.ingestion_service import IngestionService
from rbac_assistant.application.services.validation_service import ValidationService
from rbac_assistant.application.use_cases.get_normative_section import GetNormativeSectionUseCase
from rbac_assistant.application.use_cases.ingest_document import IngestDocumentUseCase
from rbac_assistant.domain.entities.regulatory_document import (
    DocumentType,
    ValidationPolicy,
    VersionStatus,
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
from tests.fixtures.synthetic_pdf_factory import create_synthetic_rbac_pdf


def _run_ingestion(base_dir: Path, pdf_path: Path) -> tuple[
    SqliteDocumentRepository,
    GetNormativeSectionUseCase,
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
    return repo, lookup, res.version.source_hash


def test_synthetic_verbatim_fidelity_preserves_punctuation_accents_and_hierarchy(
    tmp_path: Path,
) -> None:
    """Valida preservação literal de palavras, pontuação, acentos e incisos em CI."""
    pdf_path = create_synthetic_rbac_pdf(tmp_path / "rbac43_fidelity.pdf")
    _repo, lookup, _hash = _run_ingestion(tmp_path / "docs", pdf_path)

    sec_1 = lookup.execute(document_id="anac_rbac_43", section_number="43.1")
    assert sec_1.found and sec_1.section is not None
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
    """Valida fidelidade literal contra o PDF oficial do RBAC 43 Emenda 05 quando presente."""
    official_pdf = repo_root / "data" / "raw_pdfs" / "RBAC43EMD05.pdf"
    if not official_pdf.is_file():
        pytest.skip(
            "PDF oficial RBAC43EMD05.pdf não está em data/raw_pdfs/ neste ambiente (gitignored)."
        )

    fixture_path = repo_root / "tests" / "fixtures" / "rbac43_reference_samples.json"
    reference = json.loads(fixture_path.read_text(encoding="utf-8"))

    repo, lookup, file_hash = _run_ingestion(tmp_path / "official_eval", official_pdf)
    assert file_hash == reference["expected_sha256"]

    ver = repo.get_current_version("anac_rbac_43")
    assert ver is not None
    assert ver.page_count == reference["expected_total_pages"]

    roots = [u for u in repo.list_units(ver.version_id) if u.parent_unit_id is None]
    assert len(roots) == reference["expected_root_sections_count"]

    for sample in reference["samples"]:
        result = lookup.execute(
            document_id="anac_rbac_43",
            section_number=sample["section_number"],
        )
        assert result.found is True and result.section is not None
        assert list(result.section.pdf_pages) == sample["expected_pdf_pages"]
        assert list(result.section.printed_pages) == sample["expected_printed_pages"]
        for fragment in sample["expected_verbatim_fragments"]:
            assert fragment in result.section.original_text
