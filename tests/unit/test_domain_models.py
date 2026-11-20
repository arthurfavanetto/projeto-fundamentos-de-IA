"""Testes unitários das entidades de domínio, política OCR e fronteira hexagonal."""

from __future__ import annotations

import ast
import sys
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from rbac_assistant.domain.entities.citation import Citation, PageReference
from rbac_assistant.domain.entities.normative_unit import (
    ExtractionMethod,
    NormativeUnit,
    NormativeUnitType,
    PageExtractionMetadata,
    ValidationStatus,
)
from rbac_assistant.domain.entities.query_result import NormativeResponse
from rbac_assistant.domain.entities.regulation import (
    Amendment,
    DocumentMetadata,
    Regulation,
    VigenciaStatus,
)
from rbac_assistant.domain.exceptions import UnvalidatedPageBlockedError


@pytest.mark.unit
def test_domain_layer_has_zero_external_or_outer_layer_imports(repo_root: Path) -> None:
    """Garante a regra da Arquitetura Hexagonal: `domain/` usa apenas stdlib + `domain/`."""
    domain_dir = repo_root / "src" / "rbac_assistant" / "domain"
    stdlib_modules = sys.stdlib_module_names

    violations: list[str] = []
    for py_file in sorted(domain_dir.rglob("*.py")):
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    if top not in stdlib_modules and not alias.name.startswith(
                        "rbac_assistant.domain"
                    ):
                        violations.append(f"{py_file.name}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom) and node.module:
                top = node.module.split(".")[0]
                if top not in stdlib_modules and not node.module.startswith(
                    "rbac_assistant.domain"
                ):
                    violations.append(f"{py_file.name}: from {node.module} import ...")

    assert violations == [], f"Imports proibidos detectados em domain/: {violations}"


@pytest.mark.unit
def test_ocr_page_is_blocked_by_default_and_cannot_bypass_manual_review() -> None:
    ocr_page = PageExtractionMetadata.create_ocr_page(
        pdf_page_number=7,
        printed_page_number="Página 7 de 28",
        integrity_checks_passed=True,
    )
    assert ocr_page.extraction_method == ExtractionMethod.OCR_FALLBACK
    assert ocr_page.validation_status == ValidationStatus.BLOCKED_UNREVIEWED
    assert ocr_page.is_literal_transcription_allowed is False

    with pytest.raises(ValueError, match="block_until_reviewed"):
        PageExtractionMetadata(
            pdf_page_number=7,
            printed_page_number="Página 7 de 28",
            extraction_method=ExtractionMethod.OCR_FALLBACK,
            validation_status=ValidationStatus.VALIDATED,
            integrity_checks_passed=True,
        )


@pytest.mark.unit
def test_native_page_requires_integrity_checks_before_validation() -> None:
    with pytest.raises(ValueError, match="integridade textual"):
        PageExtractionMetadata(
            pdf_page_number=2,
            printed_page_number="Página 2 de 28",
            extraction_method=ExtractionMethod.NATIVE_PDF,
            validation_status=ValidationStatus.VALIDATED,
            integrity_checks_passed=False,
        )


@pytest.mark.unit
def test_section_with_all_validated_pages_allows_literal_transcription(
    validated_native_page: PageExtractionMetadata,
    reviewed_ocr_page: PageExtractionMetadata,
) -> None:
    unit = NormativeUnit(
        document_id="rbac_43_test",
        regulation_number="RBAC 43",
        amendment_number="00",
        unit_type=NormativeUnitType.SECTION,
        section_number="43.3",
        title="Pessoas autorizadas",
        exact_text="43.3 Texto normativo de teste.",
        pages=(validated_native_page, reviewed_ocr_page),
    )

    assert unit.can_guarantee_literal_transcription is True
    assert unit.unvalidated_pages == ()
    assert unit.literal_transcription_warning is None
    assert unit.require_literal_transcription() == "43.3 Texto normativo de teste."


@pytest.mark.unit
def test_section_spanning_blocked_ocr_page_warns_and_blocks_literal_guarantee(
    validated_native_page: PageExtractionMetadata,
    blocked_ocr_page: PageExtractionMetadata,
) -> None:
    unit = NormativeUnit(
        document_id="rbac_43_test",
        regulation_number="RBAC 43",
        amendment_number="00",
        unit_type=NormativeUnitType.SECTION,
        section_number="43.9",
        title="Conteúdo, forma e disposição de registros",
        exact_text="43.9 Texto parcial extraído.",
        pages=(validated_native_page, blocked_ocr_page),
    )

    assert unit.can_guarantee_literal_transcription is False
    assert unit.unvalidated_pages == (blocked_ocr_page,)
    assert unit.literal_transcription_warning is not None
    assert "não pode garantir a transcrição literal completa" in unit.literal_transcription_warning
    assert "PDF p.5" in unit.literal_transcription_warning
    assert "Página 5 de 28" in unit.literal_transcription_warning

    with pytest.raises(UnvalidatedPageBlockedError) as exc_info:
        unit.require_literal_transcription()
    assert exc_info.value.blocked_pdf_pages == (5,)


@pytest.mark.unit
def test_normative_response_requires_warning_when_literal_not_guaranteed() -> None:
    citation = Citation(
        regulation_number="RBAC 43",
        regulation_name="Regulamento Brasileiro da Aviação Civil nº 43",
        amendment_number="00",
        vigencia_status=VigenciaStatus.VIGENTE,
        section_number="43.9",
        pages=(
            PageReference(
                pdf_page_number=5,
                printed_page_number="Página 5 de 28",
                extraction_method=ExtractionMethod.OCR_FALLBACK,
                validation_status=ValidationStatus.BLOCKED_UNREVIEWED,
            ),
        ),
        official_url="https://www.anac.gov.br/exemplo",
    )
    assert citation.has_unvalidated_pages is True

    with pytest.raises(ValueError, match="fidelity_warning"):
        NormativeResponse(
            query="43.9",
            found=True,
            citations=(citation,),
            literal_transcription_guaranteed=False,
        )


@pytest.mark.unit
def test_document_metadata_and_regulation_current_amendment() -> None:
    meta = DocumentMetadata(
        document_id="rbac_43_emd_test",
        regulation_number="RBAC 43",
        regulation_name="Regulamento Brasileiro da Aviação Civil nº 43",
        document_title="Manutenção, Manutenção Preventiva, Reconstrução e Alteração",
        amendment_number="TEST",
        resolution_number="Resolução TEST",
        publication_date=date(2026, 1, 1),
        effective_date=date(2026, 2, 1),
        vigencia_status=VigenciaStatus.VIGENTE,
        is_current_amendment=True,
        official_url="https://www.anac.gov.br/exemplo.pdf",
        consulted_at=datetime(2026, 11, 20, tzinfo=UTC),
        pdf_filename="rbac43.pdf",
        sha256="f" * 64,
    )
    assert meta.regulation_number == "RBAC 43"

    with pytest.raises(ValueError, match="64 caracteres"):
        DocumentMetadata(
            document_id="rbac_43_bad",
            regulation_number="RBAC 43",
            regulation_name="RBAC 43",
            document_title="Título",
            amendment_number="01",
            resolution_number="Res 1",
            publication_date=date(2026, 1, 1),
            effective_date=date(2026, 1, 2),
            vigencia_status=VigenciaStatus.VIGENTE,
            is_current_amendment=True,
            official_url="https://www.anac.gov.br/exemplo.pdf",
            consulted_at=datetime(2026, 11, 20, tzinfo=UTC),
            pdf_filename="rbac43.pdf",
            sha256="short_hash",
        )

    regulation = Regulation(
        regulation_number="RBAC 43",
        regulation_name="Regulamento Brasileiro da Aviação Civil nº 43",
        document_title="Manutenção",
        amendments=(
            Amendment(
                amendment_number="01",
                resolution_number="Res 1",
                publication_date=date(2020, 1, 1),
                effective_date=date(2020, 2, 1),
                vigencia_status=VigenciaStatus.REVOGADO,
                is_current=False,
            ),
            Amendment(
                amendment_number="02",
                resolution_number="Res 2",
                publication_date=date(2024, 1, 1),
                effective_date=date(2024, 2, 1),
                vigencia_status=VigenciaStatus.VIGENTE,
                is_current=True,
            ),
        ),
    )
    assert regulation.current_amendment is not None
    assert regulation.current_amendment.amendment_number == "02"
