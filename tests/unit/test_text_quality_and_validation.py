"""Testes unitários de análise de qualidade textual e da arquitetura de validação híbrida."""

from __future__ import annotations

import hashlib

import pytest

from rbac_assistant.application.services.validation_service import ValidationService
from rbac_assistant.domain.entities.normative_unit import ExtractionMethod, StructureStatus
from rbac_assistant.domain.entities.regulatory_document import ValidationPolicy
from rbac_assistant.domain.entities.regulatory_page import (
    PageValidationStatus,
    RegulatoryPage,
    ReviewDecision,
)
from rbac_assistant.domain.exceptions import ValidationPolicyError
from rbac_assistant.infrastructure.pdf.text_quality_checker import TextQualityChecker


def _make_page(
    *,
    num: int,
    method: ExtractionMethod = ExtractionMethod.NATIVE_PDF,
    status: PageValidationStatus = PageValidationStatus.NATIVE_QUALITY_CHECKED,
    text: str = "43.3 Pessoas autorizadas a executar manutenção preventiva e alteração.",
) -> RegulatoryPage:
    return RegulatoryPage(
        page_id=f"ver::p{num:04d}",
        version_id="ver",
        pdf_page_number=num,
        printed_page_number=f"{num}/32",
        original_text=text,
        normalized_text=text,
        extraction_method=method,
        validation_status=status,
        content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


@pytest.mark.unit
def test_text_quality_checker_detects_clean_and_corrupted_text() -> None:
    checker = TextQualityChecker()

    clean = (
        "43.16 Limitações de aeronavegabilidade. Cada pessoa que estiver executando "
        "uma inspeção ou manutenção deve seguir o manual do fabricante."
    )
    ok = checker.evaluate(original_text=clean, normalized_text=clean)
    assert ok.passed_automated_checks is True
    assert ok.has_portuguese_accents is True
    assert ok.issues == ()

    broken = "Texto corrompido com \ufffd e mojibake manutenÃ§Ã£o (((("
    bad = checker.evaluate(original_text=broken, normalized_text=broken, min_chars=10)
    assert bad.passed_automated_checks is False
    assert bad.has_replacement_chars is True
    assert bad.has_mojibake_signals is True
    assert bad.unbalanced_parentheses is True
    assert len(bad.issues) >= 3


@pytest.mark.unit
def test_validation_service_distinguishes_native_quality_checked_from_human_approval() -> None:
    svc = ValidationService(policy=ValidationPolicy.REVIEW_ON_DEMAND)
    p1 = _make_page(num=2, status=PageValidationStatus.NATIVE_QUALITY_CHECKED)
    p2 = _make_page(num=3, status=PageValidationStatus.APPROVED_FOR_LITERAL_TRANSCRIPTION)

    # Sob review_on_demand, 1 página aprovada + 1 native_quality_checked não libera literal
    assessment = svc.evaluate_section(
        section_number="43.3",
        pages=[p1, p2],
        structure_status=StructureStatus.PARSED,
    )
    assert assessment.literal_transcription_allowed is False
    assert assessment.pending_pdf_pages == (2,)
    assert assessment.warning_message is not None
    assert "pendente de validação humana" in assessment.warning_message


@pytest.mark.unit
def test_validation_service_blocks_ocr_under_all_policies() -> None:
    p_native = _make_page(num=2, status=PageValidationStatus.APPROVED_FOR_LITERAL_TRANSCRIPTION)
    p_ocr = _make_page(
        num=3,
        method=ExtractionMethod.OCR_FALLBACK,
        status=PageValidationStatus.PENDING_MANUAL_REVIEW,
    )

    for policy in ValidationPolicy:
        svc = ValidationService(policy=policy)
        assessment = svc.evaluate_section(
            section_number="43.3",
            pages=[p_native, p_ocr],
            structure_status=StructureStatus.PARSED,
        )
        assert assessment.literal_transcription_allowed is False
        assert assessment.auxiliary_unreviewed_allowed is False
        assert assessment.ocr_blocked_pdf_pages == (3,)
        assert assessment.warning_message is not None
        assert "OCR" in assessment.warning_message


@pytest.mark.unit
def test_native_auto_ocr_manual_allows_auxiliary_unreviewed_without_claiming_human_review() -> None:
    svc = ValidationService(policy=ValidationPolicy.NATIVE_AUTO_OCR_MANUAL)
    p1 = _make_page(num=2, status=PageValidationStatus.NATIVE_QUALITY_CHECKED)

    assessment = svc.evaluate_section(
        section_number="43.1",
        pages=[p1],
        structure_status=StructureStatus.PARSED,
    )
    assert assessment.literal_transcription_allowed is False
    assert assessment.auxiliary_unreviewed_allowed is True
    assert assessment.warning_message is not None
    assert "extração não revisada" in assessment.warning_message


@pytest.mark.unit
def test_manual_review_requires_explicit_confirmation_and_reviewer() -> None:
    svc = ValidationService(policy=ValidationPolicy.REVIEW_ON_DEMAND)
    page = _make_page(num=3, status=PageValidationStatus.NATIVE_QUALITY_CHECKED)

    with pytest.raises(ValidationPolicyError, match="confirmação explícita"):
        svc.build_review_transition(
            page=page,
            decision=ReviewDecision.APPROVE,
            reviewer="Arthur",
            confirmed=False,
        )

    with pytest.raises(ValidationPolicyError, match="responsável"):
        svc.build_review_transition(
            page=page,
            decision=ReviewDecision.APPROVE,
            reviewer="   ",
            confirmed=True,
        )

    status_ok, rec_ok = svc.build_review_transition(
        page=page,
        decision=ReviewDecision.APPROVE,
        reviewer="Arthur Fava",
        confirmed=True,
        comments="Conferido com PDF oficial ANAC.",
    )
    assert status_ok == PageValidationStatus.APPROVED_FOR_LITERAL_TRANSCRIPTION
    assert rec_ok.reviewer == "Arthur Fava"
    assert rec_ok.source_hash == page.content_hash

    status_rej, rec_rej = svc.build_review_transition(
        page=page,
        decision=ReviewDecision.REJECT,
        reviewer="Arthur Fava",
        confirmed=True,
        comments="Linha quebrada incorretamente.",
    )
    assert status_rej == PageValidationStatus.REJECTED
    assert rec_rej.decision == ReviewDecision.REJECT
