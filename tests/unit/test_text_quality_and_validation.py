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

    # Sob bloqueio global, 1 página aprovada + 1 native_quality_checked não libera literal nem aux
    assessment = svc.evaluate_section(
        section_number="43.3",
        pages=[p1, p2],
        structure_status=StructureStatus.PARSED,
    )
    assert assessment.literal_transcription_allowed is False
    assert assessment.auxiliary_unreviewed_allowed is False
    assert assessment.pending_pdf_pages == (2,)
    assert assessment.warning_message is not None
    assert "pendentes de validação humana" in assessment.warning_message


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
def test_global_approval_policy_never_allows_unreviewed_auxiliary_under_any_policy() -> None:
    p1 = _make_page(num=2, status=PageValidationStatus.NATIVE_QUALITY_CHECKED)
    for policy in ValidationPolicy:
        svc = ValidationService(policy=policy)
        assessment = svc.evaluate_section(
            section_number="43.1",
            pages=[p1],
            structure_status=StructureStatus.PARSED,
        )
        assert assessment.literal_transcription_allowed is False
        assert assessment.auxiliary_unreviewed_allowed is False
        assert assessment.pending_pdf_pages == (2,)
        assert assessment.warning_message is not None
        assert "política global de aprovação documental" in assessment.warning_message


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


@pytest.mark.unit
def test_evaluate_document_version_enforces_all_global_approval_invariants() -> None:
    from rbac_assistant.domain.entities.document_version import DocumentVersion

    svc = ValidationService(policy=ValidationPolicy.REVIEW_ON_DEMAND)

    def _ver(*, page_count: int, version_id: str = "ver") -> DocumentVersion:
        return DocumentVersion(
            version_id=version_id,
            document_id="doc_43",
            amendment_number="05",
            source_hash="a" * 64,
            page_count=page_count,
        )

    def _page(
        num: int,
        *,
        status: PageValidationStatus = PageValidationStatus.APPROVED_FOR_LITERAL_TRANSCRIPTION,
        version_id: str = "ver",
        method: ExtractionMethod = ExtractionMethod.NATIVE_PDF,
        page_id_suffix: str = "",
    ) -> RegulatoryPage:
        txt = f"Texto da página {num}."
        return RegulatoryPage(
            page_id=f"{version_id}::p{num:04d}{page_id_suffix}",
            version_id=version_id,
            pdf_page_number=num,
            printed_page_number=f"{num}/3",
            original_text=txt,
            normalized_text=txt,
            extraction_method=method,
            validation_status=status,
            content_hash=hashlib.sha256(txt.encode("utf-8")).hexdigest(),
        )

    # 1. 100% das páginas 1..3 aprovadas para transcrição literal -> permitido
    ok = svc.evaluate_document_version(
        version=_ver(page_count=3),
        pages=[_page(1), _page(2), _page(3)],
    )
    assert ok.literal_transcription_allowed is True
    assert ok.block_reason is None
    assert ok.approved_pdf_pages == (1, 2, 3)
    assert ok.pending_pdf_pages == ()
    assert ok.rejected_pdf_pages == ()
    assert ok.missing_pdf_pages == ()
    assert ok.duplicate_pdf_pages == ()

    # 2. page_count == 0 -> bloqueado
    zero = svc.evaluate_document_version(version=_ver(page_count=0), pages=[])
    assert zero.literal_transcription_allowed is False
    assert zero.block_reason is not None
    assert "page_count <= 0" in zero.block_reason

    # 3. Página ausente (falta p. 2 em 1..3) -> bloqueado
    missing = svc.evaluate_document_version(
        version=_ver(page_count=3),
        pages=[_page(1), _page(3)],
    )
    assert missing.literal_transcription_allowed is False
    assert missing.missing_pdf_pages == (2,)
    assert missing.actual_page_count == 2
    assert missing.expected_page_count == 3

    # 4. Página física duplicada (duas p. 2) -> bloqueado mesmo com contagem igual
    dup = svc.evaluate_document_version(
        version=_ver(page_count=3),
        pages=[_page(1), _page(2), _page(2, page_id_suffix="_dup")],
    )
    assert dup.literal_transcription_allowed is False
    assert dup.duplicate_pdf_pages == (2,)
    assert dup.missing_pdf_pages == (3,)

    # 5. Página fora da faixa 1..page_count (p. 4 em page_count=3) -> bloqueado
    out_of_range = svc.evaluate_document_version(
        version=_ver(page_count=3),
        pages=[_page(1), _page(2), _page(4)],
    )
    assert out_of_range.literal_transcription_allowed is False
    assert out_of_range.unexpected_pdf_pages == (4,)
    assert out_of_range.missing_pdf_pages == (3,)

    # 6. Página pertencente a outra version_id -> bloqueado
    wrong_ver = svc.evaluate_document_version(
        version=_ver(page_count=2, version_id="ver_a"),
        pages=[
            _page(1, version_id="ver_a"),
            _page(2, version_id="ver_b"),
        ],
    )
    assert wrong_ver.literal_transcription_allowed is False
    assert wrong_ver.mismatched_version_page_ids == ("ver_b::p0002",)
    assert wrong_ver.missing_pdf_pages == (2,)

    # 7. Qualquer status diferente de APPROVED_FOR_LITERAL_TRANSCRIPTION bloqueia globalmente
    for non_literal_status in (
        PageValidationStatus.PENDING_EXTRACTION,
        PageValidationStatus.NATIVE_EXTRACTED,
        PageValidationStatus.NATIVE_QUALITY_CHECKED,
        PageValidationStatus.PENDING_REVIEW,
        PageValidationStatus.MANUALLY_APPROVED,
        PageValidationStatus.OCR_EXTRACTED,
        PageValidationStatus.PENDING_MANUAL_REVIEW,
        PageValidationStatus.REJECTED,
    ):
        partial = svc.evaluate_document_version(
            version=_ver(page_count=3),
            pages=[
                _page(1),
                _page(2),
                _page(
                    3,
                    status=non_literal_status,
                    method=(
                        ExtractionMethod.OCR_FALLBACK
                        if non_literal_status == PageValidationStatus.PENDING_MANUAL_REVIEW
                        else ExtractionMethod.NATIVE_PDF
                    ),
                ),
            ],
        )
        assert partial.literal_transcription_allowed is False
        assert partial.pending_pdf_pages == (3,)
        assert partial.approved_pdf_pages == (1, 2)
        if non_literal_status == PageValidationStatus.REJECTED:
            assert partial.rejected_pdf_pages == (3,)
        if non_literal_status == PageValidationStatus.PENDING_MANUAL_REVIEW:
            assert partial.ocr_blocked_pdf_pages == (3,)

    # 8. Seção cujas próprias páginas (p. 2) estão aprovadas segue bloqueada se p. 1 pendente
    all_pages_partial = [
        _page(1, status=PageValidationStatus.NATIVE_QUALITY_CHECKED),
        _page(2, status=PageValidationStatus.APPROVED_FOR_LITERAL_TRANSCRIPTION),
        _page(3, status=PageValidationStatus.APPROVED_FOR_LITERAL_TRANSCRIPTION),
    ]
    sec_blocked = svc.evaluate_section(
        section_number="43.1",
        pages=[all_pages_partial[1]],
        structure_status=StructureStatus.PARSED,
        version=_ver(page_count=3),
        all_version_pages=all_pages_partial,
    )
    assert sec_blocked.literal_transcription_allowed is False
    assert sec_blocked.auxiliary_unreviewed_allowed is False
    assert sec_blocked.document_approved is False
    assert sec_blocked.pending_pdf_pages == (1,)
    assert sec_blocked.warning_message is not None
    assert "bloqueada pela política global" in sec_blocked.warning_message
