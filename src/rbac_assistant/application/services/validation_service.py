"""Serviço de validação documental com política obrigatória de aprovação global por versão.

Regra obrigatória de aprovação global:
- Uma versão documental somente pode ser utilizada em recuperação, indexação,
  localização de passagens, prompts para LLM ou respostas ao usuário quando 100%
  de suas páginas físicas (`1..page_count`) existem, pertencem à versão, não possuem
  duplicidades nem lacunas e estão individualmente aprovadas no estado
  `PageValidationStatus.APPROVED_FOR_LITERAL_TRANSCRIPTION`.
- A aprovação isolada de uma página, seção ou conjunto parcial de páginas nunca libera
  o uso da versão documental.
"""

from __future__ import annotations

import uuid
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from rbac_assistant.domain.entities.document_version import DocumentVersion
from rbac_assistant.domain.entities.normative_unit import ExtractionMethod, StructureStatus
from rbac_assistant.domain.entities.regulatory_document import ValidationPolicy
from rbac_assistant.domain.entities.regulatory_page import (
    PageValidationStatus,
    QualityMetrics,
    RegulatoryPage,
    ReviewDecision,
    ValidationRecord,
)
from rbac_assistant.domain.exceptions import ValidationPolicyError


@dataclass(frozen=True, slots=True)
class DocumentVersionValidationAssessment:
    """Resultado determinístico da avaliação global de aprovação de uma versão documental."""

    version_id: str
    literal_transcription_allowed: bool
    expected_page_count: int
    actual_page_count: int
    approved_pdf_pages: tuple[int, ...] = ()
    pending_pdf_pages: tuple[int, ...] = ()
    ocr_blocked_pdf_pages: tuple[int, ...] = ()
    rejected_pdf_pages: tuple[int, ...] = ()
    missing_pdf_pages: tuple[int, ...] = ()
    duplicate_pdf_pages: tuple[int, ...] = ()
    unexpected_pdf_pages: tuple[int, ...] = ()
    mismatched_version_page_ids: tuple[str, ...] = ()
    block_reason: str | None = None


@dataclass(frozen=True, slots=True)
class SectionValidationAssessment:
    """Resultado da avaliação de elegibilidade literal de uma seção multi-página."""

    literal_transcription_allowed: bool
    auxiliary_unreviewed_allowed: bool
    aggregate_status: PageValidationStatus
    pending_pdf_pages: tuple[int, ...]
    ocr_blocked_pdf_pages: tuple[int, ...]
    warning_message: str | None
    document_approved: bool = True


class ValidationService:
    """Aplica as regras de transição de estado e a política global de aprovação documental."""

    def __init__(self, policy: ValidationPolicy = ValidationPolicy.REVIEW_ON_DEMAND) -> None:
        self._policy = policy

    @property
    def policy(self) -> ValidationPolicy:
        return self._policy

    def determine_initial_page_status(
        self,
        *,
        extraction_method: ExtractionMethod,
        quality_metrics: QualityMetrics,
    ) -> PageValidationStatus:
        """Define o estado inicial pós-extração — nunca concede aprovação humana automática."""
        if extraction_method == ExtractionMethod.OCR_FALLBACK:
            return PageValidationStatus.PENDING_MANUAL_REVIEW
        if extraction_method == ExtractionMethod.OCR_REQUIRED_UNAVAILABLE:
            return PageValidationStatus.PENDING_MANUAL_REVIEW

        if quality_metrics.passed_automated_checks:
            return PageValidationStatus.NATIVE_QUALITY_CHECKED
        return PageValidationStatus.PENDING_REVIEW

    def build_review_transition(
        self,
        *,
        page: RegulatoryPage,
        decision: ReviewDecision,
        reviewer: str,
        confirmed: bool,
        comments: str = "",
    ) -> tuple[PageValidationStatus, ValidationRecord]:
        """Valida e constrói a transição explícita de revisão humana para uma página."""
        if not confirmed:
            raise ValidationPolicyError(
                "Aprovação ou reprovação de página exige confirmação explícita (confirmed=True)."
            )
        clean_reviewer = reviewer.strip()
        if not clean_reviewer:
            raise ValidationPolicyError(
                "É obrigatório identificar o responsável autenticado (reviewer)."
            )

        if decision == ReviewDecision.APPROVE:
            if (
                page.extraction_method == ExtractionMethod.OCR_REQUIRED_UNAVAILABLE
                and not page.original_text.strip()
            ):
                raise ValidationPolicyError(
                    f"Página PDF {page.pdf_page_number} não possui texto extraído "
                    "(OCR indisponível) e não pode ser aprovada para transcrição literal."
                )
            target_status = PageValidationStatus.APPROVED_FOR_LITERAL_TRANSCRIPTION
        elif decision == ReviewDecision.REJECT:
            target_status = PageValidationStatus.REJECTED
        else:
            raise ValidationPolicyError(f"Decisão manual inválida: {decision}")

        record = ValidationRecord(
            review_id=f"rev_{uuid.uuid4().hex[:12]}",
            page_id=page.page_id,
            reviewer=clean_reviewer,
            reviewed_at=datetime.now(UTC),
            decision=decision,
            source_hash=page.content_hash,
            comments=comments.strip(),
        )
        return target_status, record

    def evaluate_document_version(
        self,
        *,
        version: DocumentVersion,
        pages: Sequence[RegulatoryPage],
    ) -> DocumentVersionValidationAssessment:
        """Avalia integralmente se 100% das páginas da versão estão aprovadas para uso.

        Critérios cumulativos obrigatórios:
        1. `version.page_count > 0`;
        2. quantidade de páginas recebidas igual a `version.page_count`;
        3. todas as páginas pertencem a `version.version_id`;
        4. cobertura exata e contígua `1..version.page_count`, sem lacunas nem extras;
        5. ausência de páginas físicas duplicadas;
        6. todas as páginas em `PageValidationStatus.APPROVED_FOR_LITERAL_TRANSCRIPTION`.
        """
        expected_count = version.page_count
        actual_count = len(pages)

        mismatched_ids = tuple(
            p.page_id for p in pages if p.version_id != version.version_id
        )
        page_num_counts = Counter(p.pdf_page_number for p in pages)
        duplicate_pages = tuple(
            sorted(num for num, count in page_num_counts.items() if count > 1)
        )

        expected_set = set(range(1, expected_count + 1)) if expected_count > 0 else set()
        valid_version_pages = [p for p in pages if p.version_id == version.version_id]
        present_valid_nums = {p.pdf_page_number for p in valid_version_pages}
        missing_pages = tuple(sorted(expected_set - present_valid_nums))
        unexpected_pages = tuple(
            sorted({p.pdf_page_number for p in pages} - expected_set)
        )

        approved_pages = tuple(
            sorted(
                {
                    p.pdf_page_number
                    for p in valid_version_pages
                    if p.validation_status
                    == PageValidationStatus.APPROVED_FOR_LITERAL_TRANSCRIPTION
                }
            )
        )
        rejected_pages = tuple(
            sorted(
                {
                    p.pdf_page_number
                    for p in valid_version_pages
                    if p.validation_status == PageValidationStatus.REJECTED
                }
            )
        )
        ocr_blocked_pages = tuple(
            sorted(
                {
                    p.pdf_page_number
                    for p in valid_version_pages
                    if p.extraction_method
                    in (ExtractionMethod.OCR_FALLBACK, ExtractionMethod.OCR_REQUIRED_UNAVAILABLE)
                    and p.validation_status
                    != PageValidationStatus.APPROVED_FOR_LITERAL_TRANSCRIPTION
                }
            )
        )
        pending_pages = tuple(
            sorted(
                {
                    p.pdf_page_number
                    for p in valid_version_pages
                    if p.validation_status
                    != PageValidationStatus.APPROVED_FOR_LITERAL_TRANSCRIPTION
                }
            )
        )

        reasons: list[str] = []
        if expected_count <= 0:
            reasons.append("versão sem páginas válidas (page_count <= 0)")
        if mismatched_ids:
            reasons.append(f"páginas de outra versão vinculadas: {list(mismatched_ids)}")
        if duplicate_pages:
            reasons.append(f"páginas físicas duplicadas: {list(duplicate_pages)}")
        if unexpected_pages:
            reasons.append(
                f"páginas físicas fora da faixa 1..{expected_count}: {list(unexpected_pages)}"
            )
        if missing_pages:
            reasons.append(f"páginas físicas ausentes: {list(missing_pages)}")
        if actual_count != expected_count:
            reasons.append(
                f"quantidade de páginas ({actual_count}) diverge de page_count ({expected_count})"
            )
        if rejected_pages:
            reasons.append(f"páginas reprovadas: {list(rejected_pages)}")
        if ocr_blocked_pages:
            reasons.append(f"páginas OCR pendentes de revisão manual: {list(ocr_blocked_pages)}")
        non_rejected_pending = tuple(
            p for p in pending_pages if p not in set(rejected_pages)
        )
        if non_rejected_pending:
            reasons.append(
                f"páginas pendentes de aprovação literal: {list(non_rejected_pending)}"
            )

        allowed = len(reasons) == 0
        block_reason = (
            None
            if allowed
            else (
                f"Versão documental '{version.version_id}' bloqueada pela política global "
                f"de aprovação documental: {'; '.join(reasons)}."
            )
        )

        return DocumentVersionValidationAssessment(
            version_id=version.version_id,
            literal_transcription_allowed=allowed,
            expected_page_count=expected_count,
            actual_page_count=actual_count,
            approved_pdf_pages=approved_pages,
            pending_pdf_pages=pending_pages,
            ocr_blocked_pdf_pages=ocr_blocked_pages,
            rejected_pdf_pages=rejected_pages,
            missing_pdf_pages=missing_pages,
            duplicate_pdf_pages=duplicate_pages,
            unexpected_pdf_pages=unexpected_pages,
            mismatched_version_page_ids=mismatched_ids,
            block_reason=block_reason,
        )

    def evaluate_section(
        self,
        *,
        section_number: str,
        pages: Sequence[RegulatoryPage],
        structure_status: StructureStatus,
        policy: ValidationPolicy | None = None,
        version: DocumentVersion | None = None,
        all_version_pages: Sequence[RegulatoryPage] | None = None,
        document_assessment: DocumentVersionValidationAssessment | None = None,
    ) -> SectionValidationAssessment:
        """Avalia elegibilidade literal de uma seção condicionada à aprovação global da versão."""
        del policy  # Política global de aprovação documental prevalece sobre políticas parciais

        effective_doc_assessment = document_assessment
        if (
            effective_doc_assessment is None
            and version is not None
            and all_version_pages is not None
        ):
            effective_doc_assessment = self.evaluate_document_version(
                version=version,
                pages=all_version_pages,
            )

        if (
            effective_doc_assessment is not None
            and not effective_doc_assessment.literal_transcription_allowed
        ):
            agg_status = PageValidationStatus.PENDING_REVIEW
            if effective_doc_assessment.rejected_pdf_pages:
                agg_status = PageValidationStatus.REJECTED
            elif effective_doc_assessment.ocr_blocked_pdf_pages:
                agg_status = PageValidationStatus.PENDING_MANUAL_REVIEW

            return SectionValidationAssessment(
                literal_transcription_allowed=False,
                auxiliary_unreviewed_allowed=False,
                aggregate_status=agg_status,
                pending_pdf_pages=effective_doc_assessment.pending_pdf_pages,
                ocr_blocked_pdf_pages=effective_doc_assessment.ocr_blocked_pdf_pages,
                warning_message=effective_doc_assessment.block_reason
                or (
                    f"Seção {section_number} bloqueada: a versão documental não possui "
                    "todas as páginas aprovadas para transcrição literal."
                ),
                document_approved=False,
            )

        if not pages:
            return SectionValidationAssessment(
                literal_transcription_allowed=False,
                auxiliary_unreviewed_allowed=False,
                aggregate_status=PageValidationStatus.PENDING_REVIEW,
                pending_pdf_pages=(),
                ocr_blocked_pdf_pages=(),
                warning_message=(
                    f"A seção {section_number} não possui páginas de origem vinculadas; "
                    "transcrição literal indisponível."
                ),
                document_approved=effective_doc_assessment.literal_transcription_allowed
                if effective_doc_assessment is not None
                else False,
            )

        pending_pages = tuple(
            p.pdf_page_number
            for p in pages
            if p.validation_status != PageValidationStatus.APPROVED_FOR_LITERAL_TRANSCRIPTION
        )
        ocr_blocked_pages = tuple(
            p.pdf_page_number
            for p in pages
            if p.extraction_method
            in (ExtractionMethod.OCR_FALLBACK, ExtractionMethod.OCR_REQUIRED_UNAVAILABLE)
            and p.validation_status != PageValidationStatus.APPROVED_FOR_LITERAL_TRANSCRIPTION
        )
        rejected_pages = tuple(
            p.pdf_page_number
            for p in pages
            if p.validation_status == PageValidationStatus.REJECTED
        )

        if rejected_pages:
            return SectionValidationAssessment(
                literal_transcription_allowed=False,
                auxiliary_unreviewed_allowed=False,
                aggregate_status=PageValidationStatus.REJECTED,
                pending_pdf_pages=pending_pages,
                ocr_blocked_pdf_pages=ocr_blocked_pages,
                warning_message=(
                    f"Seção {section_number} contém página(s) reprovada(s) "
                    f"{list(rejected_pages)} e não pode ser apresentada como transcrição literal."
                ),
                document_approved=False,
            )

        if structure_status != StructureStatus.PARSED:
            return SectionValidationAssessment(
                literal_transcription_allowed=False,
                auxiliary_unreviewed_allowed=False,
                aggregate_status=PageValidationStatus.PENDING_REVIEW,
                pending_pdf_pages=pending_pages,
                ocr_blocked_pdf_pages=ocr_blocked_pages,
                warning_message=(
                    f"A estrutura automática da seção {section_number} está marcada como "
                    "pendente de validação; fidelidade literal não garantida."
                ),
                document_approved=effective_doc_assessment.literal_transcription_allowed
                if effective_doc_assessment is not None
                else False,
            )

        # Transcrição literal integral somente quando TODAS as páginas da seção (e da versão,
        # quando fornecida) estiverem em APPROVED_FOR_LITERAL_TRANSCRIPTION.
        if not pending_pages:
            return SectionValidationAssessment(
                literal_transcription_allowed=True,
                auxiliary_unreviewed_allowed=False,
                aggregate_status=PageValidationStatus.APPROVED_FOR_LITERAL_TRANSCRIPTION,
                pending_pdf_pages=(),
                ocr_blocked_pdf_pages=(),
                warning_message=None,
                document_approved=True,
            )

        if ocr_blocked_pages:
            return SectionValidationAssessment(
                literal_transcription_allowed=False,
                auxiliary_unreviewed_allowed=False,
                aggregate_status=PageValidationStatus.PENDING_MANUAL_REVIEW,
                pending_pdf_pages=pending_pages,
                ocr_blocked_pdf_pages=ocr_blocked_pages,
                warning_message=(
                    f"Seção {section_number} abrange página(s) dependentes de OCR não revisadas "
                    f"(PDF {list(ocr_blocked_pages)}). Transcrição literal bloqueada até revisão "
                    "humana explícita."
                ),
                document_approved=False,
            )

        return SectionValidationAssessment(
            literal_transcription_allowed=False,
            auxiliary_unreviewed_allowed=False,
            aggregate_status=PageValidationStatus.PENDING_REVIEW,
            pending_pdf_pages=pending_pages,
            ocr_blocked_pdf_pages=(),
            warning_message=(
                f"Seção {section_number} bloqueada pela política global de aprovação documental: "
                f"páginas PDF {list(pending_pages)} ainda pendentes de validação humana para "
                "transcrição literal."
            ),
            document_approved=False,
        )
