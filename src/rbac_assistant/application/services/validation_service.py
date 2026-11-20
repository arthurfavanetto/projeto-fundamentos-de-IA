"""Serviço de validação híbrida por página e seção (`review_on_demand` como padrão).

Políticas suportadas:
- `strict_all_pages`: todas as páginas exigem aprovação humana explícita.
- `review_on_demand` (padrão): páginas nativas são verificadas automaticamente
  (`native_quality_checked`), mas a transcrição literal exige aprovação humana das
  páginas da seção sob demanda; páginas OCR ficam sempre bloqueadas até revisão manual.
- `native_auto_ocr_manual`: páginas nativas com qualidade verificada podem ser usadas
  como conteúdo auxiliar claramente rotulado como *extração não revisada*; transcrição
  literal verificada continua exigindo aprovação humana e páginas OCR seguem bloqueadas.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

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
class SectionValidationAssessment:
    """Resultado da avaliação de elegibilidade literal de uma seção multi-página."""

    literal_transcription_allowed: bool
    auxiliary_unreviewed_allowed: bool
    aggregate_status: PageValidationStatus
    pending_pdf_pages: tuple[int, ...]
    ocr_blocked_pdf_pages: tuple[int, ...]
    warning_message: str | None


class ValidationService:
    """Aplica as regras de transição de estado e validação híbrida por página/seção."""

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

    def evaluate_section(
        self,
        *,
        section_number: str,
        pages: Sequence[RegulatoryPage],
        structure_status: StructureStatus,
        policy: ValidationPolicy | None = None,
    ) -> SectionValidationAssessment:
        """Avalia todas as páginas de uma seção segundo a política configurada."""
        active_policy = policy or self._policy

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
            )

        # Transcrição literal integral somente quando TODAS as páginas estiverem aprovadas
        if not pending_pages:
            return SectionValidationAssessment(
                literal_transcription_allowed=True,
                auxiliary_unreviewed_allowed=True,
                aggregate_status=PageValidationStatus.APPROVED_FOR_LITERAL_TRANSCRIPTION,
                pending_pdf_pages=(),
                ocr_blocked_pdf_pages=(),
                warning_message=None,
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
            )

        all_native_checked = all(
            p.validation_status
            in (
                PageValidationStatus.NATIVE_QUALITY_CHECKED,
                PageValidationStatus.MANUALLY_APPROVED,
                PageValidationStatus.APPROVED_FOR_LITERAL_TRANSCRIPTION,
            )
            for p in pages
        )

        if active_policy == ValidationPolicy.NATIVE_AUTO_OCR_MANUAL and all_native_checked:
            return SectionValidationAssessment(
                literal_transcription_allowed=False,
                auxiliary_unreviewed_allowed=True,
                aggregate_status=PageValidationStatus.NATIVE_QUALITY_CHECKED,
                pending_pdf_pages=pending_pages,
                ocr_blocked_pdf_pages=(),
                warning_message=(
                    f"Seção {section_number} possui páginas com verificação automatizada "
                    f"(PDF {list(pending_pages)}), ainda sem revisão humana. Conteúdo disponível "
                    "apenas como extração não revisada (não equivale a transcrição literal "
                    "verificada)."
                ),
            )

        return SectionValidationAssessment(
            literal_transcription_allowed=False,
            auxiliary_unreviewed_allowed=all_native_checked,
            aggregate_status=(
                PageValidationStatus.NATIVE_QUALITY_CHECKED
                if all_native_checked
                else PageValidationStatus.PENDING_REVIEW
            ),
            pending_pdf_pages=pending_pages,
            ocr_blocked_pdf_pages=(),
            warning_message=(
                f"Seção {section_number} localizada, porém pendente de validação humana nas "
                f"páginas PDF {list(pending_pages)} (política '{active_policy.value}'). "
                "Não pode ser apresentada como transcrição literal integral verificada."
            ),
        )
