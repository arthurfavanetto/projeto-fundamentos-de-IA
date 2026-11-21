"""Formatador normativo que separa rigorosamente Texto Normativo (SSOT) e Explicação IA."""

from __future__ import annotations

from collections.abc import Sequence

from rbac_assistant.domain.entities.rag import (
    EvidenceTrustLevel,
    FusedEvidence,
    StructuredNormativeAnswer,
)

_TRUST_LABELS: dict[EvidenceTrustLevel, str] = {
    EvidenceTrustLevel.VALIDATED_LITERAL: "Aprovado para transcrição literal",
    EvidenceTrustLevel.PRELIMINARY_UNREVIEWED: (
        "Extração automática preliminar (não validada manualmente)"
    ),
    EvidenceTrustLevel.BLOCKED_OCR_PENDING: "Bloqueado — OCR pendente de revisão manual",
    EvidenceTrustLevel.BLOCKED_REJECTED: "Bloqueado — página(s) reprovada(s) na revisão",
    EvidenceTrustLevel.BLOCKED_UNVERIFIED: "Bloqueado — pendente de verificação",
}


def format_evidence_reference(evidence: FusedEvidence) -> str:
    """Formata citação rastreável completa com dupla paginação e estado de revisão."""
    pdf_pages_str = ", ".join(str(p) for p in evidence.pdf_pages)
    printed_pages_str = ", ".join(evidence.printed_pages)
    location_parts = [f"Seção {evidence.section_number}"]
    if evidence.subsection_number:
        location_parts.append(f"Subseção {evidence.subsection_number}")
    if evidence.paragraph_number:
        location_parts.append(f"Parágrafo {evidence.paragraph_number}")
    if evidence.item_number:
        location_parts.append(f"Item {evidence.item_number}")
    location = " / ".join(location_parts)
    trust_label = _TRUST_LABELS[evidence.trust_level]

    return (
        f"{evidence.authority} {evidence.document_type} {evidence.regulation_number} "
        f"(Emenda {evidence.amendment_number} | vigência: {evidence.version_status}) — "
        f"{location} — PDF p. [{pdf_pages_str}] | Impressa: [{printed_pages_str}] — "
        f"[{trust_label}]"
    )


def render_structured_answer_text(
    *,
    normative_text: str | None,
    preliminary_excerpt: str | None,
    explanation: str,
    references: Sequence[str],
    observations: Sequence[str],
) -> str:
    """Monta representação textual com os 4 blocos conceituais claramente separados."""
    sections: list[str] = []

    if normative_text is not None:
        sections.append(
            "=== TEXTO NORMATIVO (LITERAL VALIDADO — SQLite SSOT) ===\n"
            f"{normative_text}"
        )
    elif preliminary_excerpt is not None:
        sections.append(
            "=== LOCALIZAÇÃO PRELIMINAR (EXTRAÇÃO NÃO VALIDADA — NÃO É TRANSCRIÇÃO LITERAL) ===\n"
            f"{preliminary_excerpt}"
        )
    else:
        sections.append(
            "=== TEXTO NORMATIVO ===\n"
            "Transcrição literal validada indisponível para esta consulta."
        )

    sections.append(f"=== EXPLICAÇÃO (GERADA POR IA) ===\n{explanation}")

    if references:
        ref_lines = "\n".join(f"- {ref}" for ref in references)
    else:
        ref_lines = "- Nenhuma referência normativa validada disponível."
    sections.append(f"=== REFERÊNCIA ===\n{ref_lines}")

    if observations:
        obs_lines = "\n".join(f"- {obs}" for obs in observations)
        sections.append(f"=== OBSERVAÇÃO ===\n{obs_lines}")

    return "\n\n".join(sections)


class NormativeResponseFormatter:
    """Constrói `StructuredNormativeAnswer` preservando todas as invariantes de fidelidade."""

    def build_answer(
        self,
        *,
        query: str,
        usable_evidences: Sequence[FusedEvidence],
        blocked_evidences: Sequence[FusedEvidence],
        explanation: str,
        extra_observations: Sequence[str] = (),
        generation_model: str | None = None,
        used_fallback_without_llm: bool = False,
    ) -> StructuredNormativeAnswer:
        """Monta resposta final garantindo que `normative_text` só venha do SSOT aprovado."""
        validated = [
            ev
            for ev in usable_evidences
            if ev.trust_level == EvidenceTrustLevel.VALIDATED_LITERAL
            and ev.literal_transcription_allowed
        ]
        preliminary = [
            ev
            for ev in usable_evidences
            if ev.trust_level == EvidenceTrustLevel.PRELIMINARY_UNREVIEWED
        ]

        is_literal_validated = bool(validated)
        is_preliminary_only = (not validated) and bool(preliminary)
        found = bool(validated or preliminary)

        normative_text: str | None = None
        if validated:
            normative_blocks = [
                f"[{ev.document_type} {ev.regulation_number} — Emenda {ev.amendment_number} — "
                f"Seção {ev.section_number}]\n{ev.original_text}"
                for ev in validated
            ]
            normative_text = "\n\n---\n\n".join(normative_blocks)

        preliminary_excerpt: str | None = None
        if is_preliminary_only:
            prelim_blocks = [
                f"[LOCALIZAÇÃO PRELIMINAR — {ev.document_type} {ev.regulation_number} "
                f"Emenda {ev.amendment_number} — Seção {ev.section_number} — NÃO VALIDADO]\n"
                f"{ev.original_text}"
                for ev in preliminary
            ]
            preliminary_excerpt = "\n\n---\n\n".join(prelim_blocks)

        all_cited = [*validated, *preliminary, *blocked_evidences]
        references = tuple(format_evidence_reference(ev) for ev in all_cited)

        observations: list[str] = list(extra_observations)
        if is_preliminary_only:
            pending_details = ", ".join(
                f"{ev.document_type} {ev.regulation_number} seção {ev.section_number} "
                f"(PDF {list(ev.pending_pdf_pages)})"
                for ev in preliminary
            )
            observations.append(
                "AVISO DE REVISÃO (review_on_demand): O dispositivo foi localizado em "
                f"página(s) com verificação automática apenas ({pending_details}), ainda SEM "
                "revisão humana. Este conteúdo NÃO constitui transcrição literal validada nem "
                "interpretação oficial confirmada; valide as páginas antes de uso formal."
            )

        for ev in blocked_evidences:
            if ev.trust_level == EvidenceTrustLevel.BLOCKED_OCR_PENDING:
                observations.append(
                    f"BLOQUEIO OCR: {ev.document_type} {ev.regulation_number} seção "
                    f"{ev.section_number} depende de página(s) OCR pendentes de revisão manual "
                    f"(PDF {list(ev.ocr_blocked_pdf_pages)}) e NÃO foi utilizada para fundamentar "
                    "a resposta."
                )
            elif ev.trust_level == EvidenceTrustLevel.BLOCKED_REJECTED:
                observations.append(
                    f"BLOQUEIO REPROVAÇÃO: {ev.document_type} {ev.regulation_number} seção "
                    f"{ev.section_number} abrange página(s) reprovada(s) "
                    f"(PDF {list(ev.rejected_pdf_pages)}) e foi descartada."
                )
            elif ev.warning_message:
                observations.append(ev.warning_message)

        non_current = [ev for ev in (*validated, *preliminary) if ev.version_status != "current"]
        if non_current:
            versions_desc = ", ".join(
                f"{ev.document_type} {ev.regulation_number} Emenda {ev.amendment_number} "
                f"({ev.version_status})"
                for ev in non_current
            )
            observations.append(
                f"AVISO DE VIGÊNCIA: Esta resposta inclui versão marcada como não vigente: "
                f"{versions_desc}."
            )

        deduped_observations = tuple(dict.fromkeys(observations))
        formatted_text = render_structured_answer_text(
            normative_text=normative_text,
            preliminary_excerpt=preliminary_excerpt,
            explanation=explanation,
            references=references,
            observations=deduped_observations,
        )

        return StructuredNormativeAnswer(
            query=query,
            found=found,
            normative_text=normative_text,
            preliminary_excerpt=preliminary_excerpt,
            explanation=explanation,
            references=references,
            observations=deduped_observations,
            evidences=tuple(usable_evidences),
            blocked_evidences=tuple(blocked_evidences),
            is_literal_validated=is_literal_validated,
            is_preliminary_only=is_preliminary_only,
            has_blocked_sources=bool(blocked_evidences),
            generation_model=generation_model,
            used_fallback_without_llm=used_fallback_without_llm,
            formatted_text=formatted_text,
        )
