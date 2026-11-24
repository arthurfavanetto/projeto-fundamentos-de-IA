"""Formatador normativo que separa rigorosamente Texto Normativo (SSOT) e Explicação IA."""

from __future__ import annotations

import re
from collections.abc import Sequence

from rbac_assistant.domain.entities.normative_unit import render_structured_table
from rbac_assistant.domain.entities.rag import (
    EvidenceTrustLevel,
    FusedEvidence,
    RetrievalDiagnostics,
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

EXTERNAL_EXCERPT_SEPARATOR = "\n\n[SEPARADOR EXTERNO ENTRE TRECHOS NORMATIVOS]\n\n"
LITERAL_START_MARKER = "<<<INÍCIO_TRANSCRIÇÃO_LITERAL>>>"
LITERAL_END_MARKER = "<<<FIM_TRANSCRIÇÃO_LITERAL>>>"
_LITERAL_EXTRACT_RE = re.compile(
    re.escape(LITERAL_START_MARKER) + r"\n(.*?)\n" + re.escape(LITERAL_END_MARKER),
    re.DOTALL,
)


def extract_literal_excerpts_from_text(rendered_text: str) -> tuple[str, ...]:
    """Extrai exatamente os trechos normativos autoritativos delimitados no texto renderizado."""
    return tuple(match.group(1) for match in _LITERAL_EXTRACT_RE.finditer(rendered_text))


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
    structured_tables_text: str | None = None,
    include_explanation_block: bool = True,
    explanation_heading: str = "=== EXPLICAÇÃO (GERADA POR IA) ===",
) -> str:
    """Monta representação textual com blocos separados, sem nunca expor conteúdo preliminar."""
    del preliminary_excerpt  # Excertos preliminares não revisados nunca são exibidos
    sections: list[str] = []

    if normative_text is not None:
        sections.append(
            "=== TEXTO NORMATIVO (LITERAL VALIDADO — SQLite SSOT) ===\n"
            f"{normative_text}"
        )
    else:
        sections.append(
            "=== TEXTO NORMATIVO ===\n"
            "Transcrição literal validada indisponível para esta consulta."
        )

    if structured_tables_text:
        sections.append(
            "=== TABELAS E FORMULÁRIOS ESTRUTURADOS (SQLite SSOT) ===\n"
            f"{structured_tables_text}"
        )

    if include_explanation_block and explanation:
        sections.append(f"{explanation_heading}\n{explanation}")

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
        has_blocked_sources: bool = False,
        evidences_sent_to_llm: Sequence[str] = (),
        used_evidence_ids: Sequence[str] = (),
        discarded_citations: Sequence[str] = (),
        diagnostics: RetrievalDiagnostics | None = None,
    ) -> StructuredNormativeAnswer:
        """Monta resposta final garantindo que somente evidências aprovadas sejam exibidas."""
        validated = [
            ev
            for ev in usable_evidences
            if ev.trust_level == EvidenceTrustLevel.VALIDATED_LITERAL
            and ev.literal_transcription_allowed
        ]
        suppressed_unapproved = len(usable_evidences) > len(validated)

        is_literal_validated = bool(validated)
        found = bool(validated)

        literal_excerpts = tuple(ev.original_text for ev in validated)
        normative_text: str | None = None
        if validated:
            normative_blocks = [
                f"[IDENTIFICAÇÃO EXTERNA — {ev.document_type} {ev.regulation_number} | "
                f"Emenda {ev.amendment_number} | Seção {ev.section_number}]\n"
                f"{LITERAL_START_MARKER}\n{ev.original_text}\n{LITERAL_END_MARKER}"
                for ev in validated
            ]
            normative_text = EXTERNAL_EXCERPT_SEPARATOR.join(normative_blocks)

        rendered_tables: list[str] = []
        seen_table_ids: set[str] = set()
        observations: list[str] = list(extra_observations)

        for ev in validated:
            for tbl in ev.tables:
                if tbl.table_id in seen_table_ids:
                    continue
                seen_table_ids.add(tbl.table_id)
                rendered = render_structured_table(tbl)
                if rendered:
                    rendered_tables.append(rendered)
                if not tbl.visual_layout_preserved and tbl.layout_fidelity_note:
                    observations.append(
                        f"AVISO DE FORMULÁRIO (PDF p. {tbl.pdf_page_number}): "
                        f"{tbl.layout_fidelity_note}"
                    )

        structured_tables_text = "\n\n".join(rendered_tables) if rendered_tables else None
        references = tuple(format_evidence_reference(ev) for ev in validated)

        non_current = [ev for ev in validated if ev.version_status != "current"]
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

        # Em consultas puramente literais sem solicitação ao LLM, não inclui bloco de IA
        is_literal_only_no_llm = (
            found and used_fallback_without_llm and generation_model is None
        )
        include_explanation_block = not is_literal_only_no_llm
        explanation_heading = (
            "=== EXPLICAÇÃO (GERADA POR IA) ==="
            if found
            else "=== INFORMAÇÃO / RECUSA FUNDAMENTADA ==="
        )
        formatted_text = render_structured_answer_text(
            normative_text=normative_text,
            preliminary_excerpt=None,
            explanation=explanation,
            references=references,
            observations=deduped_observations,
            structured_tables_text=structured_tables_text,
            include_explanation_block=include_explanation_block,
            explanation_heading=explanation_heading,
        )

        return StructuredNormativeAnswer(
            query=query,
            found=found,
            normative_text=normative_text,
            preliminary_excerpt=None,
            explanation=explanation,
            references=references,
            observations=deduped_observations,
            evidences=tuple(validated),
            blocked_evidences=(),
            is_literal_validated=is_literal_validated,
            is_preliminary_only=False,
            has_blocked_sources=(
                bool(has_blocked_sources) or bool(blocked_evidences) or suppressed_unapproved
            ),
            generation_model=generation_model,
            used_fallback_without_llm=used_fallback_without_llm,
            formatted_text=formatted_text,
            evidences_sent_to_llm=tuple(evidences_sent_to_llm),
            used_evidence_ids=tuple(used_evidence_ids),
            discarded_citations=tuple(discarded_citations),
            diagnostics=diagnostics,
            literal_excerpts=literal_excerpts,
            structured_tables_text=structured_tables_text,
        )
