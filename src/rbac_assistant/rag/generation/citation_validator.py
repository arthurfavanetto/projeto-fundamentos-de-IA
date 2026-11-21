"""Validador determinístico server-side de citações produzidas pelo modelo generativo."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from rbac_assistant.domain.entities.rag import CitationValidationResult, FusedEvidence
from rbac_assistant.rag.retrieval.section_resolver import resolve_structural_references

_EV_TAG_RE = re.compile(r"\[EV-(\d+)\]", re.IGNORECASE)


class CitationValidator:
    """Verifica se todas as citações `[EV-N]` e seções na explicação IA pertencem às evidências."""

    def validate(
        self,
        *,
        explanation: str,
        evidence_map: Mapping[str, FusedEvidence],
    ) -> CitationValidationResult:
        """Audita e sanitiza citações na explicação gerada sem alterar o texto literal do SSOT."""
        allowed_ev_ids = {ev_id.upper() for ev_id in evidence_map}
        allowed_sections, allowed_reg_sections = self._collect_grounded_sections(
            tuple(evidence_map.values())
        )

        valid_ev_ids: list[str] = []
        invalid_ev_ids: list[str] = []

        def _replace_ev_tag(match: re.Match[str]) -> str:
            ev_id = f"EV-{match.group(1)}"
            if ev_id in allowed_ev_ids:
                if ev_id not in valid_ev_ids:
                    valid_ev_ids.append(ev_id)
                return f"[{ev_id}]"
            if ev_id not in invalid_ev_ids:
                invalid_ev_ids.append(ev_id)
            return ""

        sanitized = _EV_TAG_RE.sub(_replace_ev_tag, explanation)

        # Audita referências normativas explícitas (ex.: "RBAC 43.99", "seção 145.999", "Art. 9º")
        cited_refs = resolve_structural_references(sanitized)
        valid_section_count = 0
        unverified_sections: list[str] = []

        for ref in cited_refs:
            if not ref.section_number:
                continue
            sec_lower = ref.section_number.lower()
            reg_lower = ref.regulation_number.lower() if ref.regulation_number else None

            is_grounded = False
            if reg_lower is not None:
                if (reg_lower, sec_lower) in allowed_reg_sections or (
                    None,
                    sec_lower,
                ) in allowed_reg_sections:
                    is_grounded = True
            elif sec_lower in allowed_sections:
                is_grounded = True

            if is_grounded:
                valid_section_count += 1
            else:
                label = (
                    f"{ref.raw_match}"
                    if ref.raw_match
                    else f"{ref.regulation_number or ''} {ref.section_number}".strip()
                )
                if label not in unverified_sections:
                    unverified_sections.append(label)

        for raw_unverified in sorted(unverified_sections, key=len, reverse=True):
            sanitized = re.sub(
                re.escape(raw_unverified),
                "[citação descartada: não confirmada nas evidências recuperadas]",
                sanitized,
                flags=re.IGNORECASE,
            )

        # Limpa eventuais espaços duplicados deixados por tags [EV-N] removidas
        sanitized = re.sub(r"[ \t]{2,}", " ", sanitized).strip()

        total_citations = (
            len(valid_ev_ids)
            + len(invalid_ev_ids)
            + valid_section_count
            + len(unverified_sections)
        )
        valid_total = len(valid_ev_ids) + valid_section_count
        precision = (valid_total / total_citations) if total_citations > 0 else 1.0
        had_modifications = bool(invalid_ev_ids or unverified_sections)

        return CitationValidationResult(
            sanitized_explanation=sanitized,
            valid_evidence_ids=tuple(valid_ev_ids),
            invalid_evidence_ids=tuple(invalid_ev_ids),
            unverified_section_citations=tuple(unverified_sections),
            citation_precision=precision,
            had_modifications=had_modifications,
        )

    @staticmethod
    def _collect_grounded_sections(
        evidences: Sequence[FusedEvidence],
    ) -> tuple[set[str], set[tuple[str | None, str]]]:
        """Coleta seções diretamente recuperadas e referências cruzadas literais no SSOT."""
        allowed_sections: set[str] = set()
        allowed_reg_sections: set[tuple[str | None, str]] = set()

        for ev in evidences:
            sec = ev.section_number.strip().lower()
            reg = ev.regulation_number.strip().lower() or None
            if sec:
                allowed_sections.add(sec)
                allowed_reg_sections.add((reg, sec))
                allowed_reg_sections.add((None, sec))

            # Inclui referências cruzadas literalmente presentes no texto oficial da evidência
            for cross_ref in resolve_structural_references(ev.original_text):
                if cross_ref.section_number:
                    c_sec = cross_ref.section_number.strip().lower()
                    c_reg = (
                        cross_ref.regulation_number.strip().lower()
                        if cross_ref.regulation_number
                        else reg
                    )
                    allowed_sections.add(c_sec)
                    allowed_reg_sections.add((c_reg, c_sec))
                    allowed_reg_sections.add((None, c_sec))

        return allowed_sections, allowed_reg_sections
