"""Validador determinístico server-side de citações produzidas pelo modelo generativo.

Audita de forma independente:
1. Identificadores de evidência (`[EV-1]`, `[EV-2]`);
2. Referências estruturais normativas (seção, regulamento, apêndice, artigo);
3. Referências de páginas físicas e impressas (`página 99`, `PDF p. 99`, `pág. 2/10`).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from rbac_assistant.domain.entities.rag import CitationValidationResult, FusedEvidence
from rbac_assistant.rag.retrieval.section_resolver import resolve_structural_references

_EV_TAG_RE = re.compile(r"\[EV-(\d+)\]", re.IGNORECASE)
_PAGE_CITATION_RE = re.compile(
    r"\b(?:PDF\s+p\.?\s*|p[áa]ginas?\s+|p[áa]g\.?\s+|fl\.?\s+|p\.\s+)"
    r"\[?(?P<page>\d+(?:/\d+)?)\]?",
    re.IGNORECASE,
)


class CitationValidator:
    """Verifica se `[EV-N]`, seções e páginas na explicação IA pertencem às evidências."""

    def validate(
        self,
        *,
        explanation: str,
        evidence_map: Mapping[str, FusedEvidence],
    ) -> CitationValidationResult:
        """Audita e sanitiza citações na explicação gerada sem alterar o texto literal do SSOT."""
        evidences = tuple(evidence_map.values())
        allowed_ev_ids = {ev_id.upper() for ev_id in evidence_map}
        allowed_sections, allowed_reg_sections = self._collect_grounded_sections(evidences)
        allowed_pages = self._collect_grounded_pages(evidences)

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
        valid_sections: list[str] = []
        unverified_sections: list[str] = []

        for ref in cited_refs:
            if not ref.section_number:
                continue
            sec_lower = ref.section_number.lower()
            reg_lower = ref.regulation_number.lower() if ref.regulation_number else None
            label = (
                ref.raw_match
                if ref.raw_match
                else f"{ref.regulation_number or ''} {ref.section_number}".strip()
            )

            is_grounded = False
            if reg_lower is not None:
                if (reg_lower, sec_lower) in allowed_reg_sections:
                    is_grounded = True
            elif sec_lower in allowed_sections:
                is_grounded = True

            if is_grounded:
                if label not in valid_sections:
                    valid_sections.append(label)
            elif label not in unverified_sections:
                unverified_sections.append(label)

        for raw_unverified in sorted(unverified_sections, key=len, reverse=True):
            sanitized = re.sub(
                re.escape(raw_unverified),
                "[citação descartada: não confirmada nas evidências recuperadas]",
                sanitized,
                flags=re.IGNORECASE,
            )

        # Audita referências explícitas de páginas (PDF e impressas)
        valid_pages: list[str] = []
        unverified_pages: list[str] = []

        def _replace_page_ref(match: re.Match[str]) -> str:
            page_token = match.group("page").strip()
            raw_phrase = match.group(0).strip()
            if page_token in allowed_pages:
                if raw_phrase not in valid_pages:
                    valid_pages.append(raw_phrase)
                return match.group(0)
            if raw_phrase not in unverified_pages:
                unverified_pages.append(raw_phrase)
            return "[página descartada: não confirmada nas evidências recuperadas]"

        sanitized = _PAGE_CITATION_RE.sub(_replace_page_ref, sanitized)

        # Limpa eventuais espaços duplicados deixados por tags [EV-N] removidas
        sanitized = re.sub(r"[ \t]{2,}", " ", sanitized).strip()

        total_citations = (
            len(valid_ev_ids)
            + len(invalid_ev_ids)
            + len(valid_sections)
            + len(unverified_sections)
            + len(valid_pages)
            + len(unverified_pages)
        )
        valid_total = len(valid_ev_ids) + len(valid_sections) + len(valid_pages)
        precision = (valid_total / total_citations) if total_citations > 0 else 1.0
        had_modifications = bool(invalid_ev_ids or unverified_sections or unverified_pages)

        return CitationValidationResult(
            sanitized_explanation=sanitized,
            valid_evidence_ids=tuple(valid_ev_ids),
            invalid_evidence_ids=tuple(invalid_ev_ids),
            valid_section_citations=tuple(valid_sections),
            unverified_section_citations=tuple(unverified_sections),
            valid_page_citations=tuple(valid_pages),
            unverified_page_citations=tuple(unverified_pages),
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

        return allowed_sections, allowed_reg_sections

    @staticmethod
    def _collect_grounded_pages(evidences: Sequence[FusedEvidence]) -> set[str]:
        """Coleta páginas físicas do PDF e rótulos impressos presentes nas evidências."""
        pages: set[str] = set()
        for ev in evidences:
            for p_num in ev.pdf_pages:
                pages.add(str(p_num))
            for printed in ev.printed_pages:
                cleaned = printed.strip()
                if cleaned:
                    pages.add(cleaned)
                    # Também permite citar apenas o número antes da barra (ex.: "10" de "10/25")
                    if "/" in cleaned:
                        pages.add(cleaned.split("/", 1)[0].strip())
        return pages
