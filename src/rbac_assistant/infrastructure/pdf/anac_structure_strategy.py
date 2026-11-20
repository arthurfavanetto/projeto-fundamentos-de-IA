"""Estratégia estrutural genérica para Regulamentos Brasileiros da Aviação Civil (ANAC RBAC).

Funciona para qualquer RBAC (43, 145, 65, 91, 21, etc.) de forma orientada a metadados,
sem lógica exclusiva de um único regulamento.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from rbac_assistant.domain.entities.normative_unit import (
    NormativeReference,
    NormativeTable,
    NormativeUnitType,
    StructureStatus,
)
from rbac_assistant.domain.entities.regulatory_document import DocumentType
from rbac_assistant.domain.entities.regulatory_page import RegulatoryPage
from rbac_assistant.domain.ports.validation_port import ParsedUnitRecord
from rbac_assistant.infrastructure.pdf.pdfplumber_extractor import normalize_auxiliary_text

# Cabeçalhos e rodapés repetitivos dos PDFs oficiais da ANAC
_RUNNING_HEADER_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^Data da emiss[ãa]o\s*:.*RBAC\s+n[º°]?\s*\S+", re.IGNORECASE),
    re.compile(r"^Data de vig[êe]ncia\s*:.*Emenda\s+n[º°]?\s*\S+", re.IGNORECASE),
    re.compile(r"^Origem\s*:\s*[A-Z0-9/\-]+\s+\d+\s*/\s*\d+\s*$", re.IGNORECASE),
)

_SUBPART_RE = re.compile(r"^SUBPARTE\s+([A-Z0-9-]+)(?:\s*[-–—]\s*(.*))?$", re.IGNORECASE)
_APPENDIX_BANNER_RE = re.compile(
    r"^AP[ÊE]NDICE\s+([A-Z0-9-]+)(?:\s+DO\s+RBAC\s+(?:n[º°]?\s*)?\d+)?(.*)$",
    re.IGNORECASE,
)
_PARAGRAPH_RE = re.compile(r"^\(([a-z])\)(?:-[A-Z])?\s+")
_INCISO_RE = re.compile(r"^\((\d+)\)(?:-[A-Z])?\s+")
_ALINEA_RE = re.compile(r"^\(([ivxlcdm]+)\)\s+")
_SUBITEM_RE = re.compile(r"^\(([A-Z])\)\s+")
_CROSS_REF_RE = re.compile(
    r"\b(?:RBAC|RBHA|IS)\s*(?:n[º°]?\s*)?(\d+(?:-[A-Z0-9]+)?)"
    r"|\b(?:se[çc][ãa]o|par[áa]grafo)\s+([A-Z]?\d+\.\d+(?:-[A-Z0-9]+)?(?:\([a-z0-9]+\))*)",
    re.IGNORECASE,
)


@dataclass
class _LineEntry:
    pdf_page_number: int
    text: str


@dataclass
class _SectionBuffer:
    section_number: str
    title: str
    subpart: str
    unit_type: NormativeUnitType
    lines: list[_LineEntry] = field(default_factory=list)


def _is_running_chrome(line: str) -> bool:
    stripped = line.strip()
    return any(pattern.match(stripped) for pattern in _RUNNING_HEADER_PATTERNS)


def _build_section_regex(regulation_number: str) -> re.Pattern[str]:
    escaped_reg = re.escape(regulation_number.strip())
    # Títulos de seção da ANAC iniciam por letra maiúscula ou '[Reservado]'; evita falsos
    # positivos quando uma referência numérica (ex.: "43.17 podem...") cai no início da linha.
    return re.compile(
        rf"^([A-Z]?{escaped_reg}\.\d+(?:-[A-Z0-9]+)?)\s+([A-ZÁÀÂÃÉÊÍÓÔÕÚÇ\[].+)$"
    )


def _extract_references(unit_id: str, text: str) -> tuple[NormativeReference, ...]:
    refs: list[NormativeReference] = []
    seen: set[tuple[str, str]] = set()
    for match in _CROSS_REF_RE.finditer(text):
        raw = match.group(0).strip()
        if match.group(1):
            target_reg = raw.split()[0].upper() + " " + match.group(1)
            target_sec = ""
        else:
            target_reg = "SELF"
            target_sec = match.group(2)
        key = (target_reg, target_sec)
        if key in seen:
            continue
        seen.add(key)
        ref_hash = hashlib.sha256(f"{unit_id}:{target_reg}:{target_sec}".encode()).hexdigest()[:12]
        refs.append(
            NormativeReference(
                reference_id=f"ref_{ref_hash}",
                unit_id=unit_id,
                target_regulation=target_reg,
                target_section=target_sec,
                raw_citation=raw,
            )
        )
    return tuple(refs)


def _looks_like_toc_section(
    lines: list[_LineEntry],
    idx: int,
    section_re: re.Pattern[str],
    seen_toc_numbers: set[str],
) -> bool:
    """Identifica se um cabeçalho de seção ainda pertence ao Sumário (TOC) inicial.

    Um cabeçalho no corpo regulatório ou (a) repete um número já listado no sumário, ou
    (b) é seguido em até 4 linhas (título quebrado) por parágrafo `(a)` / texto de corpo
    com frase completa (terminada em ponto) antes da próxima seção.
    """
    match = section_re.match(lines[idx].text.strip())
    if not match:
        return False
    sec_num = match.group(1).strip()

    # Se este número já apareceu no sumário antes, estamos agora no corpo real!
    if sec_num in seen_toc_numbers:
        return False

    current = lines[idx].text.strip()
    if "...." in current or ". . ." in current:
        return True

    # Examina as linhas até a próxima seção/apêndice/subparte (limite de 6 linhas)
    between: list[str] = []
    for candidate in lines[idx + 1 : idx + 7]:
        text = candidate.text.strip()
        if section_re.match(text) or _APPENDIX_BANNER_RE.match(text) or _SUBPART_RE.match(text):
            break
        between.append(text)

    # Se há parágrafo normativo "(a) ..." ou "[Reservado]" ou frase normativa com ponto final
    # e mais de 60 caracteres, já é corpo regulatório (ex.: documentos sintéticos sem sumário).
    if "[Reservado]" in current:
        return False
    for line in between:
        if _PARAGRAPH_RE.match(line) or _INCISO_RE.match(line):
            return False
        if line.endswith(".") and len(line) > 50:
            return False

    # Caso contrário, só há linhas curtas de continuação de título antes da próxima seção -> TOC
    return True


class AnacRbacStructureStrategy:
    """Parser estrutural genérico para qualquer RBAC publicado pela ANAC."""

    def supports(self, *, authority: str, document_type: DocumentType) -> bool:
        return authority.strip().upper() == "ANAC" and document_type == DocumentType.RBAC

    def parse(
        self,
        *,
        version_id: str,
        regulation_number: str,
        pages: Sequence[RegulatoryPage],
        tables: Sequence[NormativeTable] = (),
    ) -> list[ParsedUnitRecord]:
        del tables
        section_re = _build_section_regex(regulation_number)

        # Coleta linhas preservando ordem exata, removendo apenas cabeçalhos/rodapés repetitivos
        all_lines: list[_LineEntry] = []
        for page in sorted(pages, key=lambda p: p.pdf_page_number):
            for raw_line in page.original_text.splitlines():
                if _is_running_chrome(raw_line):
                    continue
                if not raw_line.strip():
                    continue
                all_lines.append(_LineEntry(pdf_page_number=page.pdf_page_number, text=raw_line))

        if not all_lines:
            return []

        buffers: list[_SectionBuffer] = []
        current_buffer: _SectionBuffer | None = None
        current_subpart = ""
        entered_body = False
        seen_toc_numbers: set[str] = set()

        idx = 0
        while idx < len(all_lines):
            entry = all_lines[idx]
            stripped = entry.text.strip()

            subpart_match = _SUBPART_RE.match(stripped)
            if subpart_match and entered_body:
                sub_code = subpart_match.group(1).strip()
                sub_title = (subpart_match.group(2) or "").strip()
                current_subpart = f"Subparte {sub_code}" + (f" — {sub_title}" if sub_title else "")
                idx += 1
                continue

            sec_match = section_re.match(stripped)
            if sec_match:
                sec_num = sec_match.group(1).strip()
                if not entered_body and _looks_like_toc_section(
                    all_lines, idx, section_re, seen_toc_numbers
                ):
                    seen_toc_numbers.add(sec_num)
                    idx += 1
                    continue

                entered_body = True
                sec_title = sec_match.group(2).strip()
                unit_type = (
                    NormativeUnitType.APPENDIX
                    if sec_num[0].isalpha()
                    else NormativeUnitType.SECTION
                )
                current_buffer = _SectionBuffer(
                    section_number=sec_num,
                    title=sec_title,
                    subpart=current_subpart,
                    unit_type=unit_type,
                    lines=[entry],
                )
                buffers.append(current_buffer)
                idx += 1
                continue

            if _APPENDIX_BANNER_RE.match(stripped):
                if not entered_body:
                    idx += 1
                    continue
                upcoming = [ln.text.strip() for ln in all_lines[idx + 1 : idx + 4]]
                if any(section_re.match(u) for u in upcoming):
                    current_subpart = stripped
                    idx += 1
                    continue

            if current_buffer is not None:
                current_buffer.lines.append(entry)
            idx += 1

        # Fallback conservador: se nenhuma seção foi identificada, não inventa divisão
        if not buffers:
            full_orig = "\n".join(e.text for e in all_lines)
            full_norm = normalize_auxiliary_text(full_orig)
            page_nums = tuple(sorted({e.pdf_page_number for e in all_lines}))
            unit_id = f"{version_id}::unstructured_0"
            return [
                ParsedUnitRecord(
                    unit_id=unit_id,
                    version_id=version_id,
                    parent_unit_id=None,
                    unit_type=NormativeUnitType.PREAMBLE,
                    section_number="UNSTRUCTURED",
                    subsection_number="",
                    paragraph_number="",
                    item_number="",
                    title="Conteúdo pendente de delimitação estrutural",
                    subpart="",
                    original_text=full_orig,
                    normalized_text=full_norm,
                    start_page=page_nums[0],
                    end_page=page_nums[-1],
                    page_numbers=page_nums,
                    order_index=0,
                    structure_status=StructureStatus.PENDING_VALIDATION,
                    content_hash=hashlib.sha256(full_orig.encode("utf-8")).hexdigest(),
                )
            ]

        records: list[ParsedUnitRecord] = []
        seen_unit_ids: set[str] = set()
        order_counter = 0

        for sec_buf in buffers:
            sec_orig = "\n".join(e.text for e in sec_buf.lines)
            sec_norm = normalize_auxiliary_text(sec_orig)
            sec_pages = tuple(sorted({e.pdf_page_number for e in sec_buf.lines}))
            base_unit_id = f"{version_id}::{sec_buf.section_number}"
            is_duplicate = base_unit_id in seen_unit_ids
            sec_unit_id = (
                f"{base_unit_id}::dup_{order_counter}" if is_duplicate else base_unit_id
            )
            seen_unit_ids.add(sec_unit_id)

            sec_hash = hashlib.sha256(sec_orig.encode("utf-8")).hexdigest()
            refs = _extract_references(sec_unit_id, sec_orig)

            sec_status = (
                StructureStatus.PARSED
                if (len(sec_buf.lines) > 1 or "[Reservado]" in sec_orig) and not is_duplicate
                else StructureStatus.PENDING_VALIDATION
            )

            records.append(
                ParsedUnitRecord(
                    unit_id=sec_unit_id,
                    version_id=version_id,
                    parent_unit_id=None,
                    unit_type=sec_buf.unit_type,
                    section_number=sec_buf.section_number,
                    subsection_number="",
                    paragraph_number="",
                    item_number="",
                    title=sec_buf.title,
                    subpart=sec_buf.subpart,
                    original_text=sec_orig,
                    normalized_text=sec_norm,
                    start_page=sec_pages[0],
                    end_page=sec_pages[-1],
                    page_numbers=sec_pages,
                    order_index=order_counter,
                    structure_status=sec_status,
                    content_hash=sec_hash,
                    references=refs,
                )
            )
            order_counter += 1

            sub_records, order_counter = self._extract_subunits(
                version_id=version_id,
                parent_section_id=sec_unit_id,
                sec_buf=sec_buf,
                start_order=order_counter,
            )
            records.extend(sub_records)

        return records

    @staticmethod
    def _extract_subunits(
        *,
        version_id: str,
        parent_section_id: str,
        sec_buf: _SectionBuffer,
        start_order: int,
    ) -> tuple[list[ParsedUnitRecord], int]:
        """Segmenta parágrafos `(a)`, incisos `(1)`, alíneas `(i)` e subitens `(A)` de uma seção."""
        results: list[ParsedUnitRecord] = []
        order = start_order
        current_paragraph = ""
        current_inciso = ""
        current_sub_type: NormativeUnitType | None = None
        current_item_label = ""
        current_lines: list[_LineEntry] = []

        def flush_current() -> None:
            nonlocal order
            if current_sub_type is None or not current_lines:
                return
            text = "\n".join(e.text for e in current_lines)
            norm = normalize_auxiliary_text(text)
            pages = tuple(sorted({e.pdf_page_number for e in current_lines}))
            uid = f"{parent_section_id}::{order}"
            results.append(
                ParsedUnitRecord(
                    unit_id=uid,
                    version_id=version_id,
                    parent_unit_id=parent_section_id,
                    unit_type=current_sub_type,
                    section_number=sec_buf.section_number,
                    subsection_number="",
                    paragraph_number=current_paragraph,
                    item_number=current_item_label,
                    title="",
                    subpart=sec_buf.subpart,
                    original_text=text,
                    normalized_text=norm,
                    start_page=pages[0],
                    end_page=pages[-1],
                    page_numbers=pages,
                    order_index=order,
                    structure_status=StructureStatus.PARSED,
                    content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                )
            )
            order += 1

        for entry in sec_buf.lines[1:]:
            stripped = entry.text.strip()
            par_match = _PARAGRAPH_RE.match(stripped)
            inc_match = _INCISO_RE.match(stripped)
            ali_match = _ALINEA_RE.match(stripped)
            sub_match = _SUBITEM_RE.match(stripped)

            is_alinea = (
                ali_match is not None
                and bool(current_inciso)
                and (ali_match.group(1) != "i" or current_paragraph != "(h)")
            )

            if is_alinea and ali_match:
                flush_current()
                current_sub_type = NormativeUnitType.ALINEA
                current_item_label = f"({current_inciso})({ali_match.group(1)})"
                current_lines = [entry]
            elif par_match:
                flush_current()
                current_paragraph = f"({par_match.group(1)})"
                current_inciso = ""
                current_sub_type = NormativeUnitType.PARAGRAPH
                current_item_label = ""
                current_lines = [entry]
            elif inc_match:
                flush_current()
                current_inciso = inc_match.group(1)
                current_sub_type = NormativeUnitType.INCISO
                current_item_label = f"({current_inciso})"
                current_lines = [entry]
            elif sub_match:
                flush_current()
                current_sub_type = NormativeUnitType.SUBITEM
                current_item_label = f"({sub_match.group(1)})"
                current_lines = [entry]
            elif current_sub_type is not None:
                current_lines.append(entry)

        flush_current()
        return results, order
