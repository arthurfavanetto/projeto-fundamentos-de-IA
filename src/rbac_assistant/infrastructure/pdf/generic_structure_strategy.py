"""Estratégia estrutural genérica (fallback) para IS, Resoluções e outros documentos oficiais."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

from rbac_assistant.domain.entities.normative_unit import (
    NormativeTable,
    NormativeUnitType,
    StructureStatus,
)
from rbac_assistant.domain.entities.regulatory_document import DocumentType
from rbac_assistant.domain.entities.regulatory_page import RegulatoryPage
from rbac_assistant.domain.ports.validation_port import ParsedUnitRecord
from rbac_assistant.infrastructure.pdf.pdfplumber_extractor import normalize_auxiliary_text

# Padrões comuns em Resoluções (Art. 1º), IS e publicações técnicas (1. / 1.1 / CAPÍTULO I)
_GENERIC_HEADING_PATTERNS: tuple[tuple[NormativeUnitType, re.Pattern[str]], ...] = (
    (NormativeUnitType.SECTION, re.compile(r"^(Art\.?\s*\d+[º°o]?)\s*[-–—.]?\s*(.*)$")),
    (
        NormativeUnitType.CHAPTER,
        re.compile(r"^(CAP[ÍI]TULO\s+[IVXLCDM0-9]+)\s*[-–—]?\s*(.*)$", re.IGNORECASE),
    ),
    (NormativeUnitType.SECTION, re.compile(r"^(\d+\.\d+(?:\.\d+)?)\s+(.+)$")),
)


class GenericStructureStrategy:
    """Fallback estrutural para qualquer documento regulatório fora do padrão ANAC RBAC."""

    def supports(self, *, authority: str, document_type: DocumentType) -> bool:
        del authority, document_type
        return True

    def parse(
        self,
        *,
        version_id: str,
        regulation_number: str,
        pages: Sequence[RegulatoryPage],
        tables: Sequence[NormativeTable] = (),
    ) -> list[ParsedUnitRecord]:
        del regulation_number, tables

        buffers: list[tuple[NormativeUnitType, str, str, list[tuple[int, str]]]] = []
        current_type: NormativeUnitType | None = None
        current_sec = ""
        current_title = ""
        current_lines: list[tuple[int, str]] = []

        ordered_pages = sorted(pages, key=lambda p: p.pdf_page_number)
        for page in ordered_pages:
            for raw_line in page.original_text.splitlines():
                stripped = raw_line.strip()
                if not stripped:
                    continue

                matched_heading = False
                for unit_type, pattern in _GENERIC_HEADING_PATTERNS:
                    m = pattern.match(stripped)
                    if m:
                        if current_type is not None and current_lines:
                            buffers.append(
                                (current_type, current_sec, current_title, list(current_lines))
                            )
                        current_type = unit_type
                        current_sec = m.group(1).strip()
                        current_title = m.group(2).strip()
                        current_lines = [(page.pdf_page_number, raw_line)]
                        matched_heading = True
                        break

                if not matched_heading and current_type is not None:
                    current_lines.append((page.pdf_page_number, raw_line))

        if current_type is not None and current_lines:
            buffers.append((current_type, current_sec, current_title, list(current_lines)))

        if not buffers:
            all_text = "\n".join(p.original_text for p in ordered_pages)
            all_pages = tuple(p.pdf_page_number for p in ordered_pages)
            if not all_pages:
                return []
            return [
                ParsedUnitRecord(
                    unit_id=f"{version_id}::pending_structure_0",
                    version_id=version_id,
                    parent_unit_id=None,
                    unit_type=NormativeUnitType.PREAMBLE,
                    section_number="UNSTRUCTURED",
                    subsection_number="",
                    paragraph_number="",
                    item_number="",
                    title="Estrutura não identificada automaticamente — pendente de validação",
                    subpart="",
                    original_text=all_text,
                    normalized_text=normalize_auxiliary_text(all_text),
                    start_page=all_pages[0],
                    end_page=all_pages[-1],
                    page_numbers=all_pages,
                    order_index=0,
                    structure_status=StructureStatus.PENDING_VALIDATION,
                    content_hash=hashlib.sha256(all_text.encode("utf-8")).hexdigest(),
                )
            ]

        records: list[ParsedUnitRecord] = []
        for idx, (u_type, sec_num, title, lines) in enumerate(buffers):
            orig = "\n".join(text for _, text in lines)
            norm = normalize_auxiliary_text(orig)
            p_nums = tuple(sorted({p for p, _ in lines}))
            records.append(
                ParsedUnitRecord(
                    unit_id=f"{version_id}::{sec_num}::{idx}",
                    version_id=version_id,
                    parent_unit_id=None,
                    unit_type=u_type,
                    section_number=sec_num,
                    subsection_number="",
                    paragraph_number="",
                    item_number="",
                    title=title,
                    subpart="",
                    original_text=orig,
                    normalized_text=norm,
                    start_page=p_nums[0],
                    end_page=p_nums[-1],
                    page_numbers=p_nums,
                    order_index=idx,
                    structure_status=StructureStatus.PARSED,
                    content_hash=hashlib.sha256(orig.encode("utf-8")).hexdigest(),
                )
            )
        return records
