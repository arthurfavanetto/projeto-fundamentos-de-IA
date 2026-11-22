"""Chunker estrutural hierárquico para unidades normativas ANAC e documentos regulatórios.

Regras:
- Preserva integralmente o `original_text` (SSOT) sem reescrita ou remoção de acentos.
- Gera `chunk_id` determinístico para garantir upsert idempotente e ausência de duplicação.
- Prioriza unidades-raiz (seções, apêndices, capítulos, preâmbulos) quando completas dentro da
  janela configurada; divide em sub-blocos preservando limites de parágrafo/linha quando exceder.
- Constrói `search_text` enriquecido com cabeçalho normativo para busca lexical FTS5 e vetorial.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

from rbac_assistant.domain.entities.document_version import DocumentVersion
from rbac_assistant.domain.entities.normative_unit import NormativeUnitType
from rbac_assistant.domain.entities.rag import NormativeChunk
from rbac_assistant.domain.entities.regulatory_document import RegulatoryDocument
from rbac_assistant.domain.entities.regulatory_page import PageValidationStatus, RegulatoryPage
from rbac_assistant.domain.ports.validation_port import ParsedUnitRecord

_SLUG_RE = re.compile(r"[^a-zA-Z0-9._-]+")


def _slugify(value: str) -> str:
    cleaned = _SLUG_RE.sub("_", value.strip())
    return cleaned.strip("_") or "unit"


def _aggregate_page_statuses(pages: Sequence[RegulatoryPage]) -> PageValidationStatus:
    statuses = [p.validation_status for p in pages]
    if not statuses:
        return PageValidationStatus.PENDING_REVIEW
    if any(s == PageValidationStatus.REJECTED for s in statuses):
        return PageValidationStatus.REJECTED
    if any(
        s in (PageValidationStatus.OCR_EXTRACTED, PageValidationStatus.PENDING_MANUAL_REVIEW)
        for s in statuses
    ):
        return PageValidationStatus.PENDING_MANUAL_REVIEW
    if all(s == PageValidationStatus.APPROVED_FOR_LITERAL_TRANSCRIPTION for s in statuses):
        return PageValidationStatus.APPROVED_FOR_LITERAL_TRANSCRIPTION
    if all(
        s
        in (
            PageValidationStatus.NATIVE_QUALITY_CHECKED,
            PageValidationStatus.MANUALLY_APPROVED,
            PageValidationStatus.APPROVED_FOR_LITERAL_TRANSCRIPTION,
        )
        for s in statuses
    ):
        return PageValidationStatus.NATIVE_QUALITY_CHECKED
    return PageValidationStatus.PENDING_REVIEW


class StructuralNormativeChunker:
    """Converte unidades normativas estruturadas em chunks determinísticos rastreáveis."""

    def __init__(
        self,
        *,
        max_chunk_tokens: int = 1200,
        chars_per_token: int = 4,
    ) -> None:
        if max_chunk_tokens < 50:
            raise ValueError("max_chunk_tokens deve ser >= 50.")
        self._max_chunk_chars = max_chunk_tokens * chars_per_token

    def build_chunks(
        self,
        *,
        document: RegulatoryDocument,
        version: DocumentVersion,
        units: Sequence[ParsedUnitRecord],
        pages: Sequence[RegulatoryPage],
    ) -> list[NormativeChunk]:
        """Gera lista ordenada e determinística de `NormativeChunk` para a versão informada."""
        pages_by_number = {p.pdf_page_number: p for p in pages}
        selected_units = self._select_indexable_units(units)
        chunks: list[NormativeChunk] = []

        for unit in selected_units:
            unit_pages = [
                pages_by_number[p_num]
                for p_num in unit.page_numbers
                if p_num in pages_by_number
            ]
            pdf_pages = tuple(sorted({p.pdf_page_number for p in unit_pages} or unit.page_numbers))
            printed_pages = tuple(
                pages_by_number[p_num].printed_page_number or f"PDF {p_num}"
                for p_num in pdf_pages
                if p_num in pages_by_number
            ) or tuple(f"PDF {p_num}" for p_num in pdf_pages)
            agg_status = _aggregate_page_statuses(unit_pages)

            text_slices = self._split_preserving_boundaries(unit.original_text)
            for chunk_idx, original_slice in enumerate(text_slices):
                chunk_id = self._deterministic_chunk_id(
                    document_id=document.document_id,
                    version_id=version.version_id,
                    section_number=unit.section_number,
                    order_index=unit.order_index,
                    chunk_index=chunk_idx,
                )
                search_text = self._build_search_text(
                    document=document,
                    version=version,
                    unit=unit,
                    slice_text=original_slice,
                )
                content_hash = hashlib.sha256(original_slice.encode("utf-8")).hexdigest()
                chunks.append(
                    NormativeChunk(
                        chunk_id=chunk_id,
                        document_id=document.document_id,
                        version_id=version.version_id,
                        unit_id=unit.unit_id,
                        authority=document.authority,
                        document_type=document.document_type.value,
                        regulation_number=document.regulation_number,
                        amendment_number=version.amendment_number,
                        version_status=version.version_status.value,
                        section_number=unit.section_number,
                        subsection_number=unit.subsection_number,
                        paragraph_number=unit.paragraph_number,
                        item_number=unit.item_number,
                        title=unit.title,
                        subpart=unit.subpart,
                        start_page=min(pdf_pages),
                        end_page=max(pdf_pages),
                        pdf_pages=pdf_pages,
                        printed_pages=printed_pages,
                        chunk_index=chunk_idx,
                        original_text=original_slice,
                        search_text=search_text,
                        source_hash=version.source_hash,
                        content_hash=content_hash,
                        validation_status=agg_status.value,
                    )
                )

        return chunks

    @staticmethod
    def _select_indexable_units(units: Sequence[ParsedUnitRecord]) -> list[ParsedUnitRecord]:
        """Seleciona unidades raiz (sem duplicar filhos redundantes) preservando a hierarquia."""
        root_types = {
            NormativeUnitType.SECTION,
            NormativeUnitType.APPENDIX,
            NormativeUnitType.CHAPTER,
            NormativeUnitType.SUBPART,
            NormativeUnitType.PREAMBLE,
        }
        roots = [
            u
            for u in units
            if u.parent_unit_id is None and u.unit_type in root_types and u.original_text.strip()
        ]
        if roots:
            return sorted(roots, key=lambda u: (u.order_index, u.section_number, u.unit_id))
        return sorted(
            [u for u in units if u.original_text.strip()],
            key=lambda u: (u.order_index, u.section_number, u.unit_id),
        )

    def _split_preserving_boundaries(self, text: str) -> list[str]:
        """Mantém a unidade inteira se couber na janela; caso contrário, divide em quebras."""
        cleaned = text.strip()
        if not cleaned:
            return []
        if len(cleaned) <= self._max_chunk_chars:
            return [text]

        lines = text.splitlines(keepends=True)
        slices: list[str] = []
        current_parts: list[str] = []
        current_len = 0

        for line in lines:
            if len(line) > self._max_chunk_chars:
                if current_parts:
                    slices.append("".join(current_parts))
                    current_parts = []
                    current_len = 0
                for start in range(0, len(line), self._max_chunk_chars):
                    slices.append(line[start : start + self._max_chunk_chars])
                continue

            if current_parts and current_len + len(line) > self._max_chunk_chars:
                slices.append("".join(current_parts))
                current_parts = [line]
                current_len = len(line)
            else:
                current_parts.append(line)
                current_len += len(line)

        if current_parts:
            slices.append("".join(current_parts))
        return slices

    @staticmethod
    def _deterministic_chunk_id(
        *,
        document_id: str,
        version_id: str,
        section_number: str,
        order_index: int,
        chunk_index: int,
    ) -> str:
        doc_slug = _slugify(document_id)
        ver_slug = _slugify(version_id)
        sec_slug = _slugify(section_number)
        return f"{doc_slug}__{ver_slug}__{sec_slug}__{order_index:04d}__c{chunk_index:03d}"

    @staticmethod
    def _build_search_text(
        *,
        document: RegulatoryDocument,
        version: DocumentVersion,
        unit: ParsedUnitRecord,
        slice_text: str,
    ) -> str:
        header_parts = [
            f"{document.authority} {document.document_type.value} {document.regulation_number}",
            f"Emenda {version.amendment_number}" if version.amendment_number else "",
            f"Seção {unit.section_number}" if unit.section_number else "",
            unit.title,
        ]
        header = " | ".join(part for part in header_parts if part)
        normalized_body = re.sub(r"[ \t]+", " ", slice_text).strip()
        return f"{header}\n{normalized_body}".strip()
