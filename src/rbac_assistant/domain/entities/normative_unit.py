"""Entidades normativas hierárquicas da ANAC e governança de validação por página."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from rbac_assistant.domain.exceptions import UnvalidatedPageBlockedError

_PURE_NUMERIC_CELL_RE = re.compile(r"^[+\-±]?\s*\d+(?:[.,]\d+)*$")
_CID_TOKEN_RE = re.compile(r"\(cid:\d+\)", re.IGNORECASE)


class TableOwnershipStatus(StrEnum):
    """Estado explícito da associação proprietária de uma tabela a uma unidade normativa."""

    CONFIRMED = "confirmed"
    PAGE_GLOBAL = "page_global"
    AMBIGUOUS = "ambiguous"
    UNASSIGNED = "unassigned"


class ExtractionMethod(StrEnum):
    """Método utilizado para extrair o conteúdo textual de uma página do PDF."""

    NATIVE_PDF = "native_pdf"
    OCR_FALLBACK = "ocr_fallback"
    OCR_REQUIRED_UNAVAILABLE = "ocr_required_unavailable"


class ValidationStatus(StrEnum):
    """Estado de validação textual individual (compatível com Etapa 1)."""

    VALIDATED = "validated"
    BLOCKED_UNREVIEWED = "blocked_unreviewed"
    PENDING_REVIEW = "pending_review"
    FAILED = "failed"


class NormativeUnitType(StrEnum):
    """Níveis estruturais da hierarquia normativa oficial da ANAC e documentos regulatórios."""

    PREAMBLE = "preamble"
    CHAPTER = "chapter"
    SUBPART = "subpart"
    SECTION = "section"
    SUBSECTION = "subsection"
    PARAGRAPH = "paragraph"
    INCISO = "inciso"
    ALINEA = "alinea"
    SUBITEM = "subitem"
    APPENDIX = "appendix"
    TABLE = "table"
    NOTE = "note"


class StructureStatus(StrEnum):
    """Confiança estrutural na delimitação automática de uma unidade normativa."""

    PARSED = "parsed"
    PENDING_VALIDATION = "pending_validation"


@dataclass(frozen=True, slots=True)
class PageExtractionMetadata:
    """Metadados individuais por página garantindo a política `block_until_reviewed`."""

    pdf_page_number: int
    printed_page_number: str
    extraction_method: ExtractionMethod
    validation_status: ValidationStatus
    integrity_checks_passed: bool
    text_sha256: str = ""
    reviewed_by: str | None = None
    reviewed_at: datetime | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if self.pdf_page_number < 1:
            raise ValueError("pdf_page_number deve ser >= 1 (indexação 1-based).")
        if not self.printed_page_number.strip():
            raise ValueError("printed_page_number não pode ser vazio.")

        if self.validation_status == ValidationStatus.VALIDATED:
            if not self.integrity_checks_passed:
                raise ValueError(
                    "Página não pode ter status 'validated' sem aprovação nas verificações "
                    "de integridade textual."
                )
            if self.extraction_method == ExtractionMethod.OCR_FALLBACK and (
                not self.reviewed_by or self.reviewed_at is None
            ):
                raise ValueError(
                    "Política block_until_reviewed: página extraída via OCR exige revisão "
                    "manual (reviewed_by e reviewed_at) antes de receber status 'validated'."
                )

    @classmethod
    def create_ocr_page(
        cls,
        *,
        pdf_page_number: int,
        printed_page_number: str,
        integrity_checks_passed: bool = False,
        text_sha256: str = "",
        notes: str = "",
    ) -> PageExtractionMetadata:
        """Cria metadados de página OCR bloqueada por padrão até revisão manual."""
        return cls(
            pdf_page_number=pdf_page_number,
            printed_page_number=printed_page_number,
            extraction_method=ExtractionMethod.OCR_FALLBACK,
            validation_status=ValidationStatus.BLOCKED_UNREVIEWED,
            integrity_checks_passed=integrity_checks_passed,
            text_sha256=text_sha256,
            notes=notes,
        )

    @property
    def is_literal_transcription_allowed(self) -> bool:
        """Indica se a página está apta para transcrição normativa literal."""
        return (
            self.validation_status == ValidationStatus.VALIDATED
            and self.integrity_checks_passed
        )


@dataclass(frozen=True, slots=True)
class TableCellGeometry:
    """Geometria estrutural de uma célula individual de tabela/formulário extraído do PDF."""

    row_index: int
    col_index: int
    row_span: int = 1
    col_span: int = 1
    text: str = ""
    bbox: tuple[float, float, float, float] | None = None
    is_empty: bool = False

    def to_dict(self) -> dict[str, object]:
        """Serializa geometria da célula para persistência JSON no SQLite."""
        return {
            "row_index": self.row_index,
            "col_index": self.col_index,
            "row_span": self.row_span,
            "col_span": self.col_span,
            "text": self.text,
            "bbox": list(self.bbox) if self.bbox is not None else None,
            "is_empty": self.is_empty,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> TableCellGeometry:
        """Reconstrói `TableCellGeometry` a partir de dicionário JSON persistido."""
        raw_bbox = data.get("bbox")
        bbox: tuple[float, float, float, float] | None = None
        if isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) == 4:
            bbox = (
                float(raw_bbox[0]),
                float(raw_bbox[1]),
                float(raw_bbox[2]),
                float(raw_bbox[3]),
            )
        return cls(
            row_index=int(data.get("row_index", 0)),  # type: ignore[call-overload]
            col_index=int(data.get("col_index", 0)),  # type: ignore[call-overload]
            row_span=int(data.get("row_span", 1)),  # type: ignore[call-overload]
            col_span=int(data.get("col_span", 1)),  # type: ignore[call-overload]
            text=str(data.get("text", "")),
            bbox=bbox,
            is_empty=bool(data.get("is_empty", False)),
        )


@dataclass(frozen=True, slots=True)
class NormativeTable:
    """Tabela ou formulário normativo preservando células, geometria e associação proprietária."""

    table_id: str
    version_id: str
    pdf_page_number: int
    rows: tuple[tuple[str, ...], ...]
    unit_id: str | None = None
    caption: str = ""
    bbox: tuple[float, float, float, float] | None = None
    cells: tuple[TableCellGeometry, ...] = ()
    header_row_count: int = 1
    table_kind: str = "data_table"
    has_merged_cells: bool = False
    visual_layout_preserved: bool = True
    layout_fidelity_note: str = ""
    ownership_status: str = ""
    owner_section_number: str = ""
    ownership_note: str = ""

    def metadata_to_dict(self) -> dict[str, object]:
        """Serializa metadados estruturais, geométricos e de propriedade da tabela para JSON."""
        eff_ownership = effective_table_ownership_status(self).value
        return {
            "bbox": list(self.bbox) if self.bbox is not None else None,
            "cells": [cell.to_dict() for cell in self.cells],
            "header_row_count": self.header_row_count,
            "table_kind": self.table_kind,
            "has_merged_cells": self.has_merged_cells,
            "visual_layout_preserved": self.visual_layout_preserved,
            "layout_fidelity_note": self.layout_fidelity_note,
            "ownership_status": eff_ownership,
            "owner_section_number": self.owner_section_number,
            "ownership_note": self.ownership_note,
        }


def effective_table_ownership_status(table: NormativeTable) -> TableOwnershipStatus:
    """Determina o estado efetivo de propriedade da tabela (nunca infere `page_global`)."""
    raw = table.ownership_status.strip().lower()
    if raw:
        try:
            return TableOwnershipStatus(raw)
        except ValueError:
            return TableOwnershipStatus.AMBIGUOUS
    if table.unit_id and table.unit_id.strip():
        return TableOwnershipStatus.CONFIRMED
    return TableOwnershipStatus.UNASSIGNED


def validate_table_ownership(table: NormativeTable) -> tuple[str, ...]:
    """Valida exclusivamente a confiabilidade da associação proprietária (fail-closed)."""
    status = effective_table_ownership_status(table)
    has_unit_id = bool(table.unit_id and table.unit_id.strip())
    if status == TableOwnershipStatus.CONFIRMED:
        if not has_unit_id:
            return (
                f"Tabela '{table.table_id}' (p.{table.pdf_page_number}) marcada como "
                "'confirmed' sem unit_id proprietário válido.",
            )
        return ()
    if status == TableOwnershipStatus.PAGE_GLOBAL:
        if has_unit_id:
            return (
                f"Tabela '{table.table_id}' (p.{table.pdf_page_number}) marcada como "
                "'page_global', porém vinculada a unit_id específico.",
            )
        return ()
    if status == TableOwnershipStatus.AMBIGUOUS:
        detail = f" ({table.ownership_note})" if table.ownership_note else ""
        return (
            f"Tabela '{table.table_id}' (p.{table.pdf_page_number}) possui associação "
            f"proprietária ambígua{detail}.",
        )
    detail = f" ({table.ownership_note})" if table.ownership_note else ""
    return (
        f"Tabela '{table.table_id}' (p.{table.pdf_page_number}) sem associação "
        f"proprietária confiável (unassigned){detail}.",
    )


def resolve_section_tables(
    *,
    page_tables: Sequence[NormativeTable],
    unit_id: str | None,
    section_number: str,
) -> tuple[tuple[NormativeTable, ...], tuple[str, ...], tuple[str, ...]]:
    """Resolve tabelas pertencentes à seção separando validação de propriedade e integridade.

    Retorna `(tabelas_validas_da_secao, bloqueios_propriedade, bloqueios_integridade)`.
    Regras fail-closed:
    - Tabelas com `ownership_status='confirmed'` de OUTRA unidade na mesma página são ignoradas
      (jamais vazam para a seção errada).
    - Tabelas sem proprietário confiável (`unassigned` ou `ambiguous`, inclusive legado com
      `unit_id=None` sem marcação explícita `page_global`) bloqueiam a resposta da página.
    - Integridade estrutural (`validate_table_integrity`) é verificada separadamente e nunca
      substitui a validação de propriedade.
    """
    owned_candidates: list[NormativeTable] = []
    ownership_issues: list[str] = []
    target_uid = (unit_id or "").strip()
    target_sec = section_number.strip().lower()

    for tbl in page_tables:
        status = effective_table_ownership_status(tbl)
        tbl_uid = (tbl.unit_id or "").strip()
        tbl_sec = tbl.owner_section_number.strip().lower()

        if status == TableOwnershipStatus.CONFIRMED:
            if not tbl_uid:
                ownership_issues.append(
                    f"Tabela '{tbl.table_id}' (p.{tbl.pdf_page_number}) marcada como "
                    "'confirmed' sem unit_id proprietário."
                )
                continue
            if target_uid and tbl_uid != target_uid:
                # Pertence estritamente a outra unidade normativa na mesma página
                continue
            if not target_uid:
                if tbl_sec and tbl_sec != target_sec:
                    continue
                if not tbl_sec:
                    ownership_issues.append(
                        f"Tabela '{tbl.table_id}' (p.{tbl.pdf_page_number}) não pôde ter "
                        f"propriedade confirmada para a seção '{section_number}'."
                    )
                    continue
            if tbl_sec and tbl_sec != target_sec:
                ownership_issues.append(
                    f"Tabela '{tbl.table_id}' (p.{tbl.pdf_page_number}) possui divergência "
                    f"de proprietário (unit_id={tbl_uid!r}, owner_section="
                    f"{tbl.owner_section_number!r} vs seção consultada {section_number!r})."
                )
                continue
            owned_candidates.append(tbl)
        elif status == TableOwnershipStatus.PAGE_GLOBAL:
            if tbl_uid:
                ownership_issues.append(
                    f"Tabela '{tbl.table_id}' (p.{tbl.pdf_page_number}) marcada como "
                    "'page_global' com unit_id preenchido."
                )
                continue
            owned_candidates.append(tbl)
        else:
            # UNASSIGNED ou AMBIGUOUS: nunca associa automaticamente nem descarta em silêncio
            detail = f" — {tbl.ownership_note}" if tbl.ownership_note else ""
            ownership_issues.append(
                f"Tabela '{tbl.table_id}' (p.{tbl.pdf_page_number}) com propriedade "
                f"'{status.value}' bloqueia uso confiável na seção '{section_number}'{detail}."
            )

    integrity_issues: list[str] = []
    for tbl in owned_candidates:
        integrity_issues.extend(validate_table_integrity(tbl))

    if ownership_issues or integrity_issues:
        return ((), tuple(ownership_issues), tuple(integrity_issues))
    return (tuple(owned_candidates), (), ())


def validate_table_integrity(table: NormativeTable) -> tuple[str, ...]:
    """Valida integridade estrutural de tabela/formulário sob política fail-closed.

    Retorna tupla vazia se a estrutura for íntegra; caso contrário retorna diagnósticos
    objetivos que impedem apresentar valores tabulares sem cabeçalhos ou associação confiável.
    """
    issues: list[str] = []
    if not table.rows:
        return (f"Tabela '{table.table_id}' (p.{table.pdf_page_number}) sem linhas extraídas.",)

    col_counts = {len(row) for row in table.rows}
    if len(col_counts) > 1:
        issues.append(
            f"Tabela '{table.table_id}' (p.{table.pdf_page_number}) com número inconsistente "
            f"de colunas entre linhas: {sorted(col_counts)}."
        )
    num_cols = len(table.rows[0])
    if num_cols == 0:
        issues.append(
            f"Tabela '{table.table_id}' (p.{table.pdf_page_number}) sem colunas válidas."
        )

    for r_idx, row in enumerate(table.rows):
        for c_idx, cell in enumerate(row):
            if "\ufffd" in cell or _CID_TOKEN_RE.search(cell):
                issues.append(
                    f"Tabela '{table.table_id}' (p.{table.pdf_page_number}, "
                    f"L{r_idx + 1}C{c_idx + 1}) contém glifo corrompido ou CID não mapeado."
                )

    for geom in table.cells:
        if geom.row_span < 1 or geom.col_span < 1:
            issues.append(
                f"Tabela '{table.table_id}' (p.{table.pdf_page_number}) possui span inválido "
                f"em ({geom.row_index}, {geom.col_index})."
            )
        if geom.row_index < 0 or geom.row_index >= len(table.rows):
            issues.append(
                f"Tabela '{table.table_id}' (p.{table.pdf_page_number}) possui row_index fora "
                f"dos limites: {geom.row_index}."
            )
        if num_cols > 0 and (geom.col_index < 0 or geom.col_index >= num_cols):
            issues.append(
                f"Tabela '{table.table_id}' (p.{table.pdf_page_number}) possui col_index fora "
                f"dos limites: {geom.col_index}."
            )

    if table.table_kind == "data_table" and num_cols > 0:
        if table.header_row_count < 1 or table.header_row_count > len(table.rows):
            issues.append(
                f"Tabela '{table.table_id}' (p.{table.pdf_page_number}) sem cabeçalho estrutural "
                f"válido (header_row_count={table.header_row_count}, linhas={len(table.rows)})."
            )
        else:
            header_rows = table.rows[: table.header_row_count]
            non_empty_headers = [
                cell.strip() for row in header_rows for cell in row if cell.strip()
            ]
            if not non_empty_headers:
                issues.append(
                    f"Tabela '{table.table_id}' (p.{table.pdf_page_number}) possui linhas de "
                    "cabeçalho completamente vazias."
                )
            elif len(table.rows) >= 2:
                first_row_non_empty = [cell.strip() for cell in table.rows[0] if cell.strip()]
                if first_row_non_empty and all(
                    _PURE_NUMERIC_CELL_RE.match(cell) for cell in first_row_non_empty
                ):
                    issues.append(
                        f"Tabela '{table.table_id}' (p.{table.pdf_page_number}) contém apenas "
                        "valores numéricos na primeira linha sem cabeçalhos/unidades."
                    )

    return tuple(issues)


def _resolve_table_column_headers(table: NormativeTable) -> tuple[str, ...]:
    """Resolve rótulo completo de cada coluna combinando cabeçalhos mesclados e subcabeçalhos."""
    if not table.rows:
        return ()
    num_cols = len(table.rows[0])
    hdr_count = min(max(table.header_row_count, 0), len(table.rows))
    if hdr_count == 0:
        return tuple(f"Coluna {idx + 1}" for idx in range(num_cols))

    geom_by_row: dict[int, list[TableCellGeometry]] = {}
    for cell in table.cells:
        geom_by_row.setdefault(cell.row_index, []).append(cell)

    per_row_labels: list[list[str]] = []
    for r_idx in range(hdr_count):
        row_labels = [""] * num_cols
        row_geoms = geom_by_row.get(r_idx)
        if row_geoms:
            for cell in sorted(row_geoms, key=lambda c: c.col_index):
                clean = " ".join(part.strip() for part in cell.text.splitlines() if part.strip())
                end_col = min(num_cols, cell.col_index + max(1, cell.col_span))
                for c_idx in range(cell.col_index, end_col):
                    row_labels[c_idx] = clean
        else:
            last_non_empty = ""
            for c_idx, raw_cell in enumerate(table.rows[r_idx]):
                clean = " ".join(part.strip() for part in raw_cell.splitlines() if part.strip())
                if clean:
                    row_labels[c_idx] = clean
                    last_non_empty = clean
                elif hdr_count > 1 and r_idx == 0 and last_non_empty:
                    # Propaga cabeçalho mesclado da linha superior quando geometria não foi salva
                    row_labels[c_idx] = last_non_empty
        per_row_labels.append(row_labels)

    resolved: list[str] = []
    for c_idx in range(num_cols):
        parts: list[str] = []
        for r_idx in range(hdr_count):
            piece = per_row_labels[r_idx][c_idx]
            if piece and (not parts or parts[-1] != piece):
                parts.append(piece)
        resolved.append(" ".join(parts) if parts else f"Coluna {c_idx + 1}")
    return tuple(resolved)


def render_structured_table(table: NormativeTable) -> str:
    """Renderiza tabela ou formulário preservando cabeçalhos, unidades, sinais e células vazias."""
    if not table.rows:
        return ""

    lines: list[str] = []
    kind_label = "FORMULÁRIO ESTRUTURADO" if table.table_kind == "form" else "TABELA ESTRUTURADA"
    caption_suffix = f" — {table.caption}" if table.caption else ""
    lines.append(f"[{kind_label} — Página PDF {table.pdf_page_number}{caption_suffix}]")

    if table.table_kind == "form" or not table.visual_layout_preserved:
        note = table.layout_fidelity_note or (
            "Representação estruturada em texto plano; o layout gráfico original de caixas e "
            "campos de preenchimento do PDF oficial não é reproduzido em fac-símile."
        )
        lines.append(f"Aviso de fidelidade de layout: {note}")

        geom_by_row: dict[int, list[TableCellGeometry]] = {}
        for cell in table.cells:
            geom_by_row.setdefault(cell.row_index, []).append(cell)

        num_cols = len(table.rows[0])
        for r_idx, row in enumerate(table.rows):
            row_geoms = geom_by_row.get(r_idx)
            if row_geoms:
                ordered = sorted(row_geoms, key=lambda c: c.col_index)
                rendered_cells: list[str] = []
                for cell in ordered:
                    span_tag = f" [mesclado {cell.col_span} cols]" if cell.col_span > 1 else ""
                    val = cell.text if cell.text else "(vazio)"
                    rendered_cells.append(f"Campo {cell.col_index + 1}{span_tag}: {val}")
                lines.append(f"Linha {r_idx + 1}: " + " | ".join(rendered_cells))
            else:
                rendered_cells = [
                    f"Col {c_idx + 1}: {cell if cell else '(vazio)'}"
                    for c_idx, cell in enumerate(row[:num_cols])
                ]
                lines.append(f"Linha {r_idx + 1}: " + " | ".join(rendered_cells))
        return "\n".join(lines)

    headers = _resolve_table_column_headers(table)
    hdr_count = min(max(table.header_row_count, 0), len(table.rows))
    lines.append("Colunas: " + " | ".join(headers))

    for data_idx, row in enumerate(table.rows[hdr_count:], start=1):
        pairs = [
            f"{headers[c_idx]}={cell if cell else '(vazio)'}"
            for c_idx, cell in enumerate(row)
        ]
        lines.append(f"Linha {data_idx}: " + " | ".join(pairs))

    return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class NormativeReference:
    """Referência cruzada identificada no texto normativo (ex.: RBAC 145, seção 43.9)."""

    reference_id: str
    unit_id: str
    target_regulation: str
    target_section: str = ""
    raw_citation: str = ""


@dataclass(frozen=True, slots=True)
class NormativeUnit:
    """Unidade normativa estruturada de um regulamento (Seção, Parágrafo, Inciso, Apêndice)."""

    document_id: str
    regulation_number: str
    amendment_number: str
    unit_type: NormativeUnitType
    section_number: str
    title: str
    exact_text: str
    pages: tuple[PageExtractionMetadata, ...]
    subpart: str = ""
    subsection_number: str = ""
    paragraph_number: str = ""
    item_number: str = ""
    content_hash: str = ""
    unit_id: str = ""
    version_id: str = ""
    parent_unit_id: str | None = None
    normalized_text: str = ""
    start_page: int = 1
    end_page: int = 1
    order_index: int = 0
    structure_status: StructureStatus = StructureStatus.PARSED
    validation_status: str = "pending_review"

    def __post_init__(self) -> None:
        if not self.section_number.strip():
            raise ValueError("section_number não pode ser vazio.")
        if not self.pages:
            raise ValueError("NormativeUnit deve referenciar ao menos uma página de origem.")

    @property
    def unvalidated_pages(self) -> tuple[PageExtractionMetadata, ...]:
        """Retorna todas as páginas abrangidas pela seção que ainda não estão validadas."""
        return tuple(page for page in self.pages if not page.is_literal_transcription_allowed)

    @property
    def can_guarantee_literal_transcription(self) -> bool:
        """Retorna True somente se todas as páginas abrangidas estiverem validadas."""
        return (
            self.structure_status == StructureStatus.PARSED
            and len(self.unvalidated_pages) == 0
        )

    @property
    def literal_transcription_warning(self) -> str | None:
        """Gera aviso explícito quando a seção abranger páginas não validadas ou bloqueadas."""
        if self.structure_status != StructureStatus.PARSED:
            return (
                f"AVISO DE ESTRUTURA: A delimitação automática da seção {self.section_number} "
                "está marcada como pendente de validação. O sistema não garante a transcrição "
                "literal completa desta seção."
            )
        unvalidated = self.unvalidated_pages
        if not unvalidated:
            return None

        details = ", ".join(
            f"PDF p.{p.pdf_page_number} (impressa {p.printed_page_number}, "
            f"{p.extraction_method.value}/{p.validation_status.value})"
            for p in unvalidated
        )
        return (
            f"AVISO DE FIDELIDADE: A seção {self.section_number} abrange página(s) ainda "
            f"não validadas [{details}]. Sob a política block_until_reviewed, o sistema "
            "não pode garantir a transcrição literal completa desta seção até revisão manual."
        )

    def require_literal_transcription(self) -> str:
        """Retorna o texto literal exato ou bloqueia se houver páginas não validadas."""
        unvalidated = self.unvalidated_pages
        if unvalidated:
            blocked_pages = tuple(p.pdf_page_number for p in unvalidated)
            raise UnvalidatedPageBlockedError(self.section_number, blocked_pages)
        return self.exact_text


@dataclass(frozen=True, slots=True)
class NormativeSection:
    """Seção normativa reconstruída em ordem (incluindo seções que atravessam páginas)."""

    document_id: str
    version_id: str
    authority: str
    document_type: str
    regulation_number: str
    amendment_number: str
    version_status: str
    section_number: str
    title: str
    original_text: str
    normalized_text: str
    start_page: int
    end_page: int
    pdf_pages: tuple[int, ...]
    printed_pages: tuple[str, ...]
    validation_status: str
    structure_status: StructureStatus
    literal_transcription_allowed: bool
    pending_pdf_pages: tuple[int, ...] = ()
    ocr_blocked_pdf_pages: tuple[int, ...] = ()
    warning_message: str | None = None
    tables: tuple[NormativeTable, ...] = ()
    references: tuple[NormativeReference, ...] = ()
