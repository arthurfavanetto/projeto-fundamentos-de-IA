"""Extrator nativo de PDF página a página baseado em `pdfplumber` (licença MIT).

Recursos de fidelidade literal:
- Calcula o SHA-256 do PDF original antes da extração.
- Detecta anomalias vetoriais/tipográficas (ex.: `+` sublinhado simulando `±`, `o`/`a`
  sobrescrito sublinhado simulando `º`/`ª`, tokens `(cid:...)`, PUA e ligaduras).
- Aplica exclusivamente correções verificadas vinculadas ao SHA-256 + página + contexto exato
  tanto no texto quanto nas células de tabelas/formulários antes de gerar `normalized_text` e
  o `content_hash` canônico.
- Captura geometria de tabelas/formulários (`find_tables()`), células mescladas (`col_span`/
  `row_span`), legendas (`TABELA I..IV`), caixas gráficas de formulários e notas de fidelidade.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from pathlib import Path
from statistics import median
from typing import Any

import pdfplumber

from rbac_assistant.domain.entities.normative_unit import TableCellGeometry
from rbac_assistant.domain.entities.regulatory_page import compute_page_content_hash
from rbac_assistant.domain.exceptions import InvalidPdfError
from rbac_assistant.domain.ports.pdf_extractor_port import (
    ExtractedPageData,
    ExtractedTableData,
    GlyphAnomaly,
)
from rbac_assistant.infrastructure.pdf.verified_corrections import (
    DEFAULT_VERIFIED_CORRECTION_RULES,
    VerifiedTextCorrectionRule,
    apply_verified_page_corrections,
)

# Padrões comuns de numeração impressa em documentos oficiais (ANAC e afins)
_PRINTED_PAGE_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Rodapé típico ANAC: "Origem: SAR 2/32" ou "Origem: SPO 14/50"
    re.compile(r"Origem\s*:\s*[A-Z0-9/\-]+\s+(\d+\s*/\s*\d+)\s*$", re.IGNORECASE),
    # "Página 5 de 28" / "Pág. 5/28" / "Page 5 of 28"
    re.compile(r"(?:P[áa]gina|P[áa]g\.?|Page)\s+(\d+(?:\s*(?:de|of|/)\s*\d+)?)\b", re.IGNORECASE),
    # Número isolado "X/Y" no início ou fim de linha de rodapé/cabeçalho
    re.compile(r"^\s*(\d+\s*/\s*\d+)\s*$"),
    # Número isolado em rodapé (ex.: "- 12 -" ou "12")
    re.compile(r"^\s*[-–—]?\s*(\d{1,4})\s*[-–—]?\s*$"),
)

_RUNNING_CHROME_RE = re.compile(
    r"^(?:Data da emiss[ãa]o\s*:|Data de vig[êe]ncia\s*:|Origem\s*:)",
    re.IGNORECASE,
)
_TABLE_CAPTION_RE = re.compile(r"^TABELA\s+[IVXLCDM0-9]+(?:\b.*)?$", re.IGNORECASE)
_FORM_CUE_RE = re.compile(
    r"(?:_{3,}|\bAssinatura\b|\bItem\s*:|\bEndere[çc]o\b|\bN[º°o]\s*Tel\b|"
    r"\bOrganiza[çc][ãa]o de Manuten[çc][ãa]o\b)",
    re.IGNORECASE,
)
_PURE_NUMERIC_RE = re.compile(r"^[+\-±]?\s*\d+(?:[.,]\d+)*$")
_CID_TOKEN_RE = re.compile(r"\(cid:\d+\)", re.IGNORECASE)
_PUA_CHAR_RE = re.compile(r"[\uE000-\uF8FF]")
_LIGATURE_CHAR_RE = re.compile(r"[\uFB00-\uFB06]")

_EXTRACTOR_VERSION = "pdfplumber_verified_v2"


def normalize_auxiliary_text(original_text: str) -> str:
    """Produz texto auxiliar normalizado para busca/estrutura sem alterar `original_text`."""
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in original_text.splitlines()]
    collapsed = "\n".join(line for line in lines if line)
    return re.sub(r"\n{3,}", "\n\n", collapsed).strip()


def detect_printed_page_number(lines: list[str]) -> str:
    """Identifica o número de página impresso em cabeçalho/rodapé sem presumir o índice do PDF."""
    if not lines:
        return ""
    candidates = [*lines[-3:], *lines[:3]]
    for line in candidates:
        stripped = line.strip()
        if not stripped:
            continue
        for pattern in _PRINTED_PAGE_PATTERNS:
            match = pattern.search(stripped)
            if match:
                return re.sub(r"\s+", "", match.group(1))
    return ""


def _compute_file_sha256(pdf_path: Path) -> str:
    digest = hashlib.sha256()
    with pdf_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_coordinates_summary(
    page: Any,
    *,
    max_entries: int = 12,
) -> tuple[dict[str, float | str], ...]:
    """Captura resumo leve de coordenadas (top/x0/text) das primeiras palavras para auditoria."""
    try:
        words = page.extract_words(keep_blank_chars=False, use_text_flow=True) or []
    except Exception:
        return ()
    summary: list[dict[str, float | str]] = []
    for word in words[:max_entries]:
        summary.append(
            {
                "text": str(word.get("text", "")),
                "x0": round(float(word.get("x0", 0.0)), 1),
                "top": round(float(word.get("top", 0.0)), 1),
            }
        )
    return tuple(summary)


def _extract_positioned_lines(page: Any) -> list[tuple[float, float, str]]:
    """Extrai linhas com faixa vertical `(top, bottom, text)` ordenadas de cima para baixo."""
    try:
        words = page.extract_words(keep_blank_chars=False, use_text_flow=True) or []
    except Exception:
        return []

    grouped: list[list[Any]] = []
    ordered_words = sorted(
        words,
        key=lambda w: (round(float(w.get("top", 0.0)), 1), float(w.get("x0", 0.0))),
    )
    for word in ordered_words:
        top = float(word.get("top", 0.0))
        if not grouped or abs(top - float(grouped[-1][0].get("top", 0.0))) > 3.0:
            grouped.append([word])
        else:
            grouped[-1].append(word)

    lines: list[tuple[float, float, str]] = []
    for line_words in grouped:
        line_words.sort(key=lambda w: float(w.get("x0", 0.0)))
        line_top = min(float(w.get("top", 0.0)) for w in line_words)
        line_bottom = max(float(w.get("bottom", 0.0)) for w in line_words)
        text = " ".join(str(w.get("text", "")) for w in line_words).strip()
        if text and not _RUNNING_CHROME_RE.match(text):
            lines.append((line_top, line_bottom, text))
    return lines


def _cluster_coords(values: Sequence[float], *, tol: float = 2.0) -> list[float]:
    if not values:
        return []
    ordered = sorted(values)
    clusters: list[list[float]] = [[ordered[0]]]
    for val in ordered[1:]:
        if val - clusters[-1][-1] <= tol:
            clusters[-1].append(val)
        else:
            clusters.append([val])
    return [sum(group) / len(group) for group in clusters]


def _detect_table_caption(
    positioned_lines: Sequence[tuple[float, float, str]],
    *,
    table_top: float,
    previous_table_bottom: float,
) -> str:
    """Identifica legenda estrutural (`TABELA I` + subtítulo) imediatamente acima da tabela."""
    above = [
        (top, bottom, text)
        for top, bottom, text in positioned_lines
        if previous_table_bottom - 1.0 <= top < table_top - 1.0 and (table_top - bottom) <= 58.0
    ]
    if not above:
        return ""
    for idx, (_top, _bottom, text) in enumerate(above):
        if _TABLE_CAPTION_RE.match(text):
            caption_parts = [text]
            if idx + 1 < len(above):
                next_top, _next_bot, next_text = above[idx + 1]
                if (next_top - _bottom) <= 22.0 and not _TABLE_CAPTION_RE.match(next_text):
                    caption_parts.append(next_text)
            return " — ".join(caption_parts)
    return ""


def _classify_table_structure(
    *,
    rows: tuple[tuple[str, ...], ...],
    cells: tuple[TableCellGeometry, ...],
    caption: str,
) -> tuple[str, int, bool, bool, str]:
    """Classifica tabela de dados vs formulário gráfico e define `header_row_count`."""
    if not rows:
        return ("data_table", 0, False, True, "")

    num_cols = len(rows[0])
    has_merged = any(c.col_span > 1 or c.row_span > 1 for c in cells)
    full_width_merged_rows = {
        c.row_index for c in cells if num_cols >= 2 and c.col_span >= num_cols
    }
    tall_fill_box = any(
        c.bbox is not None and (c.bbox[3] - c.bbox[1]) > 50.0 and len(c.text) < 160
        for c in cells
    )
    joined_all = "\n".join(cell for row in rows for cell in row if cell)
    has_form_cues = bool(_FORM_CUE_RE.search(joined_all))

    is_form = (
        not _TABLE_CAPTION_RE.match(caption)
        and (
            (len(full_width_merged_rows) >= 2 and (has_form_cues or tall_fill_box))
            or (tall_fill_box and has_form_cues)
        )
    )
    if is_form:
        note = (
            "Formulário com campos gráficos/mesclados extraído em estrutura tabular de texto "
            "plano; o layout visual original (caixas de preenchimento e espaço de assinatura) "
            "do PDF oficial não é reproduzido em fac-símile."
        )
        return ("form", 0, has_merged, False, note)

    # Tabela de dados: detecta cabeçalho simples (1 linha) ou composto (2 linhas com unidades)
    first_row_non_empty = [cell.strip() for cell in rows[0] if cell.strip()]
    if not first_row_non_empty or (
        len(rows) >= 2 and all(_PURE_NUMERIC_RE.match(cell) for cell in first_row_non_empty)
    ):
        return ("data_table", 0, has_merged, True, "")

    row0_has_merged = any(c.row_index == 0 and c.col_span > 1 for c in cells)
    if len(rows) >= 2 and row0_has_merged:
        row1_non_empty = [cell.strip() for cell in rows[1] if cell.strip()]
        row1_is_subheader = bool(row1_non_empty) and not any(
            _PURE_NUMERIC_RE.match(cell) for cell in row1_non_empty
        )
        if row1_is_subheader:
            return ("data_table", 2, has_merged, True, "")

    return ("data_table", 1, has_merged, True, "")


def _extract_tables(page: Any, pdf_page_number: int) -> tuple[ExtractedTableData, ...]:
    """Extrai tabelas e formulários preservando geometria, spans mesclados e legendas."""
    positioned_lines = _extract_positioned_lines(page)

    found_tables: list[Any] = []
    if hasattr(page, "find_tables"):
        try:
            found_tables = list(page.find_tables() or [])
        except Exception:
            found_tables = []

    if found_tables:
        found_tables.sort(key=lambda t: (float(t.bbox[1]), float(t.bbox[0])))
        parsed_tables: list[ExtractedTableData] = []
        prev_bottom = 0.0

        for tbl in found_tables:
            raw_rows = tbl.extract() or []
            if not raw_rows:
                continue

            tbl_bbox = (
                round(float(tbl.bbox[0]), 2),
                round(float(tbl.bbox[1]), 2),
                round(float(tbl.bbox[2]), 2),
                round(float(tbl.bbox[3]), 2),
            )
            caption = _detect_table_caption(
                positioned_lines,
                table_top=tbl_bbox[1],
                previous_table_bottom=prev_bottom,
            )
            prev_bottom = tbl_bbox[3]

            # Constrói malha geométrica (x_edges e y_edges) para calcular col_span e row_span exatos
            x_coords: list[float] = []
            y_coords: list[float] = []
            tbl_rows_objs = list(getattr(tbl, "rows", []) or [])
            for row_obj in tbl_rows_objs:
                for c_bbox in getattr(row_obj, "cells", []) or []:
                    if c_bbox is not None:
                        x_coords.extend((float(c_bbox[0]), float(c_bbox[2])))
                        y_coords.extend((float(c_bbox[1]), float(c_bbox[3])))
            x_edges = _cluster_coords(x_coords, tol=2.0)
            y_edges = _cluster_coords(y_coords, tol=2.0)
            col_bounds = [
                (x_edges[i], x_edges[i + 1]) for i in range(max(0, len(x_edges) - 1))
            ]
            row_bounds = [
                (y_edges[i], y_edges[i + 1]) for i in range(max(0, len(y_edges) - 1))
            ]

            cleaned_rows: list[tuple[str, ...]] = []
            kept_row_indices: list[int] = []
            for raw_r_idx, row in enumerate(raw_rows):
                if not row:
                    continue
                cells_tuple = tuple((cell or "").strip() for cell in row)
                if any(cells_tuple):
                    cleaned_rows.append(cells_tuple)
                    kept_row_indices.append(raw_r_idx)

            if not cleaned_rows:
                continue

            cell_geometries: list[TableCellGeometry] = []
            for out_r_idx, raw_r_idx in enumerate(kept_row_indices):
                row_texts = cleaned_rows[out_r_idx]
                row_cells_bboxes = (
                    list(tbl_rows_objs[raw_r_idx].cells)
                    if raw_r_idx < len(tbl_rows_objs)
                    else []
                )
                for c_idx, cell_text in enumerate(row_texts):
                    c_bbox = row_cells_bboxes[c_idx] if c_idx < len(row_cells_bboxes) else None
                    if row_cells_bboxes and c_bbox is None:
                        # Slot coberto por célula mesclada à esquerda ou acima
                        continue
                    col_span = 1
                    row_span = 1
                    norm_bbox: tuple[float, float, float, float] | None = None
                    if c_bbox is not None:
                        cx0, ctop, cx1, cbot = (
                            float(c_bbox[0]),
                            float(c_bbox[1]),
                            float(c_bbox[2]),
                            float(c_bbox[3]),
                        )
                        norm_bbox = (
                            round(cx0, 2),
                            round(ctop, 2),
                            round(cx1, 2),
                            round(cbot, 2),
                        )
                        if col_bounds:
                            covered_cols = sum(
                                1
                                for bx0, bx1 in col_bounds
                                if bx0 >= cx0 - 2.5 and bx1 <= cx1 + 2.5
                            )
                            col_span = max(1, covered_cols)
                        if row_bounds:
                            covered_rows = sum(
                                1
                                for by0, by1 in row_bounds
                                if by0 >= ctop - 2.5 and by1 <= cbot + 2.5
                            )
                            row_span = max(1, covered_rows)
                    cell_geometries.append(
                        TableCellGeometry(
                            row_index=out_r_idx,
                            col_index=c_idx,
                            row_span=row_span,
                            col_span=col_span,
                            text=cell_text,
                            bbox=norm_bbox,
                            is_empty=(cell_text == ""),
                        )
                    )

            rows_tuple = tuple(cleaned_rows)
            geoms_tuple = tuple(cell_geometries)
            table_kind, header_row_count, has_merged, visual_ok, note = _classify_table_structure(
                rows=rows_tuple,
                cells=geoms_tuple,
                caption=caption,
            )
            parsed_tables.append(
                ExtractedTableData(
                    pdf_page_number=pdf_page_number,
                    rows=rows_tuple,
                    caption=caption,
                    bbox=tbl_bbox,
                    cells=geoms_tuple,
                    header_row_count=header_row_count,
                    table_kind=table_kind,
                    has_merged_cells=has_merged,
                    visual_layout_preserved=visual_ok,
                    layout_fidelity_note=note,
                )
            )
        return tuple(parsed_tables)

    # Fallback para objetos simulados sem `find_tables()`
    try:
        raw_tables = page.extract_tables() or []
    except Exception:
        return ()

    fallback_tables: list[ExtractedTableData] = []
    for raw_table in raw_tables:
        if not raw_table:
            continue
        cleaned_rows = []
        for row in raw_table:
            if not row:
                continue
            cells_tuple = tuple((cell or "").strip() for cell in row)
            if any(cells_tuple):
                cleaned_rows.append(cells_tuple)
        if cleaned_rows:
            rows_tuple = tuple(cleaned_rows)
            table_kind, header_row_count, has_merged, visual_ok, note = _classify_table_structure(
                rows=rows_tuple,
                cells=(),
                caption="",
            )
            fallback_tables.append(
                ExtractedTableData(
                    pdf_page_number=pdf_page_number,
                    rows=rows_tuple,
                    header_row_count=header_row_count,
                    table_kind=table_kind,
                    has_merged_cells=has_merged,
                    visual_layout_preserved=visual_ok,
                    layout_fidelity_note=note,
                )
            )
    return tuple(fallback_tables)


def _detect_page_glyph_anomalies(
    page: Any,
    *,
    pdf_page_number: int,
    original_text: str,
) -> list[GlyphAnomaly]:
    """Detecta anomalias de glifos/vetores (sublinhado sob `+` ou `o`/`a` sobrescrito, CID, PUA)."""
    anomalies: list[GlyphAnomaly] = []

    # 1. Anomalias textuais diretas: CID não mapeado, PUA ou ligaduras tipográficas
    for cid_match in _CID_TOKEN_RE.finditer(original_text):
        start = max(0, cid_match.start() - 25)
        end = min(len(original_text), cid_match.end() + 25)
        anomalies.append(
            GlyphAnomaly(
                kind="cid_unmapped",
                pdf_page_number=pdf_page_number,
                text=cid_match.group(0),
                context=original_text[start:end].replace("\n", " "),
            )
        )
    for pua_match in _PUA_CHAR_RE.finditer(original_text):
        start = max(0, pua_match.start() - 25)
        end = min(len(original_text), pua_match.end() + 25)
        anomalies.append(
            GlyphAnomaly(
                kind="private_use_area",
                pdf_page_number=pdf_page_number,
                text=f"U+{ord(pua_match.group(0)):04X}",
                context=original_text[start:end].replace("\n", " "),
            )
        )
    for lig_match in _LIGATURE_CHAR_RE.finditer(original_text):
        start = max(0, lig_match.start() - 25)
        end = min(len(original_text), lig_match.end() + 25)
        anomalies.append(
            GlyphAnomaly(
                kind="ligature_glyph",
                pdf_page_number=pdf_page_number,
                text=lig_match.group(0),
                context=original_text[start:end].replace("\n", " "),
            )
        )

    # 2. Anomalias vetoriais: traços horizontais curtos sublinhando '+' ou 'o'/'a' sobrescrito
    chars = list(getattr(page, "chars", []) or [])
    if not chars:
        return anomalies

    char_sizes = [float(c.get("size", 0.0)) for c in chars if float(c.get("size", 0.0)) > 0.0]
    median_size = float(median(char_sizes)) if char_sizes else 12.0

    segments: list[tuple[float, float, float, float]] = []
    for source_attr in ("rects", "lines", "curves"):
        for item in getattr(page, source_attr, []) or []:
            x0 = float(item.get("x0", 0.0))
            x1 = float(item.get("x1", 0.0))
            top = float(item.get("top", 0.0))
            bottom = float(item.get("bottom", 0.0))
            width = x1 - x0
            height = abs(bottom - top)
            if 2.0 <= width <= 15.0 and height <= 2.0:
                segments.append((x0, x1, top, bottom))

    seen_keys: set[tuple[str, int, int]] = set()
    for sx0, sx1, stop, _sbot in segments:
        covered = [
            c
            for c in chars
            if (sx0 - 0.5)
            <= ((float(c.get("x0", 0.0)) + float(c.get("x1", 0.0))) / 2.0)
            <= (sx1 + 0.5)
            and -4.0 <= (stop - float(c.get("bottom", 0.0))) <= 4.0
            and stop > float(c.get("top", 0.0))
            and str(c.get("text", "")).strip() != ""
        ]
        if not covered:
            continue
        covered.sort(key=lambda c: float(c.get("x0", 0.0)))
        covered_text = "".join(str(c.get("text", "")) for c in covered)

        kind: str | None = None
        if covered_text == "+":
            kind = "underlined_plus_operator"
        elif covered_text in ("o", "a", "O", "A"):
            avg_size = sum(float(c.get("size", 0.0)) for c in covered) / len(covered)
            if avg_size < median_size * 0.85:
                kind = "underlined_superscript_ordinal"

        if kind is None:
            continue

        dedup_key = (kind, round(sx0), round(stop))
        if dedup_key in seen_keys:
            continue
        seen_keys.add(dedup_key)

        anchor_top = float(covered[0].get("top", 0.0))
        same_line_chars = sorted(
            (c for c in chars if abs(float(c.get("top", 0.0)) - anchor_top) <= 4.0),
            key=lambda c: float(c.get("x0", 0.0)),
        )
        line_context = "".join(str(c.get("text", "")) for c in same_line_chars).strip()
        anomalies.append(
            GlyphAnomaly(
                kind=kind,
                pdf_page_number=pdf_page_number,
                text=covered_text,
                context=line_context,
            )
        )

    return anomalies


class PdfPlumberExtractorAdapter:
    """Implementação de `PDFExtractorPort` usando `pdfplumber` (MIT)."""

    def __init__(
        self,
        *,
        verified_correction_rules: Sequence[VerifiedTextCorrectionRule] = (
            DEFAULT_VERIFIED_CORRECTION_RULES
        ),
    ) -> None:
        self._verified_correction_rules = tuple(verified_correction_rules)

    def extract_pages(
        self,
        pdf_path: Path,
        *,
        min_native_chars: int = 50,
        max_pages: int = 1000,
    ) -> list[ExtractedPageData]:
        if not pdf_path.is_file():
            raise InvalidPdfError(f"Arquivo PDF não encontrado: {pdf_path}")

        source_sha256 = _compute_file_sha256(pdf_path)

        try:
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                if total_pages == 0:
                    raise InvalidPdfError(f"PDF não contém páginas: {pdf_path}")
                if total_pages > max_pages:
                    raise InvalidPdfError(
                        f"PDF possui {total_pages} páginas, excedendo o limite de {max_pages}."
                    )

                extracted: list[ExtractedPageData] = []
                for idx, page in enumerate(pdf.pages, start=1):
                    raw_text = page.extract_text(x_tolerance=2, y_tolerance=3) or ""
                    raw_tables = _extract_tables(page, idx)
                    raw_anomalies = _detect_page_glyph_anomalies(
                        page,
                        pdf_page_number=idx,
                        original_text=raw_text,
                    )

                    # Aplica somente correções verificadas vinculadas ao SHA-256 exato
                    corr_outcome = apply_verified_page_corrections(
                        source_sha256=source_sha256,
                        pdf_page_number=idx,
                        original_text=raw_text,
                        tables=raw_tables,
                        rules=self._verified_correction_rules,
                        extractor_version=_EXTRACTOR_VERSION,
                    )
                    original_text = corr_outcome.original_text
                    tables = corr_outcome.tables

                    # Marca anomalias de glifo efetivamente resolvidas pelas correções verificadas
                    resolved_anomalies: list[GlyphAnomaly] = []
                    for anomaly in raw_anomalies:
                        resolved_id: str | None = None
                        for rec in corr_outcome.applied_corrections:
                            if rec.original_context in anomaly.context and (
                                rec.corrected_context in original_text
                            ):
                                resolved_id = rec.correction_id
                                break
                        resolved_anomalies.append(
                            GlyphAnomaly(
                                kind=anomaly.kind,
                                pdf_page_number=anomaly.pdf_page_number,
                                text=anomaly.text,
                                context=anomaly.context,
                                resolved_by_correction_id=resolved_id,
                            )
                        )

                    normalized_text = normalize_auxiliary_text(original_text)
                    lines = original_text.splitlines()
                    printed_page = detect_printed_page_number(lines)
                    content_hash = compute_page_content_hash(original_text, tables)
                    image_count = len(getattr(page, "images", []) or [])
                    coords = _extract_coordinates_summary(page)
                    needs_ocr = len(normalized_text) < min_native_chars

                    extracted.append(
                        ExtractedPageData(
                            pdf_page_number=idx,
                            printed_page_number=printed_page,
                            original_text=original_text,
                            normalized_text=normalized_text,
                            content_hash=content_hash,
                            page_width=float(page.width or 0.0),
                            page_height=float(page.height or 0.0),
                            needs_ocr=needs_ocr,
                            tables=tables,
                            image_count=image_count,
                            coordinates_summary=coords,
                            source_sha256=source_sha256,
                            applied_corrections=tuple(
                                rec.to_dict() for rec in corr_outcome.applied_corrections
                            ),
                            correction_issues=corr_outcome.correction_issues,
                            glyph_anomalies=tuple(resolved_anomalies),
                        )
                    )
                return extracted
        except InvalidPdfError:
            raise
        except Exception as exc:
            raise InvalidPdfError(
                f"Falha ao abrir ou extrair PDF '{pdf_path.name}': {exc}"
            ) from exc
