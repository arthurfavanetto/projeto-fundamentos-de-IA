"""Extrator nativo de PDF página a página baseado em `pdfplumber` (licença MIT)."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import pdfplumber

from rbac_assistant.domain.exceptions import InvalidPdfError
from rbac_assistant.domain.ports.pdf_extractor_port import (
    ExtractedPageData,
    ExtractedTableData,
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


def _extract_tables(page: Any, pdf_page_number: int) -> tuple[ExtractedTableData, ...]:
    """Extrai tabelas estruturadas da página preservando células sem descartar texto."""
    try:
        raw_tables = page.extract_tables() or []
    except Exception:
        return ()

    parsed_tables: list[ExtractedTableData] = []
    for raw_table in raw_tables:
        if not raw_table:
            continue
        cleaned_rows: list[tuple[str, ...]] = []
        for row in raw_table:
            if not row:
                continue
            cells = tuple((cell or "").strip() for cell in row)
            if any(cells):
                cleaned_rows.append(cells)
        if cleaned_rows:
            parsed_tables.append(
                ExtractedTableData(
                    pdf_page_number=pdf_page_number,
                    rows=tuple(cleaned_rows),
                )
            )
    return tuple(parsed_tables)


class PdfPlumberExtractorAdapter:
    """Implementação de `PDFExtractorPort` usando `pdfplumber` (MIT)."""

    def extract_pages(
        self,
        pdf_path: Path,
        *,
        min_native_chars: int = 50,
        max_pages: int = 1000,
    ) -> list[ExtractedPageData]:
        if not pdf_path.is_file():
            raise InvalidPdfError(f"Arquivo PDF não encontrado: {pdf_path}")

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
                    original_text = page.extract_text(x_tolerance=2, y_tolerance=3) or ""
                    normalized_text = normalize_auxiliary_text(original_text)
                    lines = original_text.splitlines()
                    printed_page = detect_printed_page_number(lines)
                    content_hash = hashlib.sha256(original_text.encode("utf-8")).hexdigest()
                    tables = _extract_tables(page, idx)
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
                        )
                    )
                return extracted
        except InvalidPdfError:
            raise
        except Exception as exc:
            raise InvalidPdfError(
                f"Falha ao abrir ou extrair PDF '{pdf_path.name}': {exc}"
            ) from exc
