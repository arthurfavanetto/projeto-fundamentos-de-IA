"""Analisador automatizado de qualidade e integridade textual por página.

IMPORTANTE: Este verificador produz a classificação automatizada
`native_quality_checked` — jamais deve ser confundido com revisão humana.
Sob política fail-closed, qualquer anomalia de codificação, glifo não resolvido,
parênteses desequilibrados, falha de correção verificada ou falha estrutural/de
propriedade tabular bloqueia `passed_automated_checks=True` e mantém a página
em `pending_review`.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from rbac_assistant.domain.entities.normative_unit import (
    NormativeTable,
    validate_table_integrity,
)
from rbac_assistant.domain.entities.regulatory_page import QualityMetrics

_PT_ACCENTS_RE = re.compile(r"[áàâãéêíóôõúçÁÀÂÃÉÊÍÓÔÕÚÇ]")
_MOJIBAKE_RE = re.compile(r"(?:Ã[§£¡©ª­³µº]|â€[œ “”–—])")
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_CID_TOKEN_RE = re.compile(r"\(cid:\d+\)", re.IGNORECASE)
_PUA_CHARS_RE = re.compile(r"[\uE000-\uF8FF]")
# Detecta tolerâncias aeronáuticas onde '±' foi extraído como '+' entre dígitos e unidade
# (ex.: '1.090+3 MHz', '235+5 interrogações', '-73+4 dBm'), sem afetar valores positivos
# legítimos com sinal prefixado ('+70', '+21,3', '+531', '+893', '+974').
_SUSPICIOUS_TOLERANCE_PLUS_RE = re.compile(
    r"\d\+\d+(?:[.,]\d+)?\s*(?:MHz|kHz|GHz|Hz|dBm|dBw|dB|"
    r"interroga[çc](?:[ãa]o|[õo]es)|p[ée]s|hPa|mm|cm|ms|µs|us)\b",
    re.IGNORECASE,
)
# Detecta ordinal corrompido ('Lei no 7565', 'RBAC no 43', 'No Tel') sem confundir com a
# preposição legítima 'no' ('no art. 120', 'no parágrafo (a)').
_SUSPICIOUS_ORDINAL_RE = re.compile(
    r"(?:\b(?:Lei|Decreto|Resolu[çc][ãa]o|Portaria|RBAC|RBHA|IS|Emenda|Art\.?)\s+n[oO]\s+\d+\b"
    r"|\bNo\s+Tel\b)"
)


def analyze_parentheses_balance(
    text: str,
    *,
    expected_incoming_open: int = 0,
    allow_outgoing_open: int = 0,
) -> tuple[bool, int, str | None]:
    """Audita equilíbrio de parênteses via pilha determinística (inclusive entre páginas).

    Retorna `(unbalanced, end_depth, issue_message)`.
    Regras fail-closed:
    - Qualquer `)` que leve a pilha abaixo de 0 (mesmo considerando `expected_incoming_open`)
      é sempre considerado desequilíbrio.
    - Qualquer `(` remanescente ao final (`end_depth > 0`) é considerado desequilíbrio, salvo
      quando a continuidade entre páginas foi verificada globalmente no documento e
      `end_depth == allow_outgoing_open`.
    """
    if expected_incoming_open < 0 or allow_outgoing_open < 0:
        return (
            True,
            expected_incoming_open,
            "Contexto inválido de continuidade de parênteses entre páginas.",
        )

    depth = expected_incoming_open
    min_depth = depth
    open_count = 0
    close_count = 0
    for ch in text:
        if ch == "(":
            depth += 1
            open_count += 1
        elif ch == ")":
            depth -= 1
            close_count += 1
            if depth < min_depth:
                min_depth = depth

    if min_depth < 0:
        return (
            True,
            depth,
            "Parêntese de fechamento ')' sem abertura correspondente "
            f"({open_count} '(' vs {close_count} ')', saldo mínimo {min_depth}).",
        )
    if depth == 0:
        return (False, 0, None)
    if allow_outgoing_open > 0 and depth == allow_outgoing_open:
        return (False, depth, None)
    return (
        True,
        depth,
        "Parêntese aberto '(' sem fechamento correspondente "
        f"({open_count} '(' vs {close_count} ')', saldo final +{depth}).",
    )


def compute_cross_page_parentheses_context(
    page_texts: Sequence[str],
) -> list[tuple[int, int]]:
    """Calcula contexto `(incoming_open, outgoing_open)` entre páginas somente se equilibrado.

    Se o documento completo apresentar qualquer `)` sem abertura ou saldo final `!= 0`, retorna
    `[(0, 0), ...]` para todas as páginas (fail-closed: ambiguidade não resolvida bloqueia).
    """
    if not page_texts:
        return []

    depth = 0
    per_page: list[tuple[int, int]] = []
    for text in page_texts:
        incoming = depth
        for ch in text:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth < 0:
                    return [(0, 0) for _ in page_texts]
        per_page.append((incoming, depth))

    if depth != 0:
        return [(0, 0) for _ in page_texts]
    return per_page


class TextQualityChecker:
    """Implementação de `TextQualityCheckerPort` para páginas extraídas nativamente."""

    def evaluate(
        self,
        *,
        original_text: str,
        normalized_text: str,
        min_chars: int = 50,
        unresolved_glyph_anomalies: Sequence[str] = (),
        correction_issues: Sequence[str] = (),
        tables: Sequence[NormativeTable] = (),
        table_ownership_issues: Sequence[str] = (),
        expected_incoming_open_parens: int = 0,
        allow_outgoing_open_parens: int = 0,
    ) -> QualityMetrics:
        issues: list[str] = []

        char_count = len(normalized_text)
        lines = [line for line in original_text.splitlines() if line.strip()]
        line_count = len(lines)
        word_count = len(normalized_text.split())

        is_empty_or_incomplete = char_count < min_chars or word_count < 5
        if is_empty_or_incomplete:
            issues.append(
                f"Conteúdo vazio ou incompleto ({char_count} caracteres, {word_count} palavras)."
            )

        has_replacement_chars = "\ufffd" in original_text
        if has_replacement_chars:
            issues.append("Caracteres de substituição UTF-8 ('\\ufffd') detectados.")

        has_mojibake_signals = bool(_MOJIBAKE_RE.search(original_text))
        if has_mojibake_signals:
            issues.append("Possível corrupção de codificação (mojibake) detectada.")

        has_control_chars = bool(_CONTROL_CHARS_RE.search(original_text))
        if has_control_chars:
            issues.append("Caracteres de controle inválidos detectados no texto extraído.")

        has_cid_tokens = bool(_CID_TOKEN_RE.search(original_text))
        if has_cid_tokens:
            issues.append("Tokens CID não mapeados ('(cid:...)') detectados no texto extraído.")

        has_pua_chars = bool(_PUA_CHARS_RE.search(original_text))
        if has_pua_chars:
            issues.append("Caracteres Unicode de Área de Uso Privado (PUA) detectados.")

        tolerance_matches = [
            m.group(0) for m in _SUSPICIOUS_TOLERANCE_PLUS_RE.finditer(original_text)
        ]
        if tolerance_matches:
            issues.append(
                "Possível perda do símbolo '±' (extraído como '+') detectada em tolerância "
                f"técnica: {', '.join(tolerance_matches[:5])}."
            )

        ordinal_matches = [m.group(0) for m in _SUSPICIOUS_ORDINAL_RE.finditer(original_text)]
        if ordinal_matches:
            issues.append(
                "Possível perda do indicador ordinal 'º' (extraído como 'o') detectada em: "
                f"{', '.join(ordinal_matches[:5])}."
            )

        for anomaly_desc in unresolved_glyph_anomalies:
            issues.append(f"Anomalia de glifo/vetor não resolvida: {anomaly_desc}")

        for corr_issue in correction_issues:
            issues.append(f"Falha de correção verificada: {corr_issue}")

        table_issues: list[str] = []
        for tbl in tables:
            table_issues.extend(validate_table_integrity(tbl))
        issues.extend(table_issues)

        for own_issue in table_ownership_issues:
            issues.append(f"Associação proprietária de tabela bloqueada: {own_issue}")

        has_portuguese_accents = bool(_PT_ACCENTS_RE.search(original_text))

        unbalanced_parentheses, _end_depth, paren_issue = analyze_parentheses_balance(
            original_text,
            expected_incoming_open=expected_incoming_open_parens,
            allow_outgoing_open=allow_outgoing_open_parens,
        )
        if unbalanced_parentheses and paren_issue is not None:
            issues.append(paren_issue)

        passed = (
            not is_empty_or_incomplete
            and not has_replacement_chars
            and not has_mojibake_signals
            and not has_control_chars
            and not has_cid_tokens
            and not has_pua_chars
            and not tolerance_matches
            and not ordinal_matches
            and not unresolved_glyph_anomalies
            and not correction_issues
            and not table_issues
            and not table_ownership_issues
            and not unbalanced_parentheses
        )

        return QualityMetrics(
            char_count=char_count,
            line_count=line_count,
            word_count=word_count,
            has_replacement_chars=has_replacement_chars,
            has_mojibake_signals=has_mojibake_signals,
            has_portuguese_accents=has_portuguese_accents,
            unbalanced_parentheses=unbalanced_parentheses,
            is_empty_or_incomplete=is_empty_or_incomplete,
            passed_automated_checks=passed,
            issues=tuple(issues),
        )
