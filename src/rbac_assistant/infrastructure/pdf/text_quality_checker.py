"""Analisador automatizado de qualidade e integridade textual por página.

IMPORTANTE: Este verificador produz a classificação automatizada
`native_quality_checked` — jamais deve ser confundido com revisão humana.
"""

from __future__ import annotations

import re

from rbac_assistant.domain.entities.regulatory_page import QualityMetrics

_PT_ACCENTS_RE = re.compile(r"[áàâãéêíóôõúçÁÀÂÃÉÊÍÓÔÕÚÇ]")
_MOJIBAKE_RE = re.compile(r"(?:Ã[§£¡©ª­³µº]|â€[œ“”–—])")
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


class TextQualityChecker:
    """Implementação de `TextQualityCheckerPort` para páginas extraídas nativamente."""

    def evaluate(
        self,
        *,
        original_text: str,
        normalized_text: str,
        min_chars: int = 50,
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

        if _CONTROL_CHARS_RE.search(original_text):
            issues.append("Caracteres de controle inválidos detectados no texto extraído.")

        has_portuguese_accents = bool(_PT_ACCENTS_RE.search(original_text))

        open_parens = original_text.count("(")
        close_parens = original_text.count(")")
        # Pequena tolerância para parênteses que atravessam quebra de página
        unbalanced_parentheses = abs(open_parens - close_parens) > 2
        if unbalanced_parentheses:
            issues.append(
                f"Desbalanceamento acentuado de parênteses "
                f"({open_parens} '(' vs {close_parens} ')')."
            )

        passed = (
            not is_empty_or_incomplete
            and not has_replacement_chars
            and not has_mojibake_signals
            and not bool(_CONTROL_CHARS_RE.search(original_text))
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
