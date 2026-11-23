"""Paginador fiel (`MessageSplitter`) e sanitização segura para mensagens do Telegram.

Garantias:
- Respeita o limite de 4.096 unidades UTF-16 / caracteres da API Telegram.
- Nunca resume, reescreve ou trunca texto normativo: concatenar as partes após remover
  apenas os cabeçalhos `[Parte X/Y]\\n` reconstitui 100% do texto original sanitizado.
- Trata Unicode, acentuação, emojis (pares substitutos UTF-16) e remove caracteres NUL.
- Aplica `redact_text` antes de qualquer envio externo para impedir vazamento de segredos.
"""

from __future__ import annotations

import re

from rbac_assistant.security.secret_redactor import redact_text

_PAGINATION_HEADER_RE = re.compile(r"^\[Parte \d+/\d+\]\n")


def utf16_length(text: str) -> int:
    """Calcula o comprimento em code units UTF-16 (métrica oficial do limite do Telegram)."""
    return len(text.encode("utf-16-le")) // 2


def sanitize_telegram_text(text: str) -> str:
    """Remove caracteres NUL, normaliza quebras de linha CRLF e redige segredos."""
    normalized = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    return redact_text(normalized)


def strip_pagination_header(message_part: str) -> str:
    """Remove apenas o cabeçalho `[Parte X/Y]\\n` de uma parte paginada."""
    return _PAGINATION_HEADER_RE.sub("", message_part, count=1)


def _find_split_index(text: str, max_utf16: int) -> int:
    """Encontra o maior índice de corte seguro sem perder caracteres nem exceder UTF-16."""
    # Determina o limite máximo de caracteres Python que cabem em `max_utf16` code units
    high = 0
    units = 0
    for idx, ch in enumerate(text):
        ch_units = 2 if ord(ch) > 0xFFFF else 1
        if units + ch_units > max_utf16:
            break
        units += ch_units
        high = idx + 1

    if high == 0:
        # Fallback de segurança extrema caso max_utf16 == 1 e primeiro char seja emoji (2 units)
        return 1
    if high >= len(text):
        return len(text)

    # Tenta quebrar em fronteiras naturais preservando 100% dos caracteres (incluindo \n / espaço)
    window = text[:high]
    min_acceptable = max(1, int(high * 0.5))

    para_pos = window.rfind("\n\n")
    if para_pos >= min_acceptable:
        return para_pos + 2

    line_pos = window.rfind("\n")
    if line_pos >= min_acceptable:
        return line_pos + 1

    space_pos = window.rfind(" ")
    if space_pos >= min_acceptable:
        return space_pos + 1

    return high


class MessageSplitter:
    """Divide respostas normativas longas em múltiplas mensagens compatíveis com o Telegram."""

    def __init__(self, *, max_message_length: int = 4096) -> None:
        if max_message_length < 64 or max_message_length > 4096:
            raise ValueError("max_message_length deve estar entre 64 e 4096.")
        self._max_message_length = max_message_length

    @property
    def max_message_length(self) -> int:
        return self._max_message_length

    def split(self, text: str) -> tuple[str, ...]:
        """Divide texto sanitizado em partes <= `max_message_length` sem perda de conteúdo."""
        sanitized = sanitize_telegram_text(text)
        if not sanitized:
            return ("",)

        if utf16_length(sanitized) <= self._max_message_length:
            return (sanitized,)

        # Reserva orçamento conservador para o cabeçalho `[Parte XXXX/XXXX]\n`
        # e recalcula caso o número de dígitos aumente (garantindo convergência determinística).
        header_reserve = 20
        parts_bodies: list[str] = []
        for _ in range(3):
            body_limit = self._max_message_length - header_reserve
            if body_limit < 16:
                raise ValueError("max_message_length insuficiente para cabeçalho de paginação.")

            parts_bodies = []
            remaining = sanitized
            while remaining:
                cut = _find_split_index(remaining, body_limit)
                parts_bodies.append(remaining[:cut])
                remaining = remaining[cut:]

            total = len(parts_bodies)
            max_header_len = utf16_length(f"[Parte {total}/{total}]\n")
            if max_header_len <= header_reserve:
                break
            header_reserve = max_header_len

        total = len(parts_bodies)
        formatted = tuple(
            f"[Parte {idx}/{total}]\n{body}" for idx, body in enumerate(parts_bodies, start=1)
        )
        return formatted
