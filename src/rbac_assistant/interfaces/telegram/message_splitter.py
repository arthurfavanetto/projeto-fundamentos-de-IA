"""Paginador fiel (`MessageSplitter`) e proteção literal para mensagens do Telegram.

Garantias:
- Respeita o limite de 4.096 code units UTF-16 da API Telegram.
- Nunca resume, reescreve, normaliza ou altera silenciosamente a transcrição normativa literal:
  em `split_verbatim` / `split_normative_answer`, o trecho autoritativo não passa por
  normalização de quebras de linha nem remoção silenciosa de caracteres.
- Se um trecho literal autoritativo contiver byte NUL (`\\x00`) ou acionar redação automática,
  o sistema bloqueia sua apresentação literal (fail-closed) em vez de entregar texto alterado.
- Concatenar as partes após remover apenas o cabeçalho externo `[Parte X/Y]\\n` reconstitui
  100% da sequência exata de caracteres, caractere por caractere.
"""

from __future__ import annotations

import re

from rbac_assistant.domain.entities.rag import StructuredNormativeAnswer
from rbac_assistant.rag.generation.response_formatter import render_structured_answer_text
from rbac_assistant.security.secret_redactor import redact_text

_PAGINATION_HEADER_RE = re.compile(r"^\[Parte \d+/\d+\]\n")


def utf16_length(text: str) -> int:
    """Calcula o comprimento em code units UTF-16 (métrica oficial do limite do Telegram)."""
    return len(text.encode("utf-16-le")) // 2


def sanitize_telegram_text(text: str) -> str:
    """Remove caracteres NUL, normaliza CRLF e redige segredos em textos NÃO literais."""
    normalized = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    return redact_text(normalized)


def check_literal_delivery_integrity(literal_text: str) -> tuple[str, ...]:
    """Verifica se um trecho literal pode ser entregue no Telegram sem qualquer modificação.

    Retorna tupla vazia se puder ser entregue byte-a-byte sem alteração; caso contrário retorna
    motivos objetivos de bloqueio fail-closed.
    """
    issues: list[str] = []
    if "\x00" in literal_text:
        issues.append(
            "Trecho normativo contém caractere NUL ('\\x00'), incompatível com entrega "
            "literal sem modificação no Telegram."
        )
    if redact_text(literal_text) != literal_text:
        issues.append(
            "Trecho normativo acionaria redação automática de segredos, impedindo "
            "apresentação literal inalterada."
        )
    return tuple(issues)


def strip_pagination_header(message_part: str) -> str:
    """Remove apenas o cabeçalho externo `[Parte X/Y]\\n` de uma parte paginada."""
    return _PAGINATION_HEADER_RE.sub("", message_part, count=1)


def _find_split_index(text: str, max_utf16: int) -> int:
    """Encontra o maior índice de corte seguro sem perder caracteres nem exceder UTF-16."""
    high = 0
    units = 0
    for idx, ch in enumerate(text):
        ch_units = 2 if ord(ch) > 0xFFFF else 1
        if units + ch_units > max_utf16:
            break
        units += ch_units
        high = idx + 1

    if high == 0:
        return 1
    if high >= len(text):
        return len(text)

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

    def _split_raw(self, payload: str) -> tuple[str, ...]:
        """Divide `payload` diretamente por orçamento UTF-16 sem alterar nenhum caractere."""
        if not payload:
            return ("",)

        if utf16_length(payload) <= self._max_message_length:
            return (payload,)

        header_reserve = 20
        parts_bodies: list[str] = []
        for _ in range(3):
            body_limit = self._max_message_length - header_reserve
            if body_limit < 16:
                raise ValueError("max_message_length insuficiente para cabeçalho de paginação.")

            parts_bodies = []
            remaining = payload
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
        return tuple(
            f"[Parte {idx}/{total}]\n{body}" for idx, body in enumerate(parts_bodies, start=1)
        )

    def split(self, text: str) -> tuple[str, ...]:
        """Divide texto não-normativo sanitizado em partes <= `max_message_length`."""
        sanitized = sanitize_telegram_text(text)
        return self._split_raw(sanitized)

    def split_verbatim(self, text: str) -> tuple[str, ...]:
        """Divide texto autoritativo diretamente sem sanitização ou normalização silenciosa.

        Lança `ValueError` caso o texto contenha byte NUL ou acione redação de segredos.
        """
        issues = check_literal_delivery_integrity(text)
        if issues:
            raise ValueError("; ".join(issues))
        return self._split_raw(text)

    def split_normative_answer(self, answer: StructuredNormativeAnswer) -> tuple[str, ...]:
        """Prepara resposta normativa para Telegram sem jamais modificar a transcrição literal.

        - Sanitiza/redige exclusivamente blocos externos (explicação IA, referências, observações).
        - Bloqueia a apresentação literal (fail-closed) caso qualquer trecho literal autoritativo
          ou tabela estruturada não possa ser entregue sem modificação silenciosa.
        """
        if not answer.is_literal_validated or answer.normative_text is None:
            return self.split(answer.formatted_text)

        literal_issues: list[str] = []
        for excerpt in answer.literal_excerpts:
            literal_issues.extend(check_literal_delivery_integrity(excerpt))
        literal_issues.extend(check_literal_delivery_integrity(answer.normative_text))
        if answer.structured_tables_text:
            literal_issues.extend(
                check_literal_delivery_integrity(answer.structured_tables_text)
            )

        if literal_issues:
            reason = "; ".join(dict.fromkeys(literal_issues))
            blocked_text = render_structured_answer_text(
                normative_text=None,
                preliminary_excerpt=None,
                explanation=(
                    "BLOQUEIO DE FIDELIDADE LITERAL NO ENVIO: A transcrição normativa não pôde "
                    f"ser preparada para entrega sem modificação ({reason}). Por política "
                    "fail-closed, o assistente bloqueou a exibição literal em vez de alterar o "
                    "texto autoritativo."
                ),
                references=tuple(sanitize_telegram_text(r) for r in answer.references),
                observations=tuple(sanitize_telegram_text(o) for o in answer.observations),
                include_explanation_block=True,
                explanation_heading="=== INFORMAÇÃO / RECUSA FUNDAMENTADA ===",
            )
            return self.split(blocked_text)

        # Sanitiza apenas blocos externos (explicação, referências, observações),
        # preservando 100% inalterados `answer.normative_text` e `answer.structured_tables_text`.
        is_literal_only_no_llm = (
            answer.found
            and answer.used_fallback_without_llm
            and answer.generation_model is None
        )
        safe_explanation = sanitize_telegram_text(answer.explanation)
        safe_references = tuple(sanitize_telegram_text(ref) for ref in answer.references)
        safe_observations = tuple(sanitize_telegram_text(obs) for obs in answer.observations)
        assembled = render_structured_answer_text(
            normative_text=answer.normative_text,
            preliminary_excerpt=None,
            explanation=safe_explanation,
            references=safe_references,
            observations=safe_observations,
            structured_tables_text=answer.structured_tables_text,
            include_explanation_block=not is_literal_only_no_llm,
            explanation_heading="=== EXPLICAÇÃO (GERADA POR IA) ===",
        )
        return self._split_raw(assembled)
