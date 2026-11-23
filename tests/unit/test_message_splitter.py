"""Testes unitários do `MessageSplitter` (limite 4096 UTF-16, reconstrução 100% fiel e Unicode)."""

from __future__ import annotations

import pytest

from rbac_assistant.interfaces.telegram.message_splitter import (
    MessageSplitter,
    strip_pagination_header,
    utf16_length,
)


@pytest.mark.unit
def test_message_splitter_keeps_short_message_single_part() -> None:
    splitter = MessageSplitter(max_message_length=4096)
    text = "RBAC 43.3 — Pessoas autorizadas a executar manutenção."
    parts = splitter.split(text)
    assert parts == (text,)


@pytest.mark.unit
def test_message_splitter_reconstructs_100_percent_exact_normative_text() -> None:
    splitter = MessageSplitter(max_message_length=256)
    paragraphs = [
        f"43.{i} Dispositivo normativo — manutenção preventiva, reconstrução e alteração "
        f"(a)(1)(i); pontuação: exata, com acentuação: aeronavegabilidade, inspeção, hélice.\n"
        for i in range(1, 35)
    ]
    original = "\n".join(paragraphs)
    parts = splitter.split(original)

    assert len(parts) > 5
    for idx, part in enumerate(parts, start=1):
        assert part.startswith(f"[Parte {idx}/{len(parts)}]\n")
        assert utf16_length(part) <= 256

    reconstructed = "".join(strip_pagination_header(p) for p in parts)
    assert reconstructed == original


@pytest.mark.unit
def test_message_splitter_handles_unicode_emojis_utf16_and_giant_tokens() -> None:
    splitter = MessageSplitter(max_message_length=128)
    # Emojis ocupam 2 code units em UTF-16 cada
    emoji_text = "✈️ Regulamento ANAC RBAC 43 " + ("🛩️" * 90) + "\n" + ("X" * 300)
    parts = splitter.split(emoji_text)

    assert len(parts) >= 4
    for part in parts:
        assert utf16_length(part) <= 128

    reconstructed = "".join(strip_pagination_header(p) for p in parts)
    assert reconstructed == emoji_text


@pytest.mark.unit
def test_message_splitter_sanitizes_null_bytes_and_redacts_secrets() -> None:
    splitter = MessageSplitter(max_message_length=4096)
    fake_token = "123456789:AAFakeTelegramTokenSecret0123456789012"
    fake_api_key = "AIzaSyFakeSecretApiKeyNeverReal1234567890"
    raw = f"Texto\x00 limpo com {fake_token} e {fake_api_key}\r\nLinha 2"

    parts = splitter.split(raw)
    assert len(parts) == 1
    assert "\x00" not in parts[0]
    assert "\r" not in parts[0]
    assert fake_token not in parts[0]
    assert fake_api_key not in parts[0]
    assert parts[0].count("[REDACTED]") == 2
