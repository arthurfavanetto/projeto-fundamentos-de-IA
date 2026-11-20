"""Testes unitários do redator determinístico de segredos em logs."""

from __future__ import annotations

import pytest

from rbac_assistant.security.secret_redactor import REDACTED, redact_mapping, redact_text


@pytest.mark.unit
def test_redact_text_masks_known_credential_patterns() -> None:
    sample = (
        "gemini=AIzaSyDummyKey1234567890ABCDEFGHIJ "
        "telegram=123456789:AAFakeTelegramBotToken_0123456789abcdef "
        "gh=ghp_1234567890abcdefghij1234567890 "
        "aws=AKIAIOSFODNN7EXAMPLE"
    )
    result = redact_text(sample)

    assert "AIzaSyDummyKey" not in result
    assert "AAFakeTelegramBotToken" not in result
    assert "ghp_1234567890" not in result
    assert "AKIAIOSFODNN7EXAMPLE" not in result
    assert result.count(REDACTED) == 4


@pytest.mark.unit
def test_redact_text_preserves_normative_aviation_content() -> None:
    normative = "RBAC 43 — Seção 43.3(a)(1): Pessoas autorizadas a executar manutenção."
    assert redact_text(normative) == normative


@pytest.mark.unit
def test_redact_mapping_masks_sensitive_keys_and_nested_values() -> None:
    payload = {
        "event": "startup",
        "gemini_api_key": "super-secret-value",
        "nested": {
            "telegram_bot_token": "987654321:XYZ_secret_token_value_1234567890123",
            "note": "Erro com AIzaSyDummyKey1234567890ABCDEFGHIJ no log",
        },
        "safe_items": ["RBAC 43", "AIzaSyDummyKey1234567890ABCDEFGHIJ"],
    }
    redacted = redact_mapping(payload)

    assert redacted["event"] == "startup"
    assert redacted["gemini_api_key"] == REDACTED
    assert redacted["nested"]["telegram_bot_token"] == REDACTED
    assert "AIzaSyDummyKey" not in redacted["nested"]["note"]
    assert redacted["safe_items"] == ["RBAC 43", REDACTED]
