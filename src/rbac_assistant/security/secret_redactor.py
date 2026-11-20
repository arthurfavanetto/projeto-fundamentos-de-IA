"""Redação determinística de segredos e credenciais em logs e mensagens operacionais."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

REDACTED = "[REDACTED]"

_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Google Gemini / Google Cloud API Keys
    re.compile(r"AIza[0-9A-Za-z\-_]{20,}"),
    # Telegram Bot API Tokens (<numeric_bot_id>:<alphanumeric_hash>)
    re.compile(r"\b\d{7,12}:[A-Za-z0-9_-]{30,}\b"),
    # GitHub Classic & Fine-Grained Personal Access Tokens
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    # AWS Access Key IDs
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    # Blocos de chave privada PEM
    re.compile(
        r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z0-9 ]*PRIVATE KEY-----"
    ),
)

_SENSITIVE_KEY_FRAGMENTS: tuple[str, ...] = (
    "token",
    "api_key",
    "apikey",
    "secret",
    "password",
    "passwd",
    "authorization",
    "private_key",
)


def redact_text(value: str) -> str:
    """Substitui padrões conhecidos de credenciais por `[REDACTED]` em uma string."""
    redacted = value
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub(REDACTED, redacted)
    return redacted


def _is_sensitive_key(key: str) -> bool:
    normalized = key.lower().strip()
    return any(fragment in normalized for fragment in _SENSITIVE_KEY_FRAGMENTS)


def redact_value(value: Any) -> Any:
    """Aplica redação recursivamente em strings, listas, tuplas e dicionários."""
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, Mapping):
        return redact_mapping(value)
    if isinstance(value, list):
        return [redact_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_value(item) for item in value)
    return value


def redact_mapping(data: Mapping[str, Any]) -> dict[str, Any]:
    """Retorna cópia de um dicionário com chaves sensíveis e valores de segredos mascarados."""
    sanitized: dict[str, Any] = {}
    for key, raw_value in data.items():
        if _is_sensitive_key(str(key)) and raw_value not in (None, ""):
            sanitized[str(key)] = REDACTED
        else:
            sanitized[str(key)] = redact_value(raw_value)
    return sanitized
