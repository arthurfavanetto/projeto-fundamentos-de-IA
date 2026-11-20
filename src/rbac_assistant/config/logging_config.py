"""Configuração de logging estruturado JSON com redação automática de segredos."""

from __future__ import annotations

import logging
import sys
from collections.abc import MutableMapping
from typing import Any

import structlog

from rbac_assistant.security.secret_redactor import redact_mapping


def _redact_event_dict(
    _logger: Any,
    _method_name: str,
    event_dict: MutableMapping[str, Any],
) -> MutableMapping[str, Any]:
    """Processador structlog que mascara credenciais antes da serialização JSON."""
    return redact_mapping(event_dict)


def configure_logging(log_level: str = "INFO", *, redact_secrets: bool = True) -> None:
    """Configura structlog e logging padrão para emissão estruturada em JSON."""
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=numeric_level,
        force=True,
    )

    processors: list[structlog.types.Processor] = [
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    if redact_secrets:
        processors.append(_redact_event_dict)
    processors.append(structlog.processors.JSONRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
