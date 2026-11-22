"""Configuração de logging estruturado JSON com redação automática de segredos."""

from __future__ import annotations

import logging
import sys
from collections.abc import MutableMapping
from typing import Any, Literal, TextIO, cast

import structlog

from rbac_assistant.security.secret_redactor import redact_mapping


class _StdStreamProxy:
    """Proxy leve que resolve `sys.stdout`/`sys.stderr` dinamicamente a cada escrita.

    Garante compatibilidade com captura de streams em testes (`capsys`) e redirecionamentos
    em runtime sem reter referência obsoleta ao stream anterior.
    """

    def __init__(self, stream_name: Literal["stdout", "stderr"]) -> None:
        self._stream_name = stream_name

    def write(self, data: str) -> int:
        stream = getattr(sys, self._stream_name)
        return int(stream.write(data))

    def flush(self) -> None:
        stream = getattr(sys, self._stream_name)
        stream.flush()

    def isatty(self) -> bool:
        stream = getattr(sys, self._stream_name)
        return bool(stream.isatty())


def _redact_event_dict(
    _logger: Any,
    _method_name: str,
    event_dict: MutableMapping[str, Any],
) -> MutableMapping[str, Any]:
    """Processador structlog que mascara credenciais antes da serialização JSON."""
    return redact_mapping(event_dict)


def configure_logging(
    log_level: str = "INFO",
    *,
    redact_secrets: bool = True,
    stream: Literal["stdout", "stderr"] = "stdout",
) -> None:
    """Configura structlog e logging padrão para emissão estruturada em JSON."""
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    target_stream = cast(TextIO, _StdStreamProxy(stream))
    logging.basicConfig(
        format="%(message)s",
        stream=target_stream,
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
        logger_factory=structlog.PrintLoggerFactory(file=target_stream),
        cache_logger_on_first_use=False,
    )


def configure_cli_logging(log_level: str = "INFO", *, redact_secrets: bool = True) -> None:
    """Configura logs de CLI exclusivamente em `stderr`, mantendo `stdout` puro para `--json`."""
    configure_logging(log_level=log_level, redact_secrets=redact_secrets, stream="stderr")
