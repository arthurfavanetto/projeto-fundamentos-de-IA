"""Composition Root da Etapa 1 — bootstrap de configuração, logs e verificação ChromaDB."""

from __future__ import annotations

import argparse
import signal
import sys
import threading
from types import FrameType

import structlog

from rbac_assistant.application.use_cases.check_system_status import CheckSystemStatusUseCase
from rbac_assistant.config.logging_config import configure_logging
from rbac_assistant.config.settings import load_settings
from rbac_assistant.infrastructure.vector_db.chroma_vector_store import ChromaVectorStore


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RBAC AI Assistant (Fundação — Etapa 1)")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Executa somente a verificação de inicialização e encerra.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Inicializa a fundação do serviço com encerramento gracioso por sinal."""
    args = _parse_args(argv)
    settings = load_settings()
    configure_logging(
        log_level=settings.log_level,
        redact_secrets=settings.security.audit.redact_secrets,
    )
    logger = structlog.get_logger("rbac_assistant.main")

    vector_store = ChromaVectorStore(
        host=settings.chroma_host,
        port=settings.chroma_port,
        heartbeat_path=settings.vector_db.heartbeat_path,
        timeout_seconds=settings.vector_db.request_timeout_seconds,
    )
    status_use_case = CheckSystemStatusUseCase(
        vector_store=vector_store,
        documents_dir=settings.storage.documents_dir,
        audit_logs_dir=settings.storage.audit_logs_dir,
    )
    report = status_use_case.execute()

    logger.info(
        "foundation_startup_check",
        environment=settings.environment,
        chroma_url=settings.chroma_heartbeat_url,
        chroma_healthy=report.chroma_healthy,
        chroma_heartbeat=report.chroma_heartbeat,
        documents_dir_writable=report.documents_dir_writable,
        audit_logs_dir_writable=report.audit_logs_dir_writable,
        ocr_policy=settings.extraction.ocr_unreviewed_policy,
        error=report.error,
    )

    if not report.healthy:
        logger.error("foundation_startup_failed", error=report.error)
        return 1

    if args.check_only:
        return 0

    stop_event = threading.Event()

    def _handle_shutdown(signum: int, _frame: FrameType | None) -> None:
        logger.info("shutdown_signal_received", signal=signum)
        stop_event.set()

    signal.signal(signal.SIGTERM, _handle_shutdown)
    signal.signal(signal.SIGINT, _handle_shutdown)

    logger.info(
        "foundation_ready",
        message="Fundação Etapa 1 operacional aguardando sinal de encerramento.",
    )
    stop_event.wait()
    logger.info("foundation_stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
