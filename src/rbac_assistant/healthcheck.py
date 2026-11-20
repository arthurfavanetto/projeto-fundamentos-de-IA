"""Utilitário executável de healthcheck do contêiner `rbac-bot`."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict

from rbac_assistant.application.use_cases.check_system_status import CheckSystemStatusUseCase
from rbac_assistant.config.settings import load_settings
from rbac_assistant.infrastructure.vector_db.chroma_vector_store import ChromaVectorStore


def run_healthcheck() -> int:
    """Executa a verificação de saúde interna e retorna código de saída POSIX."""
    settings = load_settings()
    vector_store = ChromaVectorStore(
        host=settings.chroma_host,
        port=settings.chroma_port,
        heartbeat_path=settings.vector_db.heartbeat_path,
        timeout_seconds=settings.vector_db.request_timeout_seconds,
    )
    use_case = CheckSystemStatusUseCase(
        vector_store=vector_store,
        documents_dir=settings.storage.documents_dir,
        audit_logs_dir=settings.storage.audit_logs_dir,
    )
    report = use_case.execute()
    print(json.dumps(asdict(report), ensure_ascii=False))
    return 0 if report.healthy else 1


def main() -> None:
    """Entrypoint CLI (`python -m rbac_assistant.healthcheck`)."""
    sys.exit(run_healthcheck())


if __name__ == "__main__":
    main()
