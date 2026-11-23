"""Caso de uso de verificação operacional do sistema (ChromaDB e volumes)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from rbac_assistant.domain.exceptions import AuditWriteError, StorageConnectionError
from rbac_assistant.domain.ports.audit_port import AuditEvent, AuditPort
from rbac_assistant.domain.ports.vector_store_port import VectorStorePort


@dataclass(frozen=True, slots=True)
class SystemStatusReport:
    """Relatório estruturado de saúde operacional da fundação do sistema."""

    healthy: bool
    chroma_healthy: bool
    chroma_heartbeat: int | None
    documents_dir_writable: bool
    audit_logs_dir_writable: bool
    error: str | None = None


def _is_directory_writable(directory: Path) -> bool:
    """Verifica se o diretório existe (ou pode ser criado) e permite escrita."""
    try:
        directory.mkdir(parents=True, exist_ok=True)
        return directory.is_dir() and os.access(directory, os.W_OK | os.R_OK)
    except OSError:
        return False


class CheckSystemStatusUseCase:
    """Verifica conectividade interna com o ChromaDB e acesso aos volumes persistentes."""

    def __init__(
        self,
        *,
        vector_store: VectorStorePort,
        documents_dir: Path,
        audit_logs_dir: Path,
        audit_port: AuditPort | None = None,
    ) -> None:
        self._vector_store = vector_store
        self._documents_dir = documents_dir
        self._audit_logs_dir = audit_logs_dir
        self._audit_port = audit_port

    def execute(self) -> SystemStatusReport:
        """Executa verificação de saúde e retorna relatório consolidado."""
        docs_writable = _is_directory_writable(self._documents_dir)
        logs_writable = _is_directory_writable(self._audit_logs_dir)

        chroma_healthy = False
        chroma_heartbeat: int | None = None
        error: str | None = None

        try:
            chroma_heartbeat = self._vector_store.check_heartbeat()
            chroma_healthy = True
        except StorageConnectionError as exc:
            error = str(exc)

        if error is None and (not docs_writable or not logs_writable):
            unwritable = []
            if not docs_writable:
                unwritable.append(str(self._documents_dir))
            if not logs_writable:
                unwritable.append(str(self._audit_logs_dir))
            error = f"Diretório(s) sem permissão de escrita: {', '.join(unwritable)}"

        healthy = chroma_healthy and docs_writable and logs_writable

        if self._audit_port is not None and logs_writable:
            try:
                self._audit_port.record(
                    AuditEvent(
                        timestamp=datetime.now(UTC),
                        action="system_status_check",
                        status="healthy" if healthy else "unhealthy",
                        actor_id="system",
                        details={
                            "chroma_healthy": chroma_healthy,
                            "documents_dir_writable": docs_writable,
                            "audit_logs_dir_writable": logs_writable,
                            "has_error": error is not None,
                        },
                    )
                )
            except AuditWriteError as exc:
                logs_writable = False
                healthy = False
                error = str(exc)

        return SystemStatusReport(
            healthy=healthy,
            chroma_healthy=chroma_healthy,
            chroma_heartbeat=chroma_heartbeat,
            documents_dir_writable=docs_writable,
            audit_logs_dir_writable=logs_writable,
            error=error,
        )
