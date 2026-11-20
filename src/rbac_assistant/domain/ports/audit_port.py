"""Contrato (Port) para registro estruturado de eventos de auditoria."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol


@dataclass(frozen=True, slots=True)
class AuditEvent:
    """Evento estruturado de auditoria operacional e de segurança."""

    timestamp: datetime
    action: str
    status: str
    actor_id: str | None = None
    role: str | None = None
    details: Mapping[str, str | int | float | bool] | None = None


class AuditPort(Protocol):
    """Interface para persistência de trilhas de auditoria (ex.: JSONL com redação)."""

    def record(self, event: AuditEvent) -> None:
        """Grava um evento de auditoria de forma durável."""
