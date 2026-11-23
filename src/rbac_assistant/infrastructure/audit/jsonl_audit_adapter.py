"""Adaptador concreto de auditoria persistente em JSONL no volume `/app/logs`.

Responsabilidades e garantias:
- Implementa o contrato de domínio `AuditPort` preservando a Arquitetura Hexagonal.
- Persiste eventos estruturados em formato JSONL (1 objeto JSON UTF-8 por linha) com
  timestamp UTC ISO-8601 (`Z`), tipo de evento, resultado e metadados seguros.
- Aplica redação determinística de segredos (`secret_redactor`) e bloqueia campos
  sensíveis (credenciais, prompts integrais e conteúdo documental bruto).
- Garante durabilidade com `flush()` + `os.fsync()` e falha explícita via
  `AuditWriteError` em qualquer erro de disco ou permissão (nunca simula sucesso).
- Suporta escrita concorrente segura (lock de thread por arquivo + lock de processo
  via `fcntl.flock` em POSIX / `msvcrt.locking` em Windows) e rotação por tamanho
  independente da rotação `json-file` de stdout/stderr do Docker.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rbac_assistant.domain.exceptions import AuditWriteError
from rbac_assistant.domain.ports.audit_port import AuditEvent
from rbac_assistant.security.secret_redactor import REDACTED, redact_mapping, redact_text

if TYPE_CHECKING:
    from rbac_assistant.config.settings import AppSettings

_REGISTRY_LOCK = threading.Lock()
_PATH_LOCKS: dict[Path, threading.Lock] = {}

_FORBIDDEN_CONTENT_KEY_FRAGMENTS: tuple[str, ...] = (
    "prompt",
    "raw_text",
    "original_text",
    "document_text",
    "full_text",
    "pdf_bytes",
    "pdf_content",
    "credential",
    "bearer",
    "cookie",
    "jwt",
    "session_id",
    "chat_history",
)


def _get_path_lock(resolved_path: Path) -> threading.Lock:
    with _REGISTRY_LOCK:
        lock = _PATH_LOCKS.get(resolved_path)
        if lock is None:
            lock = threading.Lock()
            _PATH_LOCKS[resolved_path] = lock
        return lock


@contextmanager
def _interprocess_file_lock(lock_path: Path) -> Iterator[None]:
    """Serializa gravações entre múltiplos processos de forma portável (Linux/Windows)."""
    lock_file = lock_path.open("a+b")
    try:
        if sys.platform == "win32":
            import msvcrt

            lock_file.seek(0)
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    finally:
        lock_file.close()


def _format_utc_iso8601(ts: datetime) -> str:
    """Converte qualquer datetime para timestamp UTC ISO-8601 com sufixo `Z`."""
    utc_dt = ts.replace(tzinfo=UTC) if ts.tzinfo is None else ts.astimezone(UTC)
    return utc_dt.isoformat().replace("+00:00", "Z")


def _validate_filename(filename: str) -> str:
    cleaned = filename.strip()
    if (
        not cleaned
        or "/" in cleaned
        or "\\" in cleaned
        or ".." in cleaned
        or cleaned in {".", ".."}
    ):
        raise ValueError(f"Nome de arquivo de auditoria inválido: {filename!r}")
    return cleaned


def _is_forbidden_content_key(key: str) -> bool:
    normalized = key.lower().strip()
    return any(fragment in normalized for fragment in _FORBIDDEN_CONTENT_KEY_FRAGMENTS)


class JsonlAuditAdapter:
    """Adaptador persistente de `AuditPort` em arquivos JSONL rotacionáveis."""

    SCHEMA_VERSION = "1.0"

    def __init__(
        self,
        *,
        logs_dir: Path,
        filename: str = "audit_events.jsonl",
        pseudonymize_user_ids: bool = False,
        redact_secrets: bool = True,
        max_file_size_bytes: int = 10 * 1024 * 1024,
        max_rotated_files: int = 10,
        max_detail_string_length: int = 300,
        file_mode: int = 0o640,
    ) -> None:
        if not redact_secrets:
            raise ValueError(
                "A redação de segredos na auditoria persistente não pode ser desabilitada."
            )
        if max_file_size_bytes < 256:
            raise ValueError("max_file_size_bytes deve ser >= 256 bytes.")
        if max_rotated_files < 1:
            raise ValueError("max_rotated_files deve ser >= 1.")
        if max_detail_string_length < 32:
            raise ValueError("max_detail_string_length deve ser >= 32 caracteres.")

        safe_filename = _validate_filename(filename)
        self._logs_dir = logs_dir
        self._file_path = logs_dir / safe_filename
        self._lock_path = logs_dir / f".{safe_filename}.lock"
        self._pseudonymize_user_ids = pseudonymize_user_ids
        self._max_file_size_bytes = max_file_size_bytes
        self._max_rotated_files = max_rotated_files
        self._max_detail_string_length = max_detail_string_length
        self._file_mode = file_mode

    @classmethod
    def from_settings(
        cls,
        settings: AppSettings,
        *,
        logs_dir: Path | None = None,
        filename: str | None = None,
    ) -> JsonlAuditAdapter:
        """Constrói o adaptador a partir de `AppSettings` (com overrides opcionais)."""
        audit_cfg = settings.security.audit
        return cls(
            logs_dir=logs_dir or settings.storage.audit_logs_dir,
            filename=filename or audit_cfg.filename,
            pseudonymize_user_ids=audit_cfg.pseudonymize_user_ids,
            redact_secrets=audit_cfg.redact_secrets,
            max_file_size_bytes=audit_cfg.max_file_size_bytes,
            max_rotated_files=audit_cfg.max_rotated_files,
            max_detail_string_length=audit_cfg.max_detail_string_length,
        )

    @property
    def logs_dir(self) -> Path:
        """Diretório configurado para armazenamento dos arquivos JSONL."""
        return self._logs_dir

    @property
    def audit_file_path(self) -> Path:
        """Caminho do arquivo JSONL ativo."""
        return self._file_path

    def _sanitize_actor(self, actor_id: str | None) -> str | None:
        if actor_id is None:
            return None
        cleaned = redact_text(actor_id.strip())
        if not cleaned:
            return None
        if self._pseudonymize_user_ids:
            digest = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()[:16]
            return f"usr_{digest}"
        return self._truncate(cleaned)

    def _truncate(self, value: str) -> str:
        if len(value) <= self._max_detail_string_length:
            return value
        return f"{value[: self._max_detail_string_length - 3]}..."

    def _sanitize_metadata(
        self,
        details: Mapping[str, str | int | float | bool] | None,
    ) -> dict[str, str | int | float | bool]:
        if not details:
            return {}
        pre_filtered: dict[str, Any] = {}
        for raw_key, raw_value in details.items():
            key = str(raw_key).strip()
            if not key:
                continue
            if _is_forbidden_content_key(key):
                pre_filtered[key] = REDACTED
            else:
                pre_filtered[key] = raw_value

        redacted = redact_mapping(pre_filtered)
        sanitized: dict[str, str | int | float | bool] = {}
        for key, value in redacted.items():
            if isinstance(value, bool | int | float):
                sanitized[key] = value
            elif value is None:
                continue
            else:
                sanitized[key] = self._truncate(redact_text(str(value)))
        return sanitized

    def _serialize_event(self, event: AuditEvent) -> bytes:
        action = self._truncate(redact_text(event.action.strip() or "unknown_event"))
        status = self._truncate(redact_text(event.status.strip() or "unknown"))
        role = (
            self._truncate(redact_text(event.role.strip()))
            if event.role and event.role.strip()
            else None
        )
        payload: dict[str, Any] = {
            "schema_version": self.SCHEMA_VERSION,
            "timestamp": _format_utc_iso8601(event.timestamp),
            "event_type": action,
            "outcome": status,
            "actor_id": self._sanitize_actor(event.actor_id),
            "role": role,
            "metadata": self._sanitize_metadata(event.details),
        }
        line = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        return (line + "\n").encode("utf-8")

    def _rotated_path(self, index: int) -> Path:
        return self._logs_dir / f"{self._file_path.name}.{index}"

    def _rotate_if_needed(self, incoming_bytes: int) -> None:
        if not self._file_path.exists():
            return
        current_size = self._file_path.stat().st_size
        if current_size == 0 or (current_size + incoming_bytes) <= self._max_file_size_bytes:
            return

        # Remove apenas o arquivo mais antigo excedendo `max_rotated_files`
        oldest = self._rotated_path(self._max_rotated_files)
        if oldest.exists():
            oldest.unlink()

        for idx in range(self._max_rotated_files - 1, 0, -1):
            src = self._rotated_path(idx)
            dst = self._rotated_path(idx + 1)
            if src.exists():
                os.replace(src, dst)

        os.replace(self._file_path, self._rotated_path(1))

    def _apply_safe_permissions(self, path: Path) -> None:
        if sys.platform != "win32":
            with suppress(OSError):
                # Alguns filesystems virtuais não suportam chmod sem falhar escrita
                os.chmod(path, self._file_mode)

    def record(self, event: AuditEvent) -> None:
        """Persiste um evento de auditoria de forma durável em JSONL.

        Raises:
            AuditWriteError: se não for possível criar o diretório, adquirir lock,
                rotacionar ou gravar/sincronizar o evento em disco.
        """
        encoded = self._serialize_event(event)
        resolved_key = self._file_path.resolve()
        thread_lock = _get_path_lock(resolved_key)

        with thread_lock:
            try:
                self._logs_dir.mkdir(parents=True, exist_ok=True)
                with _interprocess_file_lock(self._lock_path):
                    self._rotate_if_needed(len(encoded))
                    with self._file_path.open("ab") as handle:
                        handle.write(encoded)
                        handle.flush()
                        os.fsync(handle.fileno())
                    self._apply_safe_permissions(self._file_path)
            except OSError as exc:
                safe_error = redact_text(str(exc))
                raise AuditWriteError(
                    f"Falha ao persistir evento de auditoria '{event.action}' em "
                    f"'{self._file_path}': {safe_error}"
                ) from exc

    def read_events(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        """Lê eventos JSONL já persistidos no arquivo ativo para inspeção/auditoria."""
        if not self._file_path.exists():
            return []
        raw_lines = self._file_path.read_text(encoding="utf-8").splitlines()
        events = [json.loads(line) for line in raw_lines if line.strip()]
        if limit is not None and limit >= 0:
            return events[-limit:] if limit > 0 else []
        return events
