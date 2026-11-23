"""Smoke test operacional de persistência e redação da auditoria JSONL em `/app/logs`.

Executado dentro do contêiner `rbac-bot` (UID/GID 10001) ou localmente com `--audit-logs-dir`:
    python scripts/verify_audit_persistence.py --phase write
    docker compose restart rbac-bot
    python scripts/verify_audit_persistence.py --phase verify
    python scripts/verify_audit_persistence.py --phase cleanup
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

from rbac_assistant.config.settings import load_settings
from rbac_assistant.domain.ports.audit_port import AuditEvent
from rbac_assistant.infrastructure.audit.jsonl_audit_adapter import JsonlAuditAdapter

SMOKE_AUDIT_FILENAME = "audit_smoke_persistence.jsonl"
SMOKE_ACTION = "audit_persistence_smoke_probe"
SMOKE_PROBE_ID = "rbac_audit_smoke_001"
SYNTHETIC_API_KEY = "AIzaSySmokeFakeKeyNeverReal1234567890"
SYNTHETIC_TG_TOKEN = "123456789:AAFakeTelegramSmokeTokenNeverReal0123456789"


def run_phase(phase: str, *, audit_logs_dir: Path | None = None) -> int:
    """Executa uma fase do smoke test de auditoria persistente em JSONL."""
    settings = load_settings()
    target_dir = audit_logs_dir or settings.storage.audit_logs_dir
    adapter = JsonlAuditAdapter.from_settings(
        settings,
        logs_dir=target_dir,
        filename=SMOKE_AUDIT_FILENAME,
    )

    if phase == "write":
        adapter.record(
            AuditEvent(
                timestamp=datetime.now(UTC),
                action=SMOKE_ACTION,
                status="ok",
                actor_id="smoke_probe",
                role="ADMIN",
                details={
                    "probe_id": SMOKE_PROBE_ID,
                    "api_key": SYNTHETIC_API_KEY,
                    "detail_with_token": f"tg={SYNTHETIC_TG_TOKEN}",
                    "user_prompt": "conteudo sensivel que nunca deve aparecer em claro",
                },
            )
        )
        print(f"[OK] Evento JSONL de auditoria gravado em {adapter.audit_file_path}.")
        return 0

    if phase == "verify":
        if not adapter.audit_file_path.exists():
            raise AssertionError(
                f"Arquivo de auditoria não persistiu após restart: {adapter.audit_file_path}"
            )
        raw_content = adapter.audit_file_path.read_text(encoding="utf-8")
        if SYNTHETIC_API_KEY in raw_content or SYNTHETIC_TG_TOKEN in raw_content:
            raise AssertionError("Segredo sintético vazou em texto claro no arquivo JSONL!")
        if "conteudo sensivel" in raw_content:
            raise AssertionError("Conteúdo de prompt sensível vazou no arquivo JSONL!")

        events = adapter.read_events()
        matching = [
            e
            for e in events
            if e.get("event_type") == SMOKE_ACTION
            and e.get("metadata", {}).get("probe_id") == SMOKE_PROBE_ID
        ]
        if not matching:
            raise AssertionError("Evento de smoke test não encontrado após restart.")
        ev = matching[-1]
        if not str(ev.get("timestamp", "")).endswith("Z"):
            raise AssertionError(f"Timestamp não está em UTC ISO-8601: {ev.get('timestamp')!r}")
        if ev["metadata"].get("api_key") != "[REDACTED]":
            raise AssertionError(f"api_key não redigida: {ev['metadata'].get('api_key')!r}")
        if "[REDACTED]" not in str(ev["metadata"].get("detail_with_token", "")):
            raise AssertionError("Token Telegram embutido não foi redigido.")
        if ev["metadata"].get("user_prompt") != "[REDACTED]":
            raise AssertionError("Campo sensível user_prompt não foi bloqueado.")
        print(
            "[OK] Persistência JSONL em /app/logs, timestamp UTC e redação pós-restart validados."
        )
        return 0

    if phase == "cleanup":
        lock_file = target_dir / f".{SMOKE_AUDIT_FILENAME}.lock"
        for candidate in (adapter.audit_file_path, lock_file):
            if candidate.exists():
                candidate.unlink()
        print("[OK] Artefatos temporários de smoke test de auditoria removidos.")
        return 0

    raise ValueError(f"Fase inválida: {phase}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Valida escrita, redação e persistência JSONL da auditoria em /app/logs."
    )
    parser.add_argument(
        "--phase",
        choices=("write", "verify", "cleanup"),
        required=True,
        help="Fase do smoke test.",
    )
    parser.add_argument(
        "--audit-logs-dir",
        type=Path,
        help="Sobrescreve o diretório de logs de auditoria.",
    )
    args = parser.parse_args(argv)
    return run_phase(args.phase, audit_logs_dir=args.audit_logs_dir)


if __name__ == "__main__":
    sys.exit(main())
