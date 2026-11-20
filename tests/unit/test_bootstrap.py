"""Testes unitários do logging estruturado, healthcheck e entrypoint main (--check-only)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import structlog

from rbac_assistant import healthcheck
from rbac_assistant import main as main_module
from rbac_assistant.config.logging_config import configure_logging
from rbac_assistant.config.settings import AppSettings, StorageConfig
from rbac_assistant.domain.exceptions import StorageConnectionError
from rbac_assistant.infrastructure.vector_db.chroma_vector_store import ChromaVectorStore


@pytest.mark.unit
def test_structured_logging_redacts_secrets_in_json_output(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_logging(log_level="INFO", redact_secrets=True)
    logger = structlog.get_logger("test.redaction")
    logger.info(
        "secret_check",
        api_key="AIzaSyDummyKey1234567890ABCDEFGHIJ",
        detail="bot=123456789:AAFakeTelegramBotToken_0123456789abcdef",
    )

    output = capsys.readouterr().out.strip()
    record = json.loads(output)
    assert record["api_key"] == "[REDACTED]"
    assert "AAFakeTelegramBotToken" not in record["detail"]
    assert "[REDACTED]" in record["detail"]


@pytest.mark.unit
def test_healthcheck_returns_zero_when_healthy_and_one_when_unhealthy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = AppSettings(
        _env_file=None,  # type: ignore[call-arg]
        storage=StorageConfig(
            documents_dir=tmp_path / "documents",
            audit_logs_dir=tmp_path / "logs",
        ),
    )
    monkeypatch.setattr(healthcheck, "load_settings", lambda: settings)
    monkeypatch.setattr(ChromaVectorStore, "check_heartbeat", lambda _self: 123456789)

    assert healthcheck.run_healthcheck() == 0
    payload_ok = json.loads(capsys.readouterr().out.strip())
    assert payload_ok["healthy"] is True
    assert payload_ok["chroma_heartbeat"] == 123456789

    def _raise(_self: ChromaVectorStore) -> int:
        raise StorageConnectionError("chromadb offline")

    monkeypatch.setattr(ChromaVectorStore, "check_heartbeat", _raise)
    assert healthcheck.run_healthcheck() == 1
    payload_fail = json.loads(capsys.readouterr().out.strip())
    assert payload_fail["healthy"] is False
    assert "chromadb offline" in payload_fail["error"]


@pytest.mark.unit
def test_main_check_only_exits_cleanly_on_healthy_startup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings(
        _env_file=None,  # type: ignore[call-arg]
        storage=StorageConfig(
            documents_dir=tmp_path / "documents",
            audit_logs_dir=tmp_path / "logs",
        ),
    )
    monkeypatch.setattr(main_module, "load_settings", lambda: settings)
    monkeypatch.setattr(ChromaVectorStore, "check_heartbeat", lambda _self: 999)

    assert main_module.main(["--check-only"]) == 0
