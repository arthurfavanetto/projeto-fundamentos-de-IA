"""Testes unitários de carregamento e validação das configurações Pydantic."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from rbac_assistant.config.settings import AppSettings, VectorDBConfig, load_settings


@pytest.mark.unit
def test_load_settings_from_repository_yaml(
    repo_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CHROMA_HOST", raising=False)
    monkeypatch.delenv("CHROMA_PORT", raising=False)

    settings = load_settings(yaml_path=repo_root / "config" / "settings.yaml")

    assert settings.chroma_host == "chromadb"
    assert settings.chroma_port == 8000
    assert settings.vector_db.pinned_version == "1.5.9"
    assert settings.vector_db.heartbeat_path == "/api/v2/heartbeat"
    assert settings.chroma_heartbeat_url == "http://chromadb:8000/api/v2/heartbeat"
    assert settings.extraction.ocr_unreviewed_policy == "block_until_reviewed"
    assert settings.extraction.require_integrity_check_for_native_pages is True
    assert settings.security.deny_by_default is True
    assert settings.gemini_generation_model == "gemini-4-hydrogen"
    assert settings.gemini_embedding_model == "gemini-embedding-2"
    assert settings.gemini_allow_paid_models is False


@pytest.mark.unit
def test_env_overrides_models_and_chroma_endpoint(
    repo_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CHROMA_HOST", "internal-chroma")
    monkeypatch.setenv("CHROMA_PORT", "8100")
    monkeypatch.setenv("GEMINI_GENERATION_MODEL", "gemini-3.8-flash")
    monkeypatch.setenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")

    settings = load_settings(yaml_path=repo_root / "config" / "settings.yaml")

    assert settings.chroma_base_url == "http://internal-chroma:8100"
    assert settings.gemini_generation_model == "gemini-3.8-flash"
    assert settings.gemini_embedding_model == "gemini-embedding-2"
    assert settings.gemini_allow_paid_models is False
    assert settings.log_level == "DEBUG"


@pytest.mark.unit
def test_invalid_port_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHROMA_PORT", "70000")
    with pytest.raises(ValidationError):
        AppSettings()


@pytest.mark.unit
def test_invalid_heartbeat_path_fails_fast() -> None:
    with pytest.raises(ValidationError, match="heartbeat_path"):
        VectorDBConfig(heartbeat_path="api/v2/heartbeat")


@pytest.mark.unit
def test_explicit_missing_yaml_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_settings(yaml_path=tmp_path / "missing.yaml")


@pytest.mark.unit
def test_secret_helpers_fail_when_missing_and_mask_when_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)

    settings_empty = AppSettings(_env_file=None)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        settings_empty.require_gemini_api_key()
    with pytest.raises(ValueError, match="TELEGRAM_BOT_TOKEN"):
        settings_empty.require_telegram_bot_token()

    monkeypatch.setenv("GEMINI_API_KEY", "AIzaFakeKey12345678901234567890")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZ123456")
    settings_loaded = AppSettings(_env_file=None)  # type: ignore[call-arg]

    assert "AIzaFakeKey" not in repr(settings_loaded)
    assert "ABCDEFGHIJKLMNOPQRSTUVWXYZ" not in repr(settings_loaded)
    assert settings_loaded.require_gemini_api_key() == "AIzaFakeKey12345678901234567890"
    assert settings_loaded.require_telegram_bot_token().startswith("123456789:")
