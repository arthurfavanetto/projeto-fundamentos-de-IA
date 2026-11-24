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
    assert settings.retrieval.max_vector_distance == 0.85
    assert settings.retrieval.allow_preliminary_native_search is False
    assert settings.retrieval.telegram_only_approved_literal is True
    assert settings.telegram.allow_group_chats is False
    assert settings.telegram.max_global_concurrent_updates == 8
    assert settings.telegram.command_timeout_seconds == 45.0
    assert settings.security.public_access is True
    assert settings.security.default_role == "USER"
    assert settings.security.deny_by_default is False
    assert settings.security.rate_limit.global_requests_per_minute == 60
    assert settings.security.audit.pseudonymize_user_ids is True
    assert settings.gemini_generation_model == "gemini-4-hydrogen"
    assert settings.gemini_embedding_model == "gemini-embedding-2"
    assert settings.effective_embedding_dimensions == 768
    assert settings.gemini_allow_paid_models is False
    assert settings.gemini.allowed_free_tier_generation_models == ("gemini-4-hydrogen",)
    assert settings.gemini.allowed_free_tier_embedding_models == ("gemini-embedding-2",)


@pytest.mark.unit
def test_security_config_forbids_admin_default_role() -> None:
    from rbac_assistant.config.settings import SecurityConfig

    with pytest.raises(ValidationError):
        SecurityConfig(public_access=True, default_role="ADMIN")  # type: ignore[arg-type]


@pytest.mark.unit
def test_env_overrides_canonical_models_and_chroma_endpoint(
    repo_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CHROMA_HOST", "internal-chroma")
    monkeypatch.setenv("CHROMA_PORT", "8100")
    monkeypatch.setenv("GEMINI_GENERATION_MODEL", "gemini-4-hydrogen")
    monkeypatch.setenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2")
    monkeypatch.setenv("GEMINI_EMBEDDING_DIMENSIONS", "768")
    monkeypatch.setenv("GEMINI_ALLOW_PAID_MODELS", "false")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")

    settings = load_settings(yaml_path=repo_root / "config" / "settings.yaml")

    assert settings.chroma_base_url == "http://internal-chroma:8100"
    assert settings.gemini_generation_model == "gemini-4-hydrogen"
    assert settings.gemini_embedding_model == "gemini-embedding-2"
    assert settings.effective_embedding_dimensions == 768
    assert settings.gemini_allow_paid_models is False
    assert settings.log_level == "DEBUG"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("env_var", "bad_value", "error_pattern"),
    [
        ("GEMINI_GENERATION_MODEL", "gemini-3.5-flash-lite", "política única"),
        ("GEMINI_GENERATION_MODEL", "gemini-3.8-flash", "política única"),
        ("GEMINI_GENERATION_MODEL", "gemini-4-neon", "política única"),
        ("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001", "política única"),
        ("GEMINI_EMBEDDING_DIMENSIONS", "256", "768"),
        ("GEMINI_EMBEDDING_DIMENSIONS", "1536", "768"),
        ("GEMINI_ALLOW_PAID_MODELS", "true", "GEMINI_ALLOW_PAID_MODELS=true"),
    ],
)
def test_single_gemini_policy_rejects_unauthorized_env_values(
    monkeypatch: pytest.MonkeyPatch,
    env_var: str,
    bad_value: str,
    error_pattern: str,
) -> None:
    for var in (
        "GEMINI_GENERATION_MODEL",
        "GEMINI_CHAT_MODEL",
        "GEMINI_EMBEDDING_MODEL",
        "GEMINI_EMBEDDING_DIMENSIONS",
        "GEMINI_ALLOW_PAID_MODELS",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv(env_var, bad_value)

    with pytest.raises(ValidationError, match=error_pattern):
        AppSettings(_env_file=None)  # type: ignore[call-arg]


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


@pytest.mark.unit
def test_dotenv_isolation_and_real_gemini_block_hold_in_process_and_subprocesses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    repo_root: Path,
) -> None:
    """Comprova que testes e subprocessos filhos nunca leem `.env` nem acionam Gemini real."""
    import os
    import subprocess
    import sys

    from google import genai as google_genai

    from rbac_assistant.infrastructure.llm.gemini_client import GeminiClientAdapter

    # Cria `.env` fictício no diretório temporário simulando o `.env` local de um host Windows
    fake_key = "AIzaFakeDotEnvKeyForSubprocessRegressionTest999"
    (tmp_path / ".env").write_text(f"GEMINI_API_KEY={fake_key}\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    # 1. No processo de teste, AppSettings() e load_settings() ignoram `.env` em disco
    assert AppSettings().gemini_api_key is None
    assert load_settings(yaml_path=repo_root / "config" / "settings.yaml").gemini_api_key is None

    # 2. Mesmo com chave fictícia fornecida manualmente, chamadas reais à Gemini são bloqueadas
    adapter = GeminiClientAdapter(api_key=fake_key)
    with pytest.raises(RuntimeError, match="RBAC_BLOCK_REAL_GEMINI_API"):
        adapter.embed_documents(["teste de bloqueio sem sdk_client"])
    with pytest.raises(RuntimeError, match="RBAC_BLOCK_REAL_GEMINI_API"):
        adapter.embed_query("consulta de bloqueio sem sdk_client")
    with pytest.raises(RuntimeError, match="RBAC_BLOCK_REAL_GEMINI_API"):
        adapter.generate_text(system_instruction="sys", user_prompt="usr")
    with pytest.raises(RuntimeError, match="RBAC_BLOCK_REAL_GEMINI_API"):
        google_genai.Client(api_key=fake_key)

    # 3. Subprocesso filho herdando o ambiente do pytest TAMBÉM ignora `.env` e bloqueia SDK real
    child_env = dict(os.environ)
    child_env["PYTHONPATH"] = os.pathsep.join([str(repo_root / "src"), str(repo_root)])
    sub_check = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from rbac_assistant.config.settings import AppSettings; "
                "from rbac_assistant.infrastructure.llm.gemini_client import GeminiClientAdapter; "
                "assert AppSettings().gemini_api_key is None; "
                "ad = GeminiClientAdapter(api_key='AIzaFakeSubprocessOnlyKey'); "
                "ok = False\n"
                "try:\n"
                "    ad.embed_query('consulta bloqueada')\n"
                "except RuntimeError as exc:\n"
                "    ok = 'RBAC_BLOCK_REAL_GEMINI_API' in str(exc)\n"
                "assert ok, 'Gemini real deveria estar bloqueado no subprocesso'"
            ),
        ],
        cwd=tmp_path,
        env=child_env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert sub_check.returncode == 0, sub_check.stderr

    # 4. Subprocesso da CLI `scripts/evaluate_rag.py` no dir com `.env` falha rápido sem API real
    eval_sub = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "evaluate_rag.py"),
            "--embedding-backend",
            "gemini-embedding-2",
        ],
        cwd=tmp_path,
        env=child_env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert eval_sub.returncode != 0
    assert "GEMINI_API_KEY" in eval_sub.stderr

    # 5. Subprocesso de produção (sem RBAC_DISABLE_DOTENV) continua lendo `.env` normalmente
    prod_env = dict(child_env)
    prod_env.pop("RBAC_DISABLE_DOTENV", None)
    prod_env.pop("GEMINI_API_KEY", None)
    prod_check = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from rbac_assistant.config.settings import AppSettings; "
                "s = AppSettings(); "
                f"assert s.require_gemini_api_key() == {fake_key!r}"
            ),
        ],
        cwd=tmp_path,
        env=prod_env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert prod_check.returncode == 0, prod_check.stderr

