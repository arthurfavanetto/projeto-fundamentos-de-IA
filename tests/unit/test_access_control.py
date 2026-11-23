"""Testes unitários do controle de acesso `deny_by_default` e rate limiter Telegram."""

from __future__ import annotations

from pathlib import Path

import pytest

from rbac_assistant.infrastructure.audit.jsonl_audit_adapter import JsonlAuditAdapter
from rbac_assistant.security.access_control import AccessControl, OperatorRole
from rbac_assistant.security.rate_limiter import UserRateLimiter


@pytest.mark.unit
def test_access_control_authorizes_admin_and_user_and_denies_unknown(tmp_path: Path) -> None:
    cfg = tmp_path / "access_control.yaml"
    cfg.write_text(
        "policy:\n"
        "  deny_by_default: true\n"
        "roles:\n"
        "  admins:\n"
        "    - 1001\n"
        "  users:\n"
        "    - 2002\n"
        "    - 1001\n",
        encoding="utf-8",
    )
    audit = JsonlAuditAdapter(logs_dir=tmp_path / "logs")
    ac = AccessControl.from_yaml(cfg, audit_port=audit)

    assert ac.deny_by_default is True
    assert ac.is_allowlist_configured is True
    assert ac.admin_count == 1
    assert ac.authorized_count == 2

    d_admin = ac.authorize(1001, require_admin=True, command="/status")
    assert d_admin.allowed is True
    assert d_admin.role == OperatorRole.ADMIN

    d_user = ac.authorize(2002, command="/rbac")
    assert d_user.allowed is True
    assert d_user.role == OperatorRole.USER

    d_user_admin = ac.authorize(2002, require_admin=True, command="/admin")
    assert d_user_admin.allowed is False
    assert d_user_admin.reason == "admin_role_required"

    d_unknown = ac.authorize(9999, command="/start")
    assert d_unknown.allowed is False
    assert d_unknown.reason == "user_not_in_allowlist"
    assert "Acesso negado" in d_unknown.safe_message

    events = audit.read_events()
    assert len(events) == 4
    assert [e["outcome"] for e in events] == ["allowed", "allowed", "denied", "denied"]


@pytest.mark.unit
def test_access_control_fail_closed_when_file_missing_or_empty_or_malformed(tmp_path: Path) -> None:
    # 1. Arquivo ausente -> bloqueia todos sem inventar IDs
    missing_ac = AccessControl.from_yaml(tmp_path / "does_not_exist.yaml")
    assert missing_ac.is_allowlist_configured is False
    assert missing_ac.authorized_count == 0
    dec_missing = missing_ac.authorize(123456, command="/start")
    assert dec_missing.allowed is False
    assert dec_missing.reason == "missing_config_file"
    assert "deny-by-default" in dec_missing.safe_message

    # 2. Lista vazia -> bloqueia todos
    empty_cfg = tmp_path / "empty.yaml"
    empty_cfg.write_text(
        "policy:\n  deny_by_default: true\nroles:\n  admins: []\n  users: []\n",
        encoding="utf-8",
    )
    empty_ac = AccessControl.from_yaml(empty_cfg)
    assert empty_ac.is_allowlist_configured is False
    dec_empty = empty_ac.authorize(123456, command="text")
    assert dec_empty.allowed is False
    assert dec_empty.reason == "empty_allowlist"

    # 3. YAML malformado -> fail-closed
    broken_cfg = tmp_path / "broken.yaml"
    broken_cfg.write_text(":::invalid: [yaml", encoding="utf-8")
    broken_ac = AccessControl.from_yaml(broken_cfg)
    assert broken_ac.is_allowlist_configured is False
    assert broken_ac.authorize(123456).allowed is False

    # 4. Tentativa de desabilitar deny_by_default no YAML continua bloqueada
    open_cfg = tmp_path / "open.yaml"
    open_cfg.write_text(
        "policy:\n  deny_by_default: false\nroles:\n  admins: []\n  users: [555]\n",
        encoding="utf-8",
    )
    open_ac = AccessControl.from_yaml(open_cfg)
    assert open_ac.deny_by_default is True
    assert open_ac.authorize(555).allowed is True
    assert open_ac.authorize(777).allowed is False


@pytest.mark.unit
@pytest.mark.parametrize("bad_id", [None, "", "abc", 0, -42, True, False, 3.14, "0", "-1"])
def test_access_control_rejects_invalid_user_ids(bad_id: object) -> None:
    ac = AccessControl(admins=[101], users=[202])
    decision = ac.authorize(bad_id, command="/rbac")
    assert decision.allowed is False
    assert decision.reason == "invalid_user_id"
    assert decision.user_id is None


@pytest.mark.unit
def test_access_control_ignores_invalid_yaml_entries_and_forbids_open_constructor(
    tmp_path: Path,
) -> None:
    cfg = tmp_path / "mixed.yaml"
    cfg.write_text(
        "policy:\n  deny_by_default: true\n"
        "roles:\n"
        "  admins:\n"
        "    - 'not_a_number'\n"
        "    - -10\n"
        "    - true\n"
        "    - 5005\n"
        "  users:\n"
        "    - null\n"
        "    - 6006\n",
        encoding="utf-8",
    )
    ac = AccessControl.from_yaml(cfg)
    assert ac.config_issue == "partial_invalid_entries_ignored"
    assert ac.authorize(5005).role == OperatorRole.ADMIN
    assert ac.authorize(6006).role == OperatorRole.USER
    assert ac.authorize(1).allowed is False

    with pytest.raises(ValueError, match="deny_by_default=True"):
        AccessControl(admins=[1], deny_by_default=False)


@pytest.mark.unit
def test_user_rate_limiter_enforces_burst_and_minute_windows_deterministically() -> None:
    now = 100.0
    limiter = UserRateLimiter(
        requests_per_minute=4,
        burst_size=2,
        burst_window_seconds=10.0,
        window_seconds=60.0,
        clock=lambda: now,
    )

    # Rajada de 2 permitida
    assert limiter.check(10).allowed is True
    assert limiter.check(10).allowed is True
    # 3ª em <10s bloqueada pelo burst
    blocked_burst = limiter.check(10)
    assert blocked_burst.allowed is False
    assert blocked_burst.retry_after_seconds > 0

    # Outro usuário tem cota independente
    assert limiter.check(20).allowed is True

    # Após 11s, rajada libera mais 2 (total 4 no minuto)
    now += 11.0
    assert limiter.check(10).allowed is True
    assert limiter.check(10).allowed is True

    # Após mais 11s, burst estaria livre, mas janela de 60s atingiu 4/min
    now += 11.0
    blocked_rpm = limiter.check(10)
    assert blocked_rpm.allowed is False
    assert blocked_rpm.remaining_in_window == 0

    # Após >60s desde as primeiras requisições, janela libera novamente
    now += 45.0
    assert limiter.check(10).allowed is True
