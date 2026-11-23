"""Testes unitários do controle de acesso (modo público + 2 admins + restrito) e rate limiter."""

from __future__ import annotations

from pathlib import Path

import pytest

from rbac_assistant.infrastructure.audit.jsonl_audit_adapter import JsonlAuditAdapter
from rbac_assistant.security.access_control import AccessControl, OperatorRole
from rbac_assistant.security.rate_limiter import UserRateLimiter


@pytest.mark.unit
def test_access_control_public_mode_grants_user_to_unknown_and_admin_to_two_admins(
    tmp_path: Path,
) -> None:
    cfg = tmp_path / "access_control.yaml"
    cfg.write_text(
        "policy:\n"
        "  public_access: true\n"
        "  default_role: USER\n"
        "roles:\n"
        "  admins:\n"
        "    - 1001\n"
        "    - 1002\n",
        encoding="utf-8",
    )
    audit = JsonlAuditAdapter(logs_dir=tmp_path / "logs")
    ac = AccessControl.from_yaml(cfg, audit_port=audit)

    assert ac.public_access is True
    assert ac.default_role == OperatorRole.USER
    assert ac.deny_by_default is False
    assert ac.admin_count == 2

    # 1º administrador cadastrado recebe ADMIN
    d_admin_1 = ac.authorize(1001, require_admin=True, command="/status")
    assert d_admin_1.allowed is True
    assert d_admin_1.role == OperatorRole.ADMIN
    assert d_admin_1.reason == "authorized_admin"

    # 2º administrador cadastrado recebe ADMIN
    d_admin_2 = ac.authorize(1002, require_admin=True, command="/status")
    assert d_admin_2.allowed is True
    assert d_admin_2.role == OperatorRole.ADMIN
    assert d_admin_2.reason == "authorized_admin"

    # Usuário público sem cadastro NÃO é bloqueado e recebe USER
    d_public = ac.authorize(987654321, command="/start")
    assert d_public.allowed is True
    assert d_public.role == OperatorRole.USER
    assert d_public.reason == "public_access_user"

    # Usuário público NÃO consegue obter privilégios administrativos
    d_public_admin = ac.authorize(987654321, require_admin=True, command="/admin")
    assert d_public_admin.allowed is False
    assert d_public_admin.role == OperatorRole.USER
    assert d_public_admin.reason == "admin_role_required"
    assert "ADMIN" in d_public_admin.safe_message

    events = audit.read_events()
    assert len(events) == 4
    assert [e["outcome"] for e in events] == ["allowed", "allowed", "allowed", "denied"]


@pytest.mark.unit
def test_access_control_public_access_false_disables_public_access(tmp_path: Path) -> None:
    cfg = tmp_path / "restricted.yaml"
    cfg.write_text(
        "policy:\n"
        "  public_access: false\n"
        "  default_role: USER\n"
        "roles:\n"
        "  admins:\n"
        "    - 1001\n"
        "    - 1002\n"
        "  users:\n"
        "    - 2002\n",
        encoding="utf-8",
    )
    ac = AccessControl.from_yaml(cfg)

    assert ac.public_access is False
    assert ac.deny_by_default is True
    assert ac.admin_count == 2
    assert ac.authorize(1001, require_admin=True).allowed is True
    assert ac.authorize(1002, require_admin=True).allowed is True
    assert ac.authorize(2002).role == OperatorRole.USER
    assert ac.authorize(2002, require_admin=True).allowed is False

    # Usuário desconhecido é bloqueado quando public_access=false
    denied = ac.authorize(999999, command="/start")
    assert denied.allowed is False
    assert denied.reason == "user_not_in_allowlist"
    assert "Acesso negado" in denied.safe_message

    # Compatibilidade com formato legado deny_by_default: true sem public_access
    legacy_cfg = tmp_path / "legacy_restricted.yaml"
    legacy_cfg.write_text(
        "policy:\n  deny_by_default: true\nroles:\n  admins: [1001]\n  users: []\n",
        encoding="utf-8",
    )
    legacy_ac = AccessControl.from_yaml(legacy_cfg)
    assert legacy_ac.public_access is False
    assert legacy_ac.authorize(1001).role == OperatorRole.ADMIN
    assert legacy_ac.authorize(999999).allowed is False


@pytest.mark.unit
def test_access_control_missing_malformed_or_incompatible_file_never_grants_admin(
    tmp_path: Path,
) -> None:
    # 1. Arquivo ausente em modo público: nunca concede ADMIN (permite USER para /id e consultas)
    missing_public = AccessControl.from_yaml(
        tmp_path / "does_not_exist.yaml",
        default_public_access=True,
    )
    assert missing_public.config_loaded is False
    assert missing_public.config_issue == "missing_config_file"
    assert missing_public.admin_count == 0
    dec_pub = missing_public.authorize(123456, command="/id")
    assert dec_pub.allowed is True
    assert dec_pub.role == OperatorRole.USER
    dec_pub_admin = missing_public.authorize(123456, require_admin=True, command="/status")
    assert dec_pub_admin.allowed is False
    assert dec_pub_admin.reason == "admin_role_required"

    # 2. Arquivo ausente em modo restrito: bloqueia todos
    missing_restricted = AccessControl.from_yaml(
        tmp_path / "does_not_exist.yaml",
        default_public_access=False,
    )
    assert missing_restricted.admin_count == 0
    dec_res = missing_restricted.authorize(123456, command="/start")
    assert dec_res.allowed is False
    assert dec_res.reason == "missing_config_file"

    # 3. YAML malformado -> fail-closed e zero admins
    broken_cfg = tmp_path / "broken.yaml"
    broken_cfg.write_text(":::invalid: [yaml", encoding="utf-8")
    broken_ac = AccessControl.from_yaml(broken_cfg)
    assert broken_ac.public_access is False
    assert broken_ac.admin_count == 0
    assert broken_ac.authorize(123456).allowed is False
    assert broken_ac.authorize(123456, require_admin=True).allowed is False

    # 4. Raiz YAML não mapeada -> fail-closed e zero admins
    list_root_cfg = tmp_path / "list_root.yaml"
    list_root_cfg.write_text("- 1001\n- 1002\n", encoding="utf-8")
    list_ac = AccessControl.from_yaml(list_root_cfg)
    assert list_ac.public_access is False
    assert list_ac.admin_count == 0
    assert list_ac.authorize(1001, require_admin=True).allowed is False

    # 5. Tentativa incompatível de definir default_role: ADMIN é forçada para USER
    escalated_cfg = tmp_path / "escalated.yaml"
    escalated_cfg.write_text(
        "policy:\n"
        "  public_access: true\n"
        "  default_role: ADMIN\n"
        "roles:\n"
        "  admins: []\n",
        encoding="utf-8",
    )
    escalated_ac = AccessControl.from_yaml(escalated_cfg)
    assert escalated_ac.default_role == OperatorRole.USER
    assert escalated_ac.admin_count == 0
    assert escalated_ac.config_issue == "invalid_default_role_forced_to_user"
    assert escalated_ac.authorize(555555).role == OperatorRole.USER
    assert escalated_ac.authorize(555555, require_admin=True).allowed is False

    # 6. `public_access` com tipo inválido -> fail-closed e zero admins
    bad_type_cfg = tmp_path / "bad_type.yaml"
    bad_type_cfg.write_text(
        "policy:\n  public_access: 'yes'\nroles:\n  admins: []\n",
        encoding="utf-8",
    )
    bad_type_ac = AccessControl.from_yaml(bad_type_cfg)
    assert bad_type_ac.public_access is False
    assert bad_type_ac.authorize(555555).allowed is False


@pytest.mark.unit
@pytest.mark.parametrize("bad_id", [None, "", "abc", 0, -42, True, False, 3.14, "0", "-1"])
def test_access_control_rejects_invalid_user_ids(bad_id: object) -> None:
    ac = AccessControl(admins=[101, 102], public_access=True)
    decision = ac.authorize(bad_id, command="/rbac")
    assert decision.allowed is False
    assert decision.reason == "invalid_user_id"
    assert decision.user_id is None


@pytest.mark.unit
def test_access_control_ignores_invalid_yaml_entries_and_forbids_admin_default_role_ctor(
    tmp_path: Path,
) -> None:
    cfg = tmp_path / "mixed.yaml"
    cfg.write_text(
        "policy:\n  public_access: true\n  default_role: USER\n"
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
    assert ac.authorize(7007).role == OperatorRole.USER
    assert ac.authorize(7007, require_admin=True).allowed is False

    with pytest.raises(ValueError, match="default_role"):
        AccessControl(admins=[1], public_access=True, default_role="ADMIN")


@pytest.mark.unit
def test_user_rate_limiter_enforces_burst_minute_and_global_windows_deterministically() -> None:
    now = 100.0
    limiter = UserRateLimiter(
        requests_per_minute=4,
        burst_size=2,
        burst_window_seconds=10.0,
        window_seconds=60.0,
        global_requests_per_minute=5,
        clock=lambda: now,
    )

    # Rajada de 2 permitida para usuário 10
    assert limiter.check(10).allowed is True
    assert limiter.check(10).allowed is True
    # 3ª em <10s bloqueada pelo burst do usuário
    blocked_burst = limiter.check(10)
    assert blocked_burst.allowed is False
    assert blocked_burst.scope == "user"
    assert blocked_burst.retry_after_seconds > 0

    # Outros usuários têm cota individual independente até atingir teto global (5/min)
    assert limiter.check(20).allowed is True  # global: 3
    assert limiter.check(20).allowed is True  # global: 4
    assert limiter.check(30).allowed is True  # global: 5

    # Novo usuário 40 é bloqueado pelo limite GLOBAL (proteção de cota Free Tier)
    blocked_global = limiter.check(40)
    assert blocked_global.allowed is False
    assert blocked_global.scope == "global"
    assert blocked_global.retry_after_seconds > 0

    # Após >60s, janela global e individual são liberadas novamente
    now += 61.0
    assert limiter.check(40).allowed is True
    assert limiter.check(10).allowed is True
