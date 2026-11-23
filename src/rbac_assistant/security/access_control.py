"""Controle de acesso determinístico `deny_by_default` por Telegram User ID.

Regras de segurança:
- Política imutável `deny_by_default`: qualquer usuário ausente da allowlist é bloqueado.
- Ausência de `config/access_control.yaml`, arquivo inválido ou lista vazia bloqueiam
  100% dos usuários (nunca inventa administradores ou IDs padrão).
- Validação rigorosa de IDs inteiros positivos (rejeita `bool`, strings inválidas, `<= 0`).
- Registra eventos de bloqueio/permissão via `AuditPort` quando injetado.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import structlog
import yaml

from rbac_assistant.domain.exceptions import AuditWriteError
from rbac_assistant.domain.ports.audit_port import AuditEvent, AuditPort

logger = structlog.get_logger(__name__)

_SAFE_DENIAL_UNCONFIGURED = (
    "Acesso negado. Este assistente opera em modo restrito (deny-by-default) e não possui "
    "uma lista operacional de usuários autorizados configurada no servidor."
)
_SAFE_DENIAL_UNAUTHORIZED = (
    "Acesso negado. Seu Telegram User ID não está autorizado a utilizar o RBAC AI Assistant."
)
_SAFE_DENIAL_INVALID_ID = (
    "Acesso negado. Não foi possível identificar um Telegram User ID numérico válido."
)
_SAFE_DENIAL_ADMIN_REQUIRED = (
    "Acesso negado. Este comando é restrito a operadores com perfil ADMIN."
)


class OperatorRole(StrEnum):
    """Perfis operacionais determinísticos (terminologia separada de RBAC ANAC)."""

    ADMIN = "ADMIN"
    USER = "USER"


@dataclass(frozen=True, slots=True)
class AccessDecision:
    """Resultado imutável da avaliação determinística de controle de acesso."""

    allowed: bool
    user_id: int | None
    role: OperatorRole | None
    reason: str
    safe_message: str = ""


def _parse_valid_user_id(raw_value: object) -> int | None:
    """Valida que o valor representa um Telegram User ID inteiro estritamente positivo."""
    if isinstance(raw_value, bool):
        return None
    if isinstance(raw_value, int):
        return raw_value if raw_value > 0 else None
    if isinstance(raw_value, str):
        cleaned = raw_value.strip()
        if cleaned.isdigit():
            parsed = int(cleaned)
            return parsed if parsed > 0 else None
    return None


def _extract_id_set(raw_items: Any, *, field_name: str) -> tuple[frozenset[int], int]:
    """Extrai conjunto de IDs válidos ignorando entradas inválidas e contando rejeições."""
    if raw_items is None:
        return frozenset(), 0
    if not isinstance(raw_items, list):
        logger.warning("access_control_invalid_role_list", field=field_name)
        return frozenset(), 1

    valid: set[int] = set()
    invalid_count = 0
    for item in raw_items:
        parsed = _parse_valid_user_id(item)
        if parsed is None:
            invalid_count += 1
        else:
            valid.add(parsed)
    return frozenset(valid), invalid_count


class AccessControl:
    """Guarda determinístico `deny_by_default` para autorização de operadores Telegram."""

    def __init__(
        self,
        *,
        admins: Iterable[int] = (),
        users: Iterable[int] = (),
        deny_by_default: bool = True,
        config_loaded: bool = True,
        config_issue: str | None = None,
        audit_port: AuditPort | None = None,
    ) -> None:
        if not deny_by_default:
            raise ValueError(
                "A política de segurança exige `deny_by_default=True`; "
                "acesso aberto a usuários desconhecidos é proibido."
            )

        valid_admins: set[int] = set()
        for raw in admins:
            parsed = _parse_valid_user_id(raw)
            if parsed is None:
                raise ValueError(f"Telegram User ID inválido em admins: {raw!r}")
            valid_admins.add(parsed)

        valid_users: set[int] = set()
        for raw in users:
            parsed = _parse_valid_user_id(raw)
            if parsed is None:
                raise ValueError(f"Telegram User ID inválido em users: {raw!r}")
            valid_users.add(parsed)

        self._admins = frozenset(valid_admins)
        self._users = frozenset(valid_users - valid_admins)
        self._deny_by_default = True
        self._config_loaded = config_loaded
        self._config_issue = config_issue
        self._audit_port = audit_port

    @classmethod
    def from_yaml(
        cls,
        yaml_path: Path,
        *,
        audit_port: AuditPort | None = None,
    ) -> AccessControl:
        """Carrega allowlist de `config/access_control.yaml` em modo fail-closed.

        Se o arquivo estiver ausente, malformado ou vazio, retorna instância operacional
        que bloqueia 100% dos usuários (`deny_by_default`) sem inventar IDs.
        """
        if not yaml_path.is_file():
            logger.warning(
                "access_control_file_missing",
                path=str(yaml_path),
                policy="deny_all_fail_closed",
            )
            return cls(
                admins=(),
                users=(),
                deny_by_default=True,
                config_loaded=False,
                config_issue="missing_config_file",
                audit_port=audit_port,
            )

        try:
            raw_data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            logger.error(
                "access_control_yaml_load_failed",
                path=str(yaml_path),
                error_type=type(exc).__name__,
            )
            return cls(
                admins=(),
                users=(),
                deny_by_default=True,
                config_loaded=False,
                config_issue="invalid_yaml",
                audit_port=audit_port,
            )

        if not isinstance(raw_data, dict):
            logger.warning("access_control_invalid_root", path=str(yaml_path))
            return cls(
                admins=(),
                users=(),
                deny_by_default=True,
                config_loaded=False,
                config_issue="invalid_root_structure",
                audit_port=audit_port,
            )

        policy_section = raw_data.get("policy")
        if isinstance(policy_section, dict) and policy_section.get("deny_by_default") is False:
            logger.warning(
                "access_control_open_policy_rejected",
                path=str(yaml_path),
                enforced_policy="deny_by_default",
            )

        roles_section = raw_data.get("roles")
        if not isinstance(roles_section, dict):
            logger.warning("access_control_roles_missing", path=str(yaml_path))
            return cls(
                admins=(),
                users=(),
                deny_by_default=True,
                config_loaded=True,
                config_issue="empty_roles",
                audit_port=audit_port,
            )

        admins, bad_admins = _extract_id_set(roles_section.get("admins"), field_name="admins")
        users, bad_users = _extract_id_set(roles_section.get("users"), field_name="users")

        issue: str | None = None
        if not admins and not users:
            issue = "empty_allowlist"
            logger.warning("access_control_empty_allowlist", path=str(yaml_path))
        elif bad_admins or bad_users:
            issue = "partial_invalid_entries_ignored"
            logger.warning(
                "access_control_invalid_entries_ignored",
                invalid_admins=bad_admins,
                invalid_users=bad_users,
            )

        return cls(
            admins=admins,
            users=users,
            deny_by_default=True,
            config_loaded=True,
            config_issue=issue,
            audit_port=audit_port,
        )

    @property
    def deny_by_default(self) -> bool:
        return self._deny_by_default

    @property
    def config_loaded(self) -> bool:
        return self._config_loaded

    @property
    def config_issue(self) -> str | None:
        return self._config_issue

    @property
    def authorized_count(self) -> int:
        return len(self._admins) + len(self._users)

    @property
    def admin_count(self) -> int:
        return len(self._admins)

    @property
    def is_allowlist_configured(self) -> bool:
        return self.authorized_count > 0

    def _record_audit(self, *, decision: AccessDecision, command: str) -> None:
        if self._audit_port is None:
            return
        try:
            self._audit_port.record(
                AuditEvent(
                    timestamp=datetime.now(UTC),
                    action="telegram_access_control",
                    status="allowed" if decision.allowed else "denied",
                    actor_id=str(decision.user_id) if decision.user_id is not None else "invalid",
                    role=decision.role.value if decision.role is not None else None,
                    details={
                        "reason": decision.reason,
                        "command": command,
                        "allowlist_configured": self.is_allowlist_configured,
                    },
                )
            )
        except AuditWriteError as exc:
            logger.error("access_control_audit_write_failed", error=str(exc))

    def authorize(
        self,
        user_id: object,
        *,
        require_admin: bool = False,
        command: str = "unknown",
    ) -> AccessDecision:
        """Avalia deterministicamente se o usuário possui acesso autorizado."""
        parsed_id = _parse_valid_user_id(user_id)
        if parsed_id is None:
            decision = AccessDecision(
                allowed=False,
                user_id=None,
                role=None,
                reason="invalid_user_id",
                safe_message=_SAFE_DENIAL_INVALID_ID,
            )
            self._record_audit(decision=decision, command=command)
            return decision

        if not self.is_allowlist_configured:
            decision = AccessDecision(
                allowed=False,
                user_id=parsed_id,
                role=None,
                reason=self._config_issue or "empty_allowlist",
                safe_message=_SAFE_DENIAL_UNCONFIGURED,
            )
            self._record_audit(decision=decision, command=command)
            return decision

        role: OperatorRole | None = None
        if parsed_id in self._admins:
            role = OperatorRole.ADMIN
        elif parsed_id in self._users:
            role = OperatorRole.USER

        if role is None:
            decision = AccessDecision(
                allowed=False,
                user_id=parsed_id,
                role=None,
                reason="user_not_in_allowlist",
                safe_message=_SAFE_DENIAL_UNAUTHORIZED,
            )
            self._record_audit(decision=decision, command=command)
            return decision

        if require_admin and role != OperatorRole.ADMIN:
            decision = AccessDecision(
                allowed=False,
                user_id=parsed_id,
                role=role,
                reason="admin_role_required",
                safe_message=_SAFE_DENIAL_ADMIN_REQUIRED,
            )
            self._record_audit(decision=decision, command=command)
            return decision

        decision = AccessDecision(
            allowed=True,
            user_id=parsed_id,
            role=role,
            reason="authorized",
            safe_message="",
        )
        self._record_audit(decision=decision, command=command)
        return decision
