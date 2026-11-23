"""Controle de acesso determinístico (modo público configurável + isolamento estrito ADMIN).

Regras de segurança:
- Com `public_access: true` e `default_role: USER`, qualquer usuário do Telegram com
  User ID numérico válido recebe automaticamente o perfil `USER`, sem necessidade de
  cadastro ou whitelist prévia.
- Apenas IDs explicitamente cadastrados em `roles.admins` recebem perfil `ADMIN`.
- Usuários públicos JAMAIS recebem privilégios `ADMIN` — nem por ausência de arquivo,
  nem por arquivo malformado/incompatível, nem por tentativa de definir `default_role: ADMIN`.
- Com `public_access: false`, o acesso público é explicitamente desabilitado (modo restrito
  por lista de permissões `roles.admins` / `roles.users`, com bloqueio padrão de desconhecidos).
- Validação rigorosa de IDs inteiros positivos (rejeita `bool`, strings inválidas, `<= 0`).
- Registra eventos via `AuditPort` quando injetado, sem expor IDs em logs operacionais públicos.
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
    "Acesso negado. O modo de acesso restrito está ativo e não possui uma lista operacional "
    "de usuários autorizados configurada no servidor."
)
_SAFE_DENIAL_UNAUTHORIZED = (
    "Acesso negado. O acesso público está desabilitado e seu Telegram User ID não está "
    "autorizado a utilizar o RBAC AI Assistant."
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


def _validate_default_role(raw_role: OperatorRole | str) -> OperatorRole:
    """Garante deterministicamente que o papel padrão público seja exclusivamente `USER`."""
    normalized = str(raw_role).strip().upper()
    if normalized != OperatorRole.USER.value:
        raise ValueError(
            "O campo `default_role` só pode ser `USER`; "
            "privilégios ADMIN nunca podem ser concedidos por padrão."
        )
    return OperatorRole.USER


class AccessControl:
    """Guarda determinístico para acesso público (`USER`) e isolamento estrito `ADMIN`."""

    def __init__(
        self,
        *,
        admins: Iterable[int] = (),
        users: Iterable[int] = (),
        public_access: bool = True,
        default_role: OperatorRole | str = OperatorRole.USER,
        deny_by_default: bool | None = None,
        config_loaded: bool = True,
        config_issue: str | None = None,
        audit_port: AuditPort | None = None,
    ) -> None:
        validated_default_role = _validate_default_role(default_role)
        effective_public_access = (
            (not deny_by_default) if deny_by_default is not None else bool(public_access)
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
        self._public_access = effective_public_access
        self._default_role = validated_default_role
        self._config_loaded = config_loaded
        self._config_issue = config_issue
        self._audit_port = audit_port

    @classmethod
    def from_yaml(
        cls,
        yaml_path: Path,
        *,
        default_public_access: bool = True,
        audit_port: AuditPort | None = None,
    ) -> AccessControl:
        """Carrega política operacional de `config/access_control.yaml` com isolamento ADMIN.

        Regras de segurança no carregamento:
        - Arquivo ausente: nunca inventa administradores (`admins=()`). Mantém `public_access`
          conforme `default_public_access` (`USER` sem privilégio administrativo).
        - Arquivo malformado ou raiz inválida: opera em modo fail-closed (`public_access=False`,
          `admins=()`, `users=()`) — nunca concede `ADMIN` nem abre acesso acidental.
        - `default_role` diferente de `USER` no YAML: rejeitado e forçado para `USER`.
        - `policy.public_access: false` (ou legado `deny_by_default: true` sem `public_access`):
          desabilita explicitamente o acesso público.
        """
        if not yaml_path.is_file():
            logger.warning(
                "access_control_file_missing",
                policy="public_user_no_admins" if default_public_access else "restricted_no_admins",
            )
            return cls(
                admins=(),
                users=(),
                public_access=default_public_access,
                default_role=OperatorRole.USER,
                config_loaded=False,
                config_issue="missing_config_file",
                audit_port=audit_port,
            )

        try:
            raw_data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            logger.error(
                "access_control_yaml_load_failed",
                error_type=type(exc).__name__,
            )
            return cls(
                admins=(),
                users=(),
                public_access=False,
                default_role=OperatorRole.USER,
                config_loaded=False,
                config_issue="invalid_yaml",
                audit_port=audit_port,
            )

        if not isinstance(raw_data, dict):
            logger.warning("access_control_invalid_root")
            return cls(
                admins=(),
                users=(),
                public_access=False,
                default_role=OperatorRole.USER,
                config_loaded=False,
                config_issue="invalid_root_structure",
                audit_port=audit_port,
            )

        issues: list[str] = []
        policy_section = raw_data.get("policy")
        public_access = default_public_access

        if policy_section is not None and not isinstance(policy_section, dict):
            logger.warning("access_control_invalid_policy_section")
            public_access = False
            issues.append("invalid_policy_section")
        elif isinstance(policy_section, dict):
            if "public_access" in policy_section:
                raw_public = policy_section["public_access"]
                if isinstance(raw_public, bool):
                    public_access = raw_public
                else:
                    logger.warning("access_control_invalid_public_access_type")
                    public_access = False
                    issues.append("invalid_policy_public_access")
            elif policy_section.get("deny_by_default") is True:
                public_access = False

            if "default_role" in policy_section:
                raw_default_role = policy_section["default_role"]
                if (
                    not isinstance(raw_default_role, str)
                    or raw_default_role.strip().upper() != OperatorRole.USER.value
                ):
                    logger.warning("access_control_invalid_default_role_rejected")
                    issues.append("invalid_default_role_forced_to_user")

        roles_section = raw_data.get("roles")
        admins: frozenset[int] = frozenset()
        users: frozenset[int] = frozenset()
        bad_admins = 0
        bad_users = 0

        if roles_section is None:
            issues.append("empty_roles")
        elif not isinstance(roles_section, dict):
            logger.warning("access_control_roles_invalid")
            issues.append("invalid_roles_section")
        else:
            admins, bad_admins = _extract_id_set(roles_section.get("admins"), field_name="admins")
            users, bad_users = _extract_id_set(roles_section.get("users"), field_name="users")
            if bad_admins or bad_users:
                issues.append("partial_invalid_entries_ignored")
                logger.warning(
                    "access_control_invalid_entries_ignored",
                    invalid_admins=bad_admins,
                    invalid_users=bad_users,
                )

        if not public_access and not admins and not users and not issues:
            issues.append("empty_allowlist")
            logger.warning("access_control_empty_allowlist")

        return cls(
            admins=admins,
            users=users,
            public_access=public_access,
            default_role=OperatorRole.USER,
            config_loaded=True,
            config_issue=issues[0] if issues else None,
            audit_port=audit_port,
        )

    @property
    def public_access(self) -> bool:
        """Indica se o acesso público automático com perfil `USER` está habilitado."""
        return self._public_access

    @property
    def default_role(self) -> OperatorRole:
        """Perfil atribuído a usuários públicos (sempre `USER`)."""
        return self._default_role

    @property
    def deny_by_default(self) -> bool:
        """Compatibilidade: verdadeiro somente quando `public_access` está desabilitado."""
        return not self._public_access

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
                        "public_access": self._public_access,
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
        """Avalia deterministicamente perfil e autorização de um usuário do Telegram."""
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

        role: OperatorRole | None = None
        grant_reason = "authorized"

        if parsed_id in self._admins:
            role = OperatorRole.ADMIN
            grant_reason = "authorized_admin"
        elif parsed_id in self._users:
            role = OperatorRole.USER
            grant_reason = "authorized_user"
        elif self._public_access:
            role = self._default_role
            grant_reason = "public_access_user"

        if role is None:
            if not self.is_allowlist_configured:
                decision = AccessDecision(
                    allowed=False,
                    user_id=parsed_id,
                    role=None,
                    reason=self._config_issue or "empty_allowlist",
                    safe_message=_SAFE_DENIAL_UNCONFIGURED,
                )
            else:
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
            reason=grant_reason,
            safe_message="",
        )
        self._record_audit(decision=decision, command=command)
        return decision
