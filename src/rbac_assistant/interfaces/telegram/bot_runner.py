"""Runner Long Polling do Bot Telegram (`python-telegram-bot` v22) com encerramento gracioso.

Características operacionais:
- Comunicação exclusivamente de saída via HTTPS Long Polling (ADR-03).
- Tratamento global de erros e indisponibilidade temporária da API Telegram com retry/backoff.
- Redação determinística de tokens em todos os logs operacionais e eventos de auditoria.
- Bloqueio automático de rede real em testes via `RBAC_BLOCK_REAL_TELEGRAM_API=1`.
- Encerramento gracioso ordenado (`updater.stop()` -> `app.stop()` -> `app.shutdown()`).
"""

from __future__ import annotations

import asyncio
import math
import os
import threading
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from typing import Any

import structlog
from telegram import Update
from telegram.error import (
    Conflict,
    Forbidden,
    InvalidToken,
    NetworkError,
    RetryAfter,
    TelegramError,
    TimedOut,
)
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    Defaults,
    MessageHandler,
    filters,
)

from rbac_assistant.domain.exceptions import AuditWriteError
from rbac_assistant.domain.ports.audit_port import AuditEvent, AuditPort
from rbac_assistant.interfaces.telegram.command_router import TelegramCommandRouter
from rbac_assistant.security.secret_redactor import redact_text

logger = structlog.get_logger(__name__)

_PLACEHOLDER_PREFIXES: tuple[str, ...] = (
    "replace_with_",
    "your_",
    "changeme",
    "placeholder",
)
_MAX_REPLY_RETRY_AFTER_SECONDS: float = 10.0
_MAX_STARTUP_RETRY_DELAY_SECONDS: float = 30.0


def _normalize_retry_after_seconds(
    retry_after: int | float | timedelta,
    *,
    max_wait_seconds: float = _MAX_REPLY_RETRY_AFTER_SECONDS,
) -> float:
    """Normaliza `RetryAfter.retry_after` (`int`, `float` ou `timedelta`) para segundos finitos.

    Regras de segurança e compatibilidade (`PTB_TIMEDELTA=1` e formato legado):
    - `datetime.timedelta` usa exclusivamente `.total_seconds()` (nunca `float(timedelta)`).
    - `int` e `float` são convertidos para `float`, rejeitando `bool`.
    - Rejeita tempos negativos, `NaN`, infinitos ou tipos inválidos com `ValueError`.
    - Aplica o limite superior `max_wait_seconds` sem ampliar limites de retry.
    """
    if isinstance(max_wait_seconds, bool) or not isinstance(max_wait_seconds, int | float):
        raise ValueError("`max_wait_seconds` deve ser um número finito e não negativo.")
    max_wait = float(max_wait_seconds)
    if not math.isfinite(max_wait) or max_wait < 0.0:
        raise ValueError("`max_wait_seconds` deve ser um número finito e não negativo.")

    if isinstance(retry_after, timedelta):
        seconds = retry_after.total_seconds()
    elif isinstance(retry_after, int | float) and not isinstance(retry_after, bool):
        seconds = float(retry_after)
    else:
        raise ValueError(
            f"Tipo inválido para retry_after: {type(retry_after).__name__}. "
            "Esperado int, float ou datetime.timedelta."
        )

    if not math.isfinite(seconds) or seconds < 0.0:
        raise ValueError(
            f"Tempo de espera retry_after inválido ({seconds!r}s): deve ser finito e >= 0."
        )

    return min(seconds, max_wait)


def is_usable_telegram_token(token: str | None) -> bool:
    """Verifica se o token Telegram está presente e não é placeholder de template."""
    if not token:
        return False
    cleaned = token.strip()
    if not cleaned or ":" not in cleaned:
        return False
    lowered = cleaned.lower()
    return not any(lowered.startswith(prefix) for prefix in _PLACEHOLDER_PREFIXES)


def _real_telegram_api_blocked() -> bool:
    return os.environ.get("RBAC_BLOCK_REAL_TELEGRAM_API", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


class TelegramBotRunner:
    """Gerencia o ciclo de vida Long Polling do `python-telegram-bot` com resiliência."""

    def __init__(
        self,
        *,
        token: str,
        router: TelegramCommandRouter,
        polling_timeout_seconds: int = 30,
        startup_retry_max_attempts: int = 5,
        startup_retry_base_delay_seconds: float = 2.0,
        allow_group_chats: bool = False,
        audit_port: AuditPort | None = None,
        application: Any | None = None,
    ) -> None:
        if not token.strip():
            raise ValueError("Token Telegram não pode ser vazio.")
        self._token = token.strip()
        self._router = router
        self._polling_timeout_seconds = polling_timeout_seconds
        self._startup_retry_max_attempts = startup_retry_max_attempts
        self._startup_retry_base_delay_seconds = startup_retry_base_delay_seconds
        self._allow_group_chats = allow_group_chats
        self._audit_port = audit_port
        self._injected_application = application

    def _audit(
        self,
        *,
        action: str,
        status: str,
        details: dict[str, str | int | float | bool],
    ) -> None:
        if self._audit_port is None:
            return
        try:
            self._audit_port.record(
                AuditEvent(
                    timestamp=datetime.now(UTC),
                    action=action,
                    status=status,
                    actor_id="telegram_bot",
                    details=details,
                )
            )
        except AuditWriteError as exc:
            logger.error("telegram_runner_audit_failed", error=redact_text(str(exc)))

    async def handle_update(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Handler unificado para comandos, comandos desconhecidos e texto livre."""
        del context
        chat = getattr(update, "effective_chat", None)
        chat_type = getattr(chat, "type", None) if chat is not None else None
        if (
            not self._allow_group_chats
            and chat_type is not None
            and str(chat_type).strip().lower() != "private"
        ):
            logger.info("telegram_non_private_chat_ignored", chat_type=str(chat_type))
            return

        message = update.effective_message
        if message is None:
            return
        user = update.effective_user
        user_id = user.id if user is not None else None

        response = await self._router.handle_message(
            user_id=user_id,
            text=message.text,
            update_id=update.update_id,
        )
        if response.duplicate_ignored:
            return

        for part in response.messages:
            if not part:
                continue
            await self._reply_with_resilience(message=message, text=part)

    async def _reply_with_resilience(self, *, message: Any, text: str) -> None:
        """Envia resposta tratando erros transitórios da API Telegram (RetryAfter/TimedOut)."""
        try:
            await message.reply_text(text=text, disable_web_page_preview=True)
        except RetryAfter as exc:
            try:
                wait_seconds = _normalize_retry_after_seconds(
                    exc.retry_after,
                    max_wait_seconds=_MAX_REPLY_RETRY_AFTER_SECONDS,
                )
            except ValueError as norm_exc:
                logger.warning(
                    "telegram_invalid_retry_after",
                    error=redact_text(str(norm_exc)),
                )
                self._audit(
                    action="telegram_api_error",
                    status="invalid_retry_after",
                    details={"error_type": type(exc).__name__},
                )
                return

            logger.warning("telegram_retry_after", retry_after_seconds=wait_seconds)
            await asyncio.sleep(wait_seconds)
            try:
                await message.reply_text(text=text, disable_web_page_preview=True)
            except Forbidden as retry_exc:
                logger.warning("telegram_reply_forbidden", error=redact_text(str(retry_exc)))
            except (TimedOut, NetworkError, RetryAfter) as retry_exc:
                logger.warning(
                    "telegram_reply_network_error",
                    error_type=type(retry_exc).__name__,
                    error=redact_text(str(retry_exc)),
                    after_retry=True,
                )
                self._audit(
                    action="telegram_api_error",
                    status="network_error",
                    details={"error_type": type(retry_exc).__name__, "after_retry": True},
                )
            except TelegramError as retry_exc:
                logger.error(
                    "telegram_reply_api_error",
                    error_type=type(retry_exc).__name__,
                    error=redact_text(str(retry_exc)),
                    after_retry=True,
                )
                self._audit(
                    action="telegram_api_error",
                    status="telegram_error",
                    details={"error_type": type(retry_exc).__name__, "after_retry": True},
                )
        except Forbidden as exc:
            logger.warning("telegram_reply_forbidden", error=redact_text(str(exc)))
        except (TimedOut, NetworkError) as exc:
            logger.warning(
                "telegram_reply_network_error",
                error_type=type(exc).__name__,
                error=redact_text(str(exc)),
            )
            self._audit(
                action="telegram_api_error",
                status="network_error",
                details={"error_type": type(exc).__name__},
            )
        except TelegramError as exc:
            logger.error(
                "telegram_reply_api_error",
                error_type=type(exc).__name__,
                error=redact_text(str(exc)),
            )
            self._audit(
                action="telegram_api_error",
                status="telegram_error",
                details={"error_type": type(exc).__name__},
            )

    async def handle_error(
        self,
        update: object,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Tratador global de exceções do `python-telegram-bot` com redação garantida."""
        exc = context.error
        err_type = type(exc).__name__ if exc is not None else "UnknownError"
        safe_err = redact_text(str(exc)) if exc is not None else "erro desconhecido"

        if isinstance(exc, Conflict):
            logger.error(
                "telegram_polling_conflict",
                error_type=err_type,
                message="Outra instância do bot está usando o mesmo token (409 Conflict).",
            )
            self._audit(
                action="telegram_api_error",
                status="conflict_409",
                details={"error_type": err_type},
            )
            return

        if isinstance(exc, (NetworkError, TimedOut, RetryAfter)):
            logger.warning("telegram_transient_api_error", error_type=err_type, error=safe_err)
            self._audit(
                action="telegram_api_error",
                status="transient_error",
                details={"error_type": err_type},
            )
            return

        logger.error("telegram_unhandled_error", error_type=err_type, error=safe_err)
        self._audit(
            action="telegram_api_error",
            status="unhandled_error",
            details={"error_type": err_type},
        )
        if isinstance(update, Update) and update.effective_message is not None:
            with suppress(TelegramError):
                await update.effective_message.reply_text(
                    "Ocorreu uma falha temporária na comunicação com o serviço. "
                    "Tente novamente em instantes."
                )

    def build_application(self) -> Any:
        """Constrói e registra todos os handlers na `Application` do `python-telegram-bot`."""
        if self._injected_application is not None:
            app = self._injected_application
        else:
            if _real_telegram_api_blocked():
                raise RuntimeError(
                    "Conexão real à API do Telegram bloqueada durante testes "
                    "(RBAC_BLOCK_REAL_TELEGRAM_API=1). Injete uma `application` simulada."
                )
            app = (
                ApplicationBuilder()
                .token(self._token)
                .defaults(Defaults(parse_mode=None))
                .concurrent_updates(True)
                .build()
            )

        chat_filter = filters.ALL if self._allow_group_chats else filters.ChatType.PRIVATE

        for cmd in (
            "start",
            "help",
            "ajuda",
            "id",
            "status",
            "rbac",
            "buscar",
            "explicar",
            "documentos",
            "versao",
        ):
            app.add_handler(CommandHandler(cmd, self.handle_update, filters=chat_filter))

        # Captura comandos não reconhecidos — também protegidos por AccessControl e escopo privado!
        app.add_handler(MessageHandler(chat_filter & filters.COMMAND, self.handle_update))
        # Captura mensagens de texto livre — protegidas por AccessControl e escopo privado!
        app.add_handler(
            MessageHandler(chat_filter & filters.TEXT & ~filters.COMMAND, self.handle_update)
        )
        app.add_error_handler(self.handle_error)
        return app

    async def run_async(self, *, stop_event: threading.Event | asyncio.Event) -> int:
        """Executa Long Polling com retry na inicialização e encerramento gracioso."""
        app: Application[Any, Any, Any, Any, Any, Any] = self.build_application()
        initialized = False
        started = False
        polling = False

        try:
            for attempt in range(1, self._startup_retry_max_attempts + 1):
                if stop_event.is_set():
                    return 0
                try:
                    await app.initialize()
                    initialized = True
                    await app.start()
                    started = True
                    if app.updater is None:
                        raise RuntimeError("Updater de Long Polling indisponível na Application.")
                    await app.updater.start_polling(
                        timeout=self._polling_timeout_seconds,
                        drop_pending_updates=False,
                        allowed_updates=Update.ALL_TYPES,
                    )
                    polling = True
                    logger.info(
                        "telegram_long_polling_started",
                        timeout_seconds=self._polling_timeout_seconds,
                    )
                    self._audit(
                        action="telegram_polling",
                        status="started",
                        details={"timeout_seconds": self._polling_timeout_seconds},
                    )
                    break
                except InvalidToken:
                    logger.error("telegram_invalid_bot_token")
                    self._audit(
                        action="telegram_polling",
                        status="invalid_token",
                        details={"attempt": attempt},
                    )
                    return 1
                except (NetworkError, TimedOut, RetryAfter) as exc:
                    safe_err = redact_text(str(exc))
                    logger.warning(
                        "telegram_startup_network_retry",
                        attempt=attempt,
                        max_attempts=self._startup_retry_max_attempts,
                        error=safe_err,
                    )
                    self._audit(
                        action="telegram_polling",
                        status="startup_retry",
                        details={"attempt": attempt, "error_type": type(exc).__name__},
                    )
                    if attempt >= self._startup_retry_max_attempts:
                        logger.error("telegram_startup_exhausted_retries")
                        return 1
                    base_delay = max(0.0, float(self._startup_retry_base_delay_seconds))
                    backoff_delay = min(
                        base_delay * (2 ** (attempt - 1)),
                        _MAX_STARTUP_RETRY_DELAY_SECONDS,
                    )
                    if isinstance(exc, RetryAfter):
                        try:
                            retry_delay = _normalize_retry_after_seconds(
                                exc.retry_after,
                                max_wait_seconds=_MAX_STARTUP_RETRY_DELAY_SECONDS,
                            )
                        except ValueError as norm_exc:
                            logger.warning(
                                "telegram_invalid_retry_after",
                                error=redact_text(str(norm_exc)),
                            )
                            retry_delay = backoff_delay
                        delay = max(backoff_delay, retry_delay)
                    else:
                        delay = backoff_delay
                    await asyncio.sleep(delay)

            # Aguarda sinal externo de encerramento (SIGTERM/SIGINT)
            while not stop_event.is_set():
                await asyncio.sleep(0.2)
            return 0
        finally:
            if polling and app.updater is not None:
                try:
                    await app.updater.stop()
                except Exception as exc:
                    logger.warning("telegram_updater_stop_error", error=redact_text(str(exc)))
            if started:
                try:
                    await app.stop()
                except Exception as exc:
                    logger.warning("telegram_app_stop_error", error=redact_text(str(exc)))
            if initialized:
                try:
                    await app.shutdown()
                except Exception as exc:
                    logger.warning("telegram_app_shutdown_error", error=redact_text(str(exc)))
            self._audit(
                action="telegram_polling",
                status="stopped",
                details={"clean_shutdown": True},
            )
            logger.info("telegram_long_polling_stopped")

    def run(self, *, stop_event: threading.Event) -> int:
        """Entrypoint síncrono que executa o loop asyncio do bot com encerramento gracioso."""
        return asyncio.run(self.run_async(stop_event=stop_event))
