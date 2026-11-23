"""Adaptador de interface Telegram via Long Polling."""

from rbac_assistant.interfaces.telegram.bot_runner import (
    TelegramBotRunner,
    is_usable_telegram_token,
)
from rbac_assistant.interfaces.telegram.command_router import (
    SUPPORTED_COMMANDS,
    TelegramCommandRouter,
    TelegramResponse,
    TelegramRuntimeInfo,
)
from rbac_assistant.interfaces.telegram.message_splitter import (
    MessageSplitter,
    sanitize_telegram_text,
    strip_pagination_header,
    utf16_length,
)

__all__ = [
    "SUPPORTED_COMMANDS",
    "MessageSplitter",
    "TelegramBotRunner",
    "TelegramCommandRouter",
    "TelegramResponse",
    "TelegramRuntimeInfo",
    "is_usable_telegram_token",
    "sanitize_telegram_text",
    "strip_pagination_header",
    "utf16_length",
]
