"""Composition Root — bootstrap hexagonal, auditoria persistente, RAG e Bot Telegram."""

from __future__ import annotations

import argparse
import signal
import sys
import threading
from datetime import UTC, datetime
from types import FrameType
from typing import Any

import structlog

from rbac_assistant.application.services.catalog_service import CatalogService
from rbac_assistant.application.services.chat_service import NormativeChatService
from rbac_assistant.application.services.indexing_service import IndexingService
from rbac_assistant.application.services.validation_service import ValidationService
from rbac_assistant.application.use_cases.check_system_status import CheckSystemStatusUseCase
from rbac_assistant.config.logging_config import configure_logging
from rbac_assistant.config.settings import AppSettings, load_settings
from rbac_assistant.domain.entities.regulatory_document import ValidationPolicy
from rbac_assistant.domain.exceptions import AuditWriteError
from rbac_assistant.domain.ports.audit_port import AuditEvent
from rbac_assistant.domain.ports.rag_ports import EmbeddingPort, LLMPort
from rbac_assistant.infrastructure.audit.jsonl_audit_adapter import JsonlAuditAdapter
from rbac_assistant.infrastructure.llm.gemini_client import (
    GeminiClientAdapter,
    GeminiEmbeddingAdapter,
)
from rbac_assistant.infrastructure.storage.sqlite_document_repository import (
    SqliteDocumentRepository,
)
from rbac_assistant.infrastructure.vector_db.chroma_vector_store import ChromaVectorStore
from rbac_assistant.interfaces.telegram.bot_runner import (
    TelegramBotRunner,
    is_usable_telegram_token,
)
from rbac_assistant.interfaces.telegram.command_router import (
    TelegramCommandRouter,
    TelegramRuntimeInfo,
)
from rbac_assistant.interfaces.telegram.message_splitter import MessageSplitter
from rbac_assistant.rag.generation.prompt_templates import PromptTemplateStore
from rbac_assistant.rag.ingestion.structural_chunker import StructuralNormativeChunker
from rbac_assistant.rag.retrieval.hybrid_retriever import HybridNormativeRetriever
from rbac_assistant.security.access_control import AccessControl
from rbac_assistant.security.rate_limiter import UserRateLimiter


class _MetadataOnlyEmbeddingPort:
    """Descritor somente leitura usado para diagnóstico de índices quando sem GEMINI_API_KEY."""

    def __init__(self, *, model_name: str, dimensions: int, strategy_version: str) -> None:
        self._model_name = model_name
        self._dimensions = dimensions
        self._strategy_version = strategy_version

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def instruction_strategy_version(self) -> str:
        return self._strategy_version

    def embed_documents(
        self,
        texts: Any,
        *,
        titles: Any = None,
    ) -> list[tuple[float, ...]]:
        del texts, titles
        raise RuntimeError("GEMINI_API_KEY ausente: operação de embedding indisponível.")

    def embed_query(self, text: str) -> tuple[float, ...]:
        del text
        raise RuntimeError("GEMINI_API_KEY ausente: operação de embedding indisponível.")


def build_telegram_router(
    *,
    settings: AppSettings,
    vector_store: ChromaVectorStore,
    audit_adapter: JsonlAuditAdapter,
    status_use_case: CheckSystemStatusUseCase,
) -> tuple[TelegramCommandRouter, SqliteDocumentRepository, GeminiClientAdapter | None]:
    """Monta todas as dependências hexagonais para o Bot Telegram sem acoplar domínio."""
    repository = SqliteDocumentRepository(
        settings.storage.sqlite_path,
        busy_timeout_ms=settings.storage.sqlite_busy_timeout_ms,
    )
    repository.initialize()

    catalog_service = CatalogService(repository)
    validation_service = ValidationService(
        policy=ValidationPolicy(settings.extraction.validation_policy)
    )
    chunker = StructuralNormativeChunker(max_chunk_tokens=settings.chunking.max_chunk_tokens)

    gemini_client: GeminiClientAdapter | None = None
    embedding_port: EmbeddingPort | None = None
    llm_port: LLMPort | None = None
    has_gemini_key = (
        settings.gemini_api_key is not None
        and bool(settings.gemini_api_key.get_secret_value().strip())
    )
    if has_gemini_key:
        gemini_client = GeminiClientAdapter(
            api_key=settings.require_gemini_api_key(),
            generation_model=settings.gemini_generation_model,
            embedding_model=settings.gemini_embedding_model,
            embedding_dimensions=settings.effective_embedding_dimensions,
            embedding_instruction_version=settings.gemini.embedding_instruction_version,
            embedding_max_input_tokens=settings.gemini.embedding_max_input_tokens,
            timeout_seconds=settings.gemini.timeout_seconds,
            max_retries=settings.gemini.max_retries,
            retry_min_wait_seconds=settings.gemini.retry_min_wait_seconds,
            retry_max_wait_seconds=settings.gemini.retry_max_wait_seconds,
            default_temperature=settings.gemini.temperature,
            default_max_output_tokens=settings.gemini.max_output_tokens,
            embedding_batch_size=settings.gemini.embedding_batch_size,
        )
        embedding_port = GeminiEmbeddingAdapter(gemini_client)
        llm_port = gemini_client

    indexing_embedding: EmbeddingPort = embedding_port or _MetadataOnlyEmbeddingPort(
        model_name=settings.gemini_embedding_model,
        dimensions=settings.effective_embedding_dimensions,
        strategy_version=settings.gemini.embedding_instruction_version,
    )
    indexing_service = IndexingService(
        repository=repository,
        catalog_service=catalog_service,
        chunker=chunker,
        vector_store=vector_store,
        embedding_port=indexing_embedding,
        default_collection_name=settings.vector_db.collection_name,
        audit_port=audit_adapter,
    )
    retriever = HybridNormativeRetriever(
        repository=repository,
        vector_store=vector_store,
        embedding_port=embedding_port,
        collection_name=settings.vector_db.collection_name,
        validation_service=validation_service,
        lexical_top_k=settings.retrieval.lexical_top_k,
        semantic_top_k=settings.retrieval.semantic_top_k,
        final_top_k=settings.retrieval.final_top_k,
        rrf_k=settings.retrieval.rrf_k,
        max_vector_distance=settings.retrieval.max_vector_distance,
        reconstruct_full_section_from_ssot=settings.retrieval.reconstruct_full_section_from_ssot,
    )
    chat_service = NormativeChatService(
        retriever=retriever,
        llm=llm_port,
        prompt_store=PromptTemplateStore(settings.gemini.prompts_dir),
        audit_port=audit_adapter,
    )
    access_control = AccessControl.from_yaml(
        settings.access_control_yaml_path,
        default_public_access=settings.security.public_access,
        audit_port=audit_adapter,
    )
    rate_limiter = UserRateLimiter(
        requests_per_minute=settings.security.rate_limit.requests_per_minute,
        burst_size=settings.security.rate_limit.burst_size,
        global_requests_per_minute=settings.security.rate_limit.global_requests_per_minute,
    )
    splitter = MessageSplitter(max_message_length=settings.telegram.max_message_length)
    runtime_info = TelegramRuntimeInfo(
        environment=settings.environment,
        generation_model=settings.gemini_generation_model,
        embedding_model=settings.gemini_embedding_model,
        embedding_dimensions=settings.effective_embedding_dimensions,
        collection_name=settings.vector_db.collection_name,
        allow_paid_models=settings.gemini_allow_paid_models,
        llm_configured=llm_port is not None,
    )
    router = TelegramCommandRouter(
        access_control=access_control,
        rate_limiter=rate_limiter,
        splitter=splitter,
        status_use_case=status_use_case,
        catalog_service=catalog_service,
        indexing_service=indexing_service,
        chat_service=chat_service,
        runtime_info=runtime_info,
        audit_port=audit_adapter,
        max_input_length=settings.telegram.max_input_length,
        max_concurrent_updates_per_user=settings.telegram.max_concurrent_updates_per_user,
        max_global_concurrent_updates=settings.telegram.max_global_concurrent_updates,
        command_timeout_seconds=settings.telegram.command_timeout_seconds,
        duplicate_update_cache_size=settings.telegram.duplicate_update_cache_size,
        only_approved_literal=settings.retrieval.telegram_only_approved_literal,
        allow_preliminary_native_search=(
            not settings.retrieval.telegram_only_approved_literal
            and settings.retrieval.allow_preliminary_native_search
        ),
    )
    return router, repository, gemini_client


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RBAC AI Assistant — Serviço Principal e Bot")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Executa somente a verificação de inicialização e encerra.",
    )
    parser.add_argument(
        "--standby",
        action="store_true",
        help="Mantém a fundação operacional sem iniciar Long Polling do Telegram.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Inicializa infraestrutura, auditoria e Bot Telegram (Long Polling) com shutdown gracioso."""
    args = _parse_args(argv)
    settings = load_settings()
    configure_logging(
        log_level=settings.log_level,
        redact_secrets=settings.security.audit.redact_secrets,
    )
    logger = structlog.get_logger("rbac_assistant.main")

    audit_adapter = JsonlAuditAdapter.from_settings(settings)
    vector_store = ChromaVectorStore(
        host=settings.chroma_host,
        port=settings.chroma_port,
        heartbeat_path=settings.vector_db.heartbeat_path,
        timeout_seconds=settings.vector_db.request_timeout_seconds,
    )
    status_use_case = CheckSystemStatusUseCase(
        vector_store=vector_store,
        documents_dir=settings.storage.documents_dir,
        audit_logs_dir=settings.storage.audit_logs_dir,
        audit_port=audit_adapter,
    )
    report = status_use_case.execute()

    raw_tg_token = (
        settings.telegram_bot_token.get_secret_value()
        if settings.telegram_bot_token is not None
        else None
    )
    tg_token_usable = is_usable_telegram_token(raw_tg_token)

    logger.info(
        "foundation_startup_check",
        environment=settings.environment,
        chroma_url=settings.chroma_heartbeat_url,
        chroma_healthy=report.chroma_healthy,
        chroma_heartbeat=report.chroma_heartbeat,
        documents_dir_writable=report.documents_dir_writable,
        audit_logs_dir_writable=report.audit_logs_dir_writable,
        audit_file=str(audit_adapter.audit_file_path),
        telegram_token_configured=tg_token_usable,
        telegram_access_mode="public" if settings.security.public_access else "restricted",
        telegram_allow_group_chats=settings.telegram.allow_group_chats,
        access_control_path=str(settings.access_control_yaml_path),
        access_control_present=settings.access_control_yaml_path.is_file(),
        ocr_policy=settings.extraction.ocr_unreviewed_policy,
        error=report.error,
    )

    if not report.healthy:
        logger.error("foundation_startup_failed", error=report.error)
        return 1

    if args.check_only:
        return 0

    stop_event = threading.Event()

    def _handle_shutdown(signum: int, _frame: FrameType | None) -> None:
        logger.info("shutdown_signal_received", signal=signum)
        stop_event.set()

    signal.signal(signal.SIGTERM, _handle_shutdown)
    signal.signal(signal.SIGINT, _handle_shutdown)

    try:
        audit_adapter.record(
            AuditEvent(
                timestamp=datetime.now(UTC),
                action="service_lifecycle",
                status="started",
                actor_id="system",
                details={
                    "environment": settings.environment,
                    "telegram_token_configured": tg_token_usable,
                },
            )
        )
    except AuditWriteError as exc:
        logger.error("audit_write_failed_on_startup", error=str(exc))
        return 1

    repository: SqliteDocumentRepository | None = None
    gemini_client: GeminiClientAdapter | None = None
    exit_code = 0

    try:
        should_poll = (
            not args.standby and settings.telegram_polling_enabled and tg_token_usable
        )
        if should_poll and raw_tg_token is not None:
            router, repository, gemini_client = build_telegram_router(
                settings=settings,
                vector_store=vector_store,
                audit_adapter=audit_adapter,
                status_use_case=status_use_case,
            )
            runner = TelegramBotRunner(
                token=raw_tg_token,
                router=router,
                polling_timeout_seconds=settings.telegram.polling_timeout_seconds,
                startup_retry_max_attempts=settings.telegram.startup_retry_max_attempts,
                startup_retry_base_delay_seconds=(
                    settings.telegram.startup_retry_base_delay_seconds
                ),
                allow_group_chats=settings.telegram.allow_group_chats,
                audit_port=audit_adapter,
            )
            exit_code = runner.run(stop_event=stop_event)
        else:
            reason = (
                "standby_flag"
                if args.standby
                else (
                    "polling_disabled"
                    if not settings.telegram_polling_enabled
                    else "missing_or_placeholder_token"
                )
            )
            logger.warning(
                "telegram_bot_standby",
                reason=reason,
                message=(
                    "Serviço operacional em modo standby (Long Polling Telegram inativo); "
                    "CLIs administrativos e healthcheck seguem ativos."
                ),
            )
            try:
                audit_adapter.record(
                    AuditEvent(
                        timestamp=datetime.now(UTC),
                        action="telegram_polling",
                        status="standby",
                        actor_id="system",
                        details={"reason": reason},
                    )
                )
            except AuditWriteError as exc:
                logger.error("audit_write_failed_on_standby", error=str(exc))
            stop_event.wait()
    finally:
        if gemini_client is not None:
            gemini_client.close()
        if repository is not None:
            repository.close()
        try:
            audit_adapter.record(
                AuditEvent(
                    timestamp=datetime.now(UTC),
                    action="service_lifecycle",
                    status="stopped",
                    actor_id="system",
                    details={"environment": settings.environment, "exit_code": exit_code},
                )
            )
        except AuditWriteError as exc:
            logger.error("audit_write_failed_on_shutdown", error=str(exc))
        logger.info("foundation_stopped", exit_code=exit_code)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
