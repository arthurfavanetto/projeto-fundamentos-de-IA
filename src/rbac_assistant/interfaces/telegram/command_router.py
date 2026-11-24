"""Roteador de comandos e texto livre do Bot Telegram integrado à Clean Architecture.

Regras:
- Aplica `AccessControl` determinístico (modo público `USER` com privilégios `ADMIN` restritos
  aos administradores cadastrados, ou modo restrito quando `public_access: false`) em TODOS os
  comandos e mensagens de texto livre.
- Restringe por padrão a recuperação normativa na interface Telegram exclusivamente a documentos
  e páginas aprovados e indexados (`only_approved_literal=True`).
- Separa rigorosamente respostas públicas (`/status`, `/documentos`, `/versao`) de diagnósticos
  administrativos (`ADMIN`), sem vazar caminhos internos, credenciais ou dados de outros usuários.
- Implementa `/id` retornando exclusivamente o Telegram User ID numérico do próprio solicitante.
- Protege a operação pública contra abuso com limite de entrada, rate limit por usuário e global,
  limite global de concorrência, timeout de chamadas e deduplicação de updates — sem memória
  conversacional cruzada entre usuários.
"""

from __future__ import annotations

import asyncio
import re
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

import structlog

from rbac_assistant import __version__ as app_version
from rbac_assistant.application.services.catalog_service import CatalogService
from rbac_assistant.application.services.chat_service import NormativeChatService
from rbac_assistant.application.services.indexing_service import IndexingService
from rbac_assistant.application.use_cases.check_system_status import CheckSystemStatusUseCase
from rbac_assistant.domain.entities.rag import SearchFilter, StructuredNormativeAnswer
from rbac_assistant.domain.exceptions import AuditWriteError, DomainError
from rbac_assistant.domain.ports.audit_port import AuditEvent, AuditPort
from rbac_assistant.interfaces.telegram.message_splitter import MessageSplitter
from rbac_assistant.security.access_control import AccessControl, OperatorRole
from rbac_assistant.security.rate_limiter import UserRateLimiter
from rbac_assistant.security.secret_redactor import redact_text

logger = structlog.get_logger(__name__)

SUPPORTED_COMMANDS: tuple[str, ...] = (
    "/start",
    "/help",
    "/id",
    "/status",
    "/rbac",
    "/buscar",
    "/explicar",
    "/documentos",
    "/versao",
)

_MAX_TRACKED_USER_SEMAPHORES = 2048

# Política determinística de intenção para texto livre:
# 1. Pedido explícito de explicação/interpretação -> aciona explicação separada via Gemini.
# 2. Pedido literal/transcrição -> somente texto literal e referência (zero Gemini).
# 3. Pergunta analítica inequívoca (com '?' ou estrutura interrogativa clara) -> explicação.
# 4. Consulta ambígua, citação isolada ou termos soltos -> prioriza evidência literal (zero Gemini).
_EXPLICIT_EXPLANATION_RE = re.compile(
    r"\b(?:expliqu\w*|explica[çc][ãa]o|resum\w*|interpret\w*|esclare[çc]\w*|"
    r"o\s+que\s+significa|significa\s+o\s+qu[êe]|como\s+funciona|"
    r"qual\s+a\s+diferen[çc]a|diferen[çc]a\s+entre|quando\s+se\s+aplica)\b",
    re.IGNORECASE,
)
_LITERAL_ONLY_REQUEST_RE = re.compile(
    r"\b(?:transcrev\w*|transcri[çc][ãa]o|texto\s+literal|texto\s+exato|"
    r"texto\s+integral|ipsis\s+litteris|literalmente|reproduz\w*|copi[ae]\w*|"
    r"(?:mostr[ae]|exib[ae]|traga|envi[ae]|qual\s+(?:[ée]\s+)?o)\s+o?\s*texto\b|"
    r"ler?\s+a\s+se[çc][ãa]o|cit[ae]\s+literal\w*)\b",
    re.IGNORECASE,
)
_ANALYTICAL_QUESTION_RE = re.compile(
    r"(?:^\s*(?:quem|como|quando|por\s+que|por\s+qu[êe]|quais|qual|onde|"
    r"pode|podem|deve|devem|precisa|precisam|[ée]\s+obrigat[óo]rio|"
    r"[ée]\s+permitido|[ée]\s+poss[íi]vel)\b.*\?\s*$|"
    r"\b(?:quem\s+pode|como\s+deve|quando\s+deve|por\s+que\s+|quais\s+s[ãa]o\s+os\s+requisitos)\b)",
    re.IGNORECASE,
)


def should_generate_explanation_for_free_text(text: str) -> bool:
    """Decide deterministicamente se uma consulta livre exige explicação IA separada.

    Retorna `False` (somente transcrição literal + referência, sem chamar Gemini) para:
    - pedidos de transcrição literal;
    - citações normativas diretas (ex.: `RBAC 43.3`, `F43.1`, `Apêndice E`);
    - consultas curtas ou ambíguas sem intenção explicativa clara.
    """
    cleaned = text.strip()
    if not cleaned:
        return False
    if _EXPLICIT_EXPLANATION_RE.search(cleaned):
        return True
    if _LITERAL_ONLY_REQUEST_RE.search(cleaned):
        return False
    return bool(_ANALYTICAL_QUESTION_RE.search(cleaned))


@dataclass(frozen=True, slots=True)
class TelegramRuntimeInfo:
    """Metadados não-sensíveis de runtime expostos em `/status` e `/versao`."""

    environment: str
    generation_model: str
    embedding_model: str
    embedding_dimensions: int
    collection_name: str
    allow_paid_models: bool
    llm_configured: bool


@dataclass(frozen=True, slots=True)
class TelegramResponse:
    """Resposta pronta para envio ao Telegram (já paginada e sanitizada)."""

    messages: tuple[str, ...]
    handled: bool = True
    duplicate_ignored: bool = False
    access_denied: bool = False
    rate_limited: bool = False
    command: str = "text"


class TelegramCommandRouter:
    """Orquestra autorização, proteção operacional e delegação aos serviços de aplicação."""

    def __init__(
        self,
        *,
        access_control: AccessControl,
        rate_limiter: UserRateLimiter,
        splitter: MessageSplitter,
        status_use_case: CheckSystemStatusUseCase,
        catalog_service: CatalogService,
        indexing_service: IndexingService,
        chat_service: NormativeChatService,
        runtime_info: TelegramRuntimeInfo,
        audit_port: AuditPort | None = None,
        max_input_length: int = 1000,
        max_concurrent_updates_per_user: int = 1,
        max_global_concurrent_updates: int = 8,
        command_timeout_seconds: float = 45.0,
        duplicate_update_cache_size: int = 512,
        only_approved_literal: bool = True,
        allow_preliminary_native_search: bool = False,
    ) -> None:
        self._access_control = access_control
        self._rate_limiter = rate_limiter
        self._splitter = splitter
        self._status_use_case = status_use_case
        self._catalog_service = catalog_service
        self._indexing_service = indexing_service
        self._chat_service = chat_service
        self._runtime_info = runtime_info
        self._audit_port = audit_port
        self._max_input_length = max_input_length
        self._max_concurrent_updates_per_user = max_concurrent_updates_per_user
        self._max_global_concurrent_updates = max(1, max_global_concurrent_updates)
        self._command_timeout_seconds = max(1.0, float(command_timeout_seconds))
        self._duplicate_update_cache_size = duplicate_update_cache_size
        self._only_approved_literal = only_approved_literal
        self._allow_preliminary_native_search = (
            False if only_approved_literal else allow_preliminary_native_search
        )

        self._seen_updates: OrderedDict[int, None] = OrderedDict()
        self._global_semaphore = asyncio.Semaphore(self._max_global_concurrent_updates)
        self._user_semaphores: OrderedDict[int, asyncio.Semaphore] = OrderedDict()

    def _is_duplicate_update(self, update_id: int | None) -> bool:
        if update_id is None:
            return False
        if update_id in self._seen_updates:
            return True
        self._seen_updates[update_id] = None
        while len(self._seen_updates) > self._duplicate_update_cache_size:
            self._seen_updates.popitem(last=False)
        return False

    def _get_user_semaphore(self, user_id: int) -> asyncio.Semaphore:
        sem = self._user_semaphores.get(user_id)
        if sem is not None:
            self._user_semaphores.move_to_end(user_id)
            return sem
        if len(self._user_semaphores) >= _MAX_TRACKED_USER_SEMAPHORES:
            for stale_uid, stale_sem in list(self._user_semaphores.items()):
                if not stale_sem.locked():
                    del self._user_semaphores[stale_uid]
                    break
        sem = asyncio.Semaphore(self._max_concurrent_updates_per_user)
        self._user_semaphores[user_id] = sem
        return sem

    def _record_audit(
        self,
        *,
        action: str,
        status: str,
        actor_id: str | None,
        role: str | None,
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
                    actor_id=actor_id,
                    role=role,
                    details=details,
                )
            )
        except AuditWriteError as exc:
            logger.error("telegram_audit_write_failed", error=redact_text(str(exc)))

    def _parse_command(self, raw_text: str) -> tuple[str, str]:
        """Extrai comando normalizado (sem `@BotName`) e argumentos preservados."""
        stripped = raw_text.strip()
        if not stripped.startswith("/"):
            return "text", stripped
        parts = stripped.split(maxsplit=1)
        cmd_token = parts[0].split("@", maxsplit=1)[0].lower()
        args_text = parts[1].strip() if len(parts) > 1 else ""
        return cmd_token, args_text

    def _default_search_filter(self) -> SearchFilter:
        return SearchFilter(
            only_current_version=True,
            only_approved_literal=self._only_approved_literal,
            include_preliminary_native=self._allow_preliminary_native_search,
        )

    def _handle_start(self, role: OperatorRole) -> str:
        mode_note = (
            "Modo de acesso: público em conversas privadas (perfil padrão USER)."
            if self._access_control.public_access
            else "Modo de acesso: restrito a operadores autorizados."
        )
        return (
            "RBAC AI Assistant — Assistente Normativo de Aviação Civil (ANAC)\n"
            f"Perfil ativo: {role.value}\n"
            f"{mode_note}\n\n"
            "Este assistente responde exclusivamente com base nos regulamentos oficiais da ANAC "
            "efetivamente aprovados e indexados na base local, separando estritamente:\n"
            "- TEXTO NORMATIVO (literal validado, sem reescrita)\n"
            "- EXPLICAÇÃO (gerada por IA apenas sobre evidências aprovadas)\n"
            "- REFERÊNCIA (documento, emenda, seção e páginas PDF/impressa)\n\n"
            "Perguntas fora do escopo documental ou sem evidência suficiente recebem recusa "
            "fundamentada, sem pesquisa na internet nem uso de conhecimento geral.\n"
            "Envie /help para ver todos os comandos ou faça sua pergunta em texto livre."
        )

    def _handle_help(self) -> str:
        return (
            "Comandos disponíveis do RBAC AI Assistant (conversas privadas):\n\n"
            "/start — Apresentação, escopo normativo e perfil ativo\n"
            "/help — Lista de comandos e instruções de uso\n"
            "/id — Exibe exclusivamente o seu próprio Telegram User ID numérico\n"
            "/status — Disponibilidade do assistente e base normativa\n"
            "/rbac <seção> — Consulta normativa direta sem reescrita IA (ex.: /rbac 43.3)\n"
            "/buscar <termo> — Busca híbrida (estrutural + lexical + vetorial) sem geração IA\n"
            "/explicar <seção ou pergunta> — Texto normativo literal + explicação fundamentada IA\n"
            "/documentos — Lista regulamentos disponíveis na base\n"
            "/versao — Versão do assistente, política Gemini e emendas vigentes na base\n\n"
            "Texto livre: envie uma pergunta em linguagem natural sobre regulamentos da ANAC. "
            "As respostas utilizam exclusivamente documentos aprovados e indexados."
        )

    def _handle_id(self, user_id: int) -> str:
        """Retorna exclusivamente o Telegram User ID numérico do próprio solicitante."""
        return (
            f"Seu Telegram User ID numérico: {user_id}\n\n"
            "Este identificador é exibido exclusivamente para você nesta conversa privada."
        )

    def _handle_status(self, role: OperatorRole) -> str:
        status_report = self._status_use_case.execute()
        index_report = self._indexing_service.get_index_status()
        overall = "SAUDÁVEL" if status_report.healthy else "DEGRADADO"

        # Formato público sanitizado (sem caminhos internos, heartbeats, coleção ou credenciais)
        lines = [
            f"=== STATUS DO SISTEMA: {overall} ===",
            f"Documentos cadastrados: {index_report.total_documents}",
            (
                f"Versões indexadas: {index_report.indexed_versions_count}/"
                f"{index_report.total_versions}"
            ),
            f"Geração IA configurada: {'sim' if self._runtime_info.llm_configured else 'não'}",
            "Escopo normativo: apenas documentos ANAC aprovados e indexados",
        ]
        if role == OperatorRole.ADMIN:
            access_mode = "público (USER)" if self._access_control.public_access else "restrito"
            lines.extend(
                [
                    "",
                    "--- Diagnóstico ADMIN ---",
                    f"Ambiente: {self._runtime_info.environment}",
                    f"Política de acesso: {access_mode}",
                    f"Administradores configurados: {self._access_control.admin_count}",
                    f"ChromaDB: {'OK' if status_report.chroma_healthy else 'INDISPONÍVEL'}",
                    f"ChromaDB heartbeat: {status_report.chroma_heartbeat}",
                    f"Chunks SQLite (FTS5): {index_report.total_sqlite_chunks}",
                    f"Vetores ChromaDB: {index_report.chroma_vector_count}",
                    f"Volume documentos gravável: {status_report.documents_dir_writable}",
                    f"Volume auditoria gravável: {status_report.audit_logs_dir_writable}",
                    f"Coleção vetorial: {index_report.collection_name}",
                    f"Versões não indexadas: {len(index_report.unindexed_versions)}",
                    f"Versões defasadas: {len(index_report.stale_versions)}",
                ]
            )
            if index_report.compatibility_warning:
                lines.append(f"Alerta compatibilidade: {index_report.compatibility_warning}")
            if status_report.error:
                lines.append(f"Erro reportado: {redact_text(status_report.error)}")
        return "\n".join(lines)

    def _handle_documentos(self, role: OperatorRole) -> str:
        docs = self._catalog_service.search_documents()
        index_report = self._indexing_service.get_index_status()
        indexed_by_doc = {
            v.document_id: v for v in index_report.version_summaries if v.is_indexed
        }
        if not docs:
            if role == OperatorRole.ADMIN:
                return (
                    "Nenhum documento regulatório está cadastrado no catálogo SQLite no momento.\n"
                    "Execute a ingestão controlada via `scripts/ingest_regulation.py` no servidor."
                )
            return "Nenhum documento regulatório aprovado está disponível na base no momento."

        lines = ["=== DOCUMENTOS REGULATÓRIOS DISPONÍVEIS ==="]
        for doc in docs:
            idx_summary = indexed_by_doc.get(doc.document_id)
            if role == OperatorRole.ADMIN:
                idx_label = (
                    f"indexado ({idx_summary.indexed_chunk_count} chunks)"
                    if idx_summary is not None
                    else "não indexado"
                )
            else:
                idx_label = "indexado" if idx_summary is not None else "não indexado"
            lines.append(
                f"- {doc.authority} {doc.document_type.value} {doc.regulation_number} "
                f"(Emenda {doc.amendment_number} | vigência: {doc.version_status.value}) — "
                f"{doc.document_title} [{idx_label}]"
            )
        return "\n".join(lines)

    def _handle_versao(self, role: OperatorRole) -> str:
        docs = self._catalog_service.search_documents()
        lines = [
            f"RBAC AI Assistant v{app_version}",
            "Política Gemini (Free Tier Standard):",
            f"- Geração: {self._runtime_info.generation_model}",
            (
                f"- Embeddings: {self._runtime_info.embedding_model} "
                f"({self._runtime_info.embedding_dimensions}d)"
            ),
            f"- Modelos pagos permitidos: {self._runtime_info.allow_paid_models}",
        ]
        if role == OperatorRole.ADMIN:
            lines.extend(
                [
                    f"- Ambiente: {self._runtime_info.environment}",
                    f"- Coleção ChromaDB: {self._runtime_info.collection_name}",
                ]
            )
        lines.extend(["", "Versões regulatórias na base:"])
        if not docs:
            lines.append("- Nenhum regulamento disponível até o momento.")
        else:
            for doc in docs:
                lines.append(
                    f"- {doc.authority} {doc.document_type.value} {doc.regulation_number}: "
                    f"Emenda {doc.amendment_number} (status: {doc.version_status.value})"
                )
        return "\n".join(lines)

    def _handle_rbac(
        self, args_text: str, *, actor_id: str, role: OperatorRole
    ) -> str | StructuredNormativeAnswer:
        if not args_text:
            return "Uso: /rbac <seção ou referência>\nExemplo: /rbac 43.3 ou /rbac RBAC 43.9(a)"
        query = args_text if "rbac" in args_text.lower() else f"RBAC {args_text}"
        return self._chat_service.ask(
            question=query,
            filters=self._default_search_filter(),
            generate_explanation=False,
            actor_id=actor_id,
            actor_role=role.value,
        )

    def _handle_buscar(
        self, args_text: str, *, actor_id: str, role: OperatorRole
    ) -> str | StructuredNormativeAnswer:
        if not args_text:
            return "Uso: /buscar <termo ou assunto>\nExemplo: /buscar manutencao preventiva"
        return self._chat_service.ask(
            question=args_text,
            filters=self._default_search_filter(),
            generate_explanation=False,
            actor_id=actor_id,
            actor_role=role.value,
        )

    def _handle_explicar(
        self, args_text: str, *, actor_id: str, role: OperatorRole
    ) -> str | StructuredNormativeAnswer:
        if not args_text:
            return (
                "Uso: /explicar <seção ou pergunta>\n"
                "Exemplo: /explicar 43.3 ou /explicar Quem pode executar manutenção preventiva?"
            )
        return self._chat_service.ask(
            question=args_text,
            filters=self._default_search_filter(),
            generate_explanation=True,
            actor_id=actor_id,
            actor_role=role.value,
        )

    def _handle_free_text(
        self, text: str, *, actor_id: str, role: OperatorRole
    ) -> StructuredNormativeAnswer:
        wants_explanation = should_generate_explanation_for_free_text(text)
        return self._chat_service.ask(
            question=text,
            filters=self._default_search_filter(),
            generate_explanation=wants_explanation,
            actor_id=actor_id,
            actor_role=role.value,
        )

    def _dispatch_sync(
        self,
        *,
        command: str,
        args_text: str,
        user_id: int,
        actor_id: str,
        role: OperatorRole,
    ) -> str | StructuredNormativeAnswer:
        if command == "/start":
            return self._handle_start(role)
        if command == "/help":
            return self._handle_help()
        if command == "/id":
            return self._handle_id(user_id)
        if command == "/status":
            return self._handle_status(role)
        if command == "/documentos":
            return self._handle_documentos(role)
        if command == "/versao":
            return self._handle_versao(role)
        if command == "/rbac":
            return self._handle_rbac(args_text, actor_id=actor_id, role=role)
        if command == "/buscar":
            return self._handle_buscar(args_text, actor_id=actor_id, role=role)
        if command == "/explicar":
            return self._handle_explicar(args_text, actor_id=actor_id, role=role)
        if command == "text":
            return self._handle_free_text(args_text, actor_id=actor_id, role=role)
        return (
            f"Comando não reconhecido: {command}\n"
            "Envie /help para consultar a lista de comandos suportados."
        )

    async def handle_message(
        self,
        *,
        user_id: object,
        text: str | None,
        update_id: int | None = None,
        run_sync: Callable[..., object] | None = None,
    ) -> TelegramResponse:
        """Processa comando ou texto livre com autorização, rate limit, timeout e paginação."""
        if self._is_duplicate_update(update_id):
            logger.info("telegram_duplicate_update_ignored", update_id=update_id)
            return TelegramResponse(messages=(), handled=False, duplicate_ignored=True)

        raw_text = text or ""
        command, args_text = self._parse_command(raw_text)

        # 1. Controle de acesso determinístico (modo público USER + ADMIN isolado ou restrito)
        decision = self._access_control.authorize(user_id, command=command)
        if not decision.allowed or decision.user_id is None or decision.role is None:
            return TelegramResponse(
                messages=self._splitter.split(decision.safe_message),
                handled=True,
                access_denied=True,
                command=command,
            )

        authorized_uid = decision.user_id
        authorized_role = decision.role
        actor_id = str(authorized_uid)

        # 2. Proteção contra mensagens vazias ou excessivamente longas
        if not raw_text.strip():
            return TelegramResponse(
                messages=self._splitter.split(
                    "Envie uma pergunta normativa ou utilize /help para listar os comandos."
                ),
                command=command,
            )

        if len(raw_text) > self._max_input_length:
            self._record_audit(
                action="telegram_input_rejected",
                status="too_long",
                actor_id=actor_id,
                role=authorized_role.value,
                details={
                    "command": command,
                    "input_length": len(raw_text),
                    "max_input_length": self._max_input_length,
                },
            )
            return TelegramResponse(
                messages=self._splitter.split(
                    f"Mensagem muito longa ({len(raw_text)} caracteres). "
                    f"O limite máximo por consulta é de {self._max_input_length} caracteres. "
                    "Reduza a pergunta ou consulte uma seção específica (ex.: /rbac 43.3)."
                ),
                command=command,
            )

        # 3. Rate limiting por usuário e global para proteção da operação pública
        rate = self._rate_limiter.check(authorized_uid)
        if not rate.allowed:
            self._record_audit(
                action="telegram_rate_limited",
                status="blocked",
                actor_id=actor_id,
                role=authorized_role.value,
                details={
                    "command": command,
                    "scope": rate.scope,
                    "retry_after_seconds": rate.retry_after_seconds,
                },
            )
            return TelegramResponse(
                messages=self._splitter.split(
                    "Limite temporário de consultas atingido. "
                    f"Aguarde aproximadamente {rate.retry_after_seconds:.0f}s e tente novamente."
                ),
                rate_limited=True,
                command=command,
            )

        # 4. Execução isolada (sem memória cruzada) com concorrência global + por usuário e timeout
        user_semaphore = self._get_user_semaphore(authorized_uid)
        async with self._global_semaphore, user_semaphore:
            try:
                raw_reply: object
                if run_sync is not None:
                    raw_reply = run_sync(
                        self._dispatch_sync,
                        command=command,
                        args_text=args_text,
                        user_id=authorized_uid,
                        actor_id=actor_id,
                        role=authorized_role,
                    )
                else:
                    raw_reply = await asyncio.wait_for(
                        asyncio.to_thread(
                            self._dispatch_sync,
                            command=command,
                            args_text=args_text,
                            user_id=authorized_uid,
                            actor_id=actor_id,
                            role=authorized_role,
                        ),
                        timeout=self._command_timeout_seconds,
                    )
                self._record_audit(
                    action="telegram_command_handled",
                    status="success",
                    actor_id=actor_id,
                    role=authorized_role.value,
                    details={"command": command},
                )
                messages = (
                    self._splitter.split_normative_answer(raw_reply)
                    if isinstance(raw_reply, StructuredNormativeAnswer)
                    else self._splitter.split(str(raw_reply))
                )
                return TelegramResponse(
                    messages=messages,
                    command=command,
                )
            except TimeoutError:
                logger.warning(
                    "telegram_command_timeout",
                    command=command,
                    timeout_seconds=self._command_timeout_seconds,
                )
                self._record_audit(
                    action="telegram_command_handled",
                    status="timeout",
                    actor_id=actor_id,
                    role=authorized_role.value,
                    details={
                        "command": command,
                        "timeout_seconds": self._command_timeout_seconds,
                    },
                )
                return TelegramResponse(
                    messages=self._splitter.split(
                        "O tempo limite para processar sua consulta foi excedido. "
                        "Tente novamente em instantes ou refine sua pergunta."
                    ),
                    command=command,
                )
            except DomainError as exc:
                safe_err = redact_text(str(exc))
                logger.warning(
                    "telegram_domain_error",
                    command=command,
                    error_type=type(exc).__name__,
                    error=safe_err,
                )
                self._record_audit(
                    action="telegram_command_handled",
                    status="domain_error",
                    actor_id=actor_id,
                    role=authorized_role.value,
                    details={"command": command, "error_type": type(exc).__name__},
                )
                return TelegramResponse(
                    messages=self._splitter.split(
                        f"Não foi possível concluir a operação solicitada: {safe_err}"
                    ),
                    command=command,
                )
            except Exception as exc:
                safe_err = redact_text(str(exc))
                logger.error(
                    "telegram_unexpected_error",
                    command=command,
                    error_type=type(exc).__name__,
                    error=safe_err,
                )
                self._record_audit(
                    action="telegram_command_handled",
                    status="internal_error",
                    actor_id=actor_id,
                    role=authorized_role.value,
                    details={"command": command, "error_type": type(exc).__name__},
                )
                return TelegramResponse(
                    messages=self._splitter.split(
                        "Ocorreu um erro interno ao processar sua solicitação. "
                        "Nenhuma informação normativa foi inventada; tente novamente mais tarde."
                    ),
                    command=command,
                )
