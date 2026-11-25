"""Testes offline da camada de orientação conversacional e fidelidade normativa no Telegram.

Cobre todos os 10 cenários obrigatórios:
1. Consulta literal específica (preservando granularidade estrutural do PR #21).
2. Pergunta explicativa com evidências aprovadas (e continuidade dêitica).
3. Pergunta vaga que solicita esclarecimento (zero embeddings e zero Gemini).
4. Usuário que responde ao esclarecimento (opção numérica, texto livre, referência direta e TTL).
5. Pergunta fora dos documentos aprovados (fail-closed, zero Gemini).
6. Referência estrutural inexistente (fail-closed PR #21, zero Gemini).
7. Consulta com várias referências sendo uma inválida (bloqueio integral PR #21, zero Gemini).
8. Separação estrita entre transcrição literal oficial e explicação IA não oficial.
9. Comandos `/start` e `/ajuda` (e compatibilidade com `/help`).
10. Isolamento completo de contexto entre usuários e proteção contra conteúdo não aprovado.
"""

from __future__ import annotations

import hashlib
import math
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from rbac_assistant.application.services.catalog_service import CatalogService
from rbac_assistant.application.services.chat_service import NormativeChatService
from rbac_assistant.application.services.indexing_service import IndexingService
from rbac_assistant.application.services.validation_service import ValidationService
from rbac_assistant.application.use_cases.check_system_status import CheckSystemStatusUseCase
from rbac_assistant.domain.entities.document_version import DocumentVersion, ExtractionRun
from rbac_assistant.domain.entities.normative_unit import ExtractionMethod
from rbac_assistant.domain.entities.rag import (
    IndexStatusReport,
    IndexVersionSummary,
)
from rbac_assistant.domain.entities.regulatory_document import (
    DocumentType,
    RegulatoryDocument,
    ValidationPolicy,
    VersionStatus,
)
from rbac_assistant.domain.entities.regulatory_page import (
    PageValidationStatus,
    RegulatoryPage,
    compute_page_content_hash,
)
from rbac_assistant.domain.ports.vector_store_port import VectorRecord, VectorSearchHit
from rbac_assistant.infrastructure.audit.jsonl_audit_adapter import JsonlAuditAdapter
from rbac_assistant.infrastructure.pdf.anac_structure_strategy import AnacRbacStructureStrategy
from rbac_assistant.infrastructure.storage.sqlite_document_repository import (
    SqliteDocumentRepository,
)
from rbac_assistant.interfaces.telegram.command_router import (
    SUPPORTED_COMMANDS,
    TelegramCommandRouter,
    TelegramRuntimeInfo,
)
from rbac_assistant.interfaces.telegram.conversational_guidance import (
    ConversationIntent,
    UserConversationStore,
    classify_conversational_intent,
)
from rbac_assistant.interfaces.telegram.message_splitter import (
    MessageSplitter,
    strip_pagination_header,
)
from rbac_assistant.rag.generation.response_formatter import (
    EXPLANATION_NOT_OFFICIAL_NOTICE,
    extract_literal_excerpts_from_text,
)
from rbac_assistant.rag.ingestion.structural_chunker import StructuralNormativeChunker
from rbac_assistant.rag.retrieval.hybrid_retriever import HybridNormativeRetriever
from rbac_assistant.security.access_control import AccessControl
from rbac_assistant.security.rate_limiter import UserRateLimiter


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class _SpyEmbeddingPort:
    """Embedder determinístico offline que contabiliza chamadas a `embed_query`."""

    model_name = "spy-deterministic-embed-256d"
    dimensions = 256
    instruction_strategy_version = "spy_v1"

    def __init__(self) -> None:
        self.query_calls: list[str] = []

    _STOPWORDS = frozenset(
        {
            "para",
            "como",
            "qual",
            "quais",
            "sobre",
            "entre",
            "esta",
            "este",
            "pela",
            "pelo",
            "outro",
            "outra",
            "qualquer",
            "aplicáveis",
            "requisitos",
            "seção",
        }
    )

    def _encode(self, text: str) -> tuple[float, ...]:
        vec = [0.0] * self.dimensions
        tokens = [
            tok
            for tok in re.findall(r"[a-zA-ZÀ-ÿ0-9]+", text.lower())
            if len(tok) >= 4 and tok not in self._STOPWORDS
        ]
        for tok in tokens:
            idx = int(hashlib.sha256(tok.encode("utf-8")).hexdigest(), 16) % self.dimensions
            vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return tuple(v / norm for v in vec)

    def embed_documents(
        self,
        texts: Any,
        *,
        titles: Any = None,
    ) -> list[tuple[float, ...]]:
        del titles
        return [self._encode(str(t)) for t in texts]

    def embed_query(self, text: str) -> tuple[float, ...]:
        self.query_calls.append(text)
        return self._encode(text)


class _InMemoryVectorStore:
    """VectorStore em memória para testes de integração 100% offline."""

    def __init__(self) -> None:
        self._collections: dict[str, dict[str, VectorRecord]] = {}
        self._meta: dict[str, dict[str, Any]] = {}

    def check_heartbeat(self) -> int:
        return 12345

    def get_collection_metadata(self, collection_name: str) -> dict[str, Any] | None:
        return self._meta.get(collection_name)

    def ensure_collection_compatible(
        self,
        *,
        collection_name: str,
        embedding_model: str,
        embedding_dimension: int,
        instruction_strategy_version: str = "",
    ) -> None:
        self._meta.setdefault(
            collection_name,
            {
                "embedding_model": embedding_model,
                "embedding_dimension": embedding_dimension,
                "instruction_strategy_version": instruction_strategy_version,
            },
        )

    def upsert_records(
        self,
        collection_name: str,
        records: Any,
        *,
        collection_metadata: dict[str, Any] | None = None,
    ) -> None:
        if collection_metadata:
            self._meta[collection_name] = dict(collection_metadata)
        bucket = self._collections.setdefault(collection_name, {})
        for rec in records:
            bucket[rec.record_id] = rec

    def delete_records(self, collection_name: str, record_ids: Any) -> None:
        bucket = self._collections.get(collection_name, {})
        for rid in record_ids:
            bucket.pop(rid, None)

    def list_record_ids(
        self,
        *,
        collection_name: str,
        where: dict[str, Any] | None = None,
    ) -> list[str]:
        bucket = self._collections.get(collection_name, {})
        return [
            rid
            for rid, rec in bucket.items()
            if not where or self._matches(rec.metadata, where)
        ]

    @staticmethod
    def _matches(meta: dict[str, Any], where: dict[str, Any]) -> bool:
        if "$and" in where:
            return all(_InMemoryVectorStore._matches(meta, clause) for clause in where["$and"])
        for k, v in where.items():
            if isinstance(v, dict) and "$in" in v:
                if meta.get(k) not in v["$in"]:
                    return False
            elif meta.get(k) != v:
                return False
        return True

    def query_similar(
        self,
        collection_name: str,
        query_embedding: tuple[float, ...],
        *,
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[VectorSearchHit]:
        bucket = self._collections.get(collection_name, {})
        scored: list[VectorSearchHit] = []
        for rec in bucket.values():
            if where and not self._matches(rec.metadata, where):
                continue
            dot = sum(a * b for a, b in zip(query_embedding, rec.embedding, strict=True))
            dist = max(0.0, 1.0 - dot)
            scored.append(
                VectorSearchHit(
                    record_id=rec.record_id,
                    document=rec.document,
                    metadata=rec.metadata,
                    distance=dist,
                )
            )
        scored.sort(key=lambda h: h.distance)
        return scored[:top_k]

    def delete_collection(self, collection_name: str) -> None:
        self._collections.pop(collection_name, None)
        self._meta.pop(collection_name, None)


class _SpyLLM:
    """Stub determinístico de Gemini que registra todas as chamadas e prompts recebidos."""

    model_name = "gemini-4-hydrogen"

    def __init__(self, reply: str = "Explicação baseada nas evidências aprovadas [EV-1].") -> None:
        self.calls: list[str] = []
        self._reply = reply

    def generate_text(
        self,
        *,
        system_instruction: str,
        user_prompt: str,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> str:
        del system_instruction, temperature, max_output_tokens
        self.calls.append(user_prompt)
        return self._reply


class _StubIndexingStatus(IndexingService):
    def __init__(self) -> None:
        pass

    def get_index_status(self, *, collection_name: str | None = None) -> IndexStatusReport:
        del collection_name
        return IndexStatusReport(
            collection_name="test_guidance_col",
            configured_embedding_model="spy-deterministic-embed-256d",
            configured_embedding_dimension=256,
            collection_embedding_model="spy-deterministic-embed-256d",
            collection_embedding_dimension=256,
            chroma_available=True,
            chroma_vector_count=3,
            total_sqlite_chunks=3,
            total_documents=1,
            total_versions=1,
            indexed_versions_count=1,
            version_summaries=(
                IndexVersionSummary(
                    document_id="anac_rbac_43",
                    version_id="anac_rbac_43_emd_05_5026d2da",
                    regulation_number="43",
                    amendment_number="05",
                    version_status="current",
                    sqlite_chunk_count=3,
                    indexed_chunk_count=3,
                    is_indexed=True,
                    is_stale=False,
                ),
            ),
        )


def _build_full_ssot_telegram_stack(
    tmp_path: Path,
    *,
    approve_rbac43: bool = True,
    public_access: bool = True,
    admins: list[int] | None = None,
    clock: Any = None,
    ttl_seconds: float = 600.0,
) -> tuple[
    TelegramCommandRouter,
    _SpyEmbeddingPort,
    _SpyLLM,
    SqliteDocumentRepository,
    UserConversationStore,
]:
    """Monta stack real (SQLite SSOT + HybridNormativeRetriever + ChatService + TelegramRouter)."""
    repo = SqliteDocumentRepository(tmp_path / "catalog.sqlite3")
    repo.initialize()
    catalog = CatalogService(repo)
    validation = ValidationService(ValidationPolicy.STRICT_ALL_PAGES)

    sec_43_9_root = (
        "43.9 Conteúdo, forma e disposição de registros de manutenção\n"
        "(a) Cada pessoa que executar manutenção deve efetuar o registro correspondente."
    )
    sec_43_10_a_hdr = "(a) Definições usadas nesta seção:"
    sec_43_10_a1 = (
        "(1) Peça com limite de vida significa qualquer peça para a qual esteja especificado "
        "um limite obrigatório de substituição."
    )
    sec_43_10_a2 = (
        "(2) Situação de vida significa o número acumulado de horas, ciclos ou qualquer outro "
        "limite de utilização de uma peça com limite de vida."
    )
    sec_43_10_b_hdr = "(b) Remoção temporária de peças com limite de vida."
    sec_43_10_root = "\n".join(
        [
            "43.10 Disposição de peças com limite de vida",
            sec_43_10_a_hdr,
            sec_43_10_a1,
            sec_43_10_a2,
            sec_43_10_b_hdr,
        ]
    )
    page_text = f"{sec_43_9_root}\n\n{sec_43_10_root}"
    now = datetime.now(UTC)
    src_hash = _sha("synth_rbac43_conversational_guidance_test")
    vid = "anac_rbac_43_emd_05_5026d2da"
    doc = RegulatoryDocument(
        document_id="anac_rbac_43",
        authority="ANAC",
        document_type=DocumentType.RBAC,
        regulation_number="43",
        document_title="Manutenção, Manutenção Preventiva, Reconstrução e Alteração",
        amendment_number="05",
        version_status=VersionStatus.CURRENT,
        file_hash=src_hash,
        imported_at=now,
    )
    ver = DocumentVersion(
        version_id=vid,
        document_id=doc.document_id,
        amendment_number="05",
        version_status=VersionStatus.CURRENT,
        source_hash=src_hash,
        page_count=1,
        imported_at=now,
    )
    page_status = (
        PageValidationStatus.APPROVED_FOR_LITERAL_TRANSCRIPTION
        if approve_rbac43
        else PageValidationStatus.PENDING_REVIEW
    )
    pages = [
        RegulatoryPage(
            page_id=f"{vid}::p0001",
            version_id=vid,
            pdf_page_number=1,
            printed_page_number="9/32",
            original_text=page_text,
            normalized_text=page_text,
            extraction_method=ExtractionMethod.NATIVE_PDF,
            validation_status=page_status,
            content_hash=compute_page_content_hash(page_text, ()),
        )
    ]
    units = AnacRbacStructureStrategy().parse(
        version_id=vid,
        regulation_number="43",
        pages=pages,
    )
    repo.save_version_ingestion(
        document=doc,
        version=ver,
        pages=pages,
        units=units,
        tables=(),
        extraction_run=ExtractionRun(
            run_id="run_guidance_test",
            document_id=doc.document_id,
            version_id=vid,
            started_at=now,
            finished_at=now,
            status="succeeded",
            extraction_version="2.0.0",
            source_hash=src_hash,
            pages_extracted=1,
            units_created=len(units),
        ),
    )

    embedder = _SpyEmbeddingPort()
    vstore = _InMemoryVectorStore()

    if approve_rbac43:
        indexing = IndexingService(
            repository=repo,
            catalog_service=catalog,
            chunker=StructuralNormativeChunker(),
            vector_store=vstore,  # type: ignore[arg-type]
            embedding_port=embedder,  # type: ignore[arg-type]
            default_collection_name="test_guidance_col",
            validation_service=validation,
        )
        indexing.index_document(document_id="anac_rbac_43", version_id=vid)

    retriever = HybridNormativeRetriever(
        repository=repo,
        vector_store=vstore,  # type: ignore[arg-type]
        embedding_port=embedder,  # type: ignore[arg-type]
        collection_name="test_guidance_col",
        validation_service=validation,
        max_vector_distance=0.80,
    )
    llm = _SpyLLM()
    audit = JsonlAuditAdapter(logs_dir=tmp_path / "logs")
    chat = NormativeChatService(retriever=retriever, llm=llm, audit_port=audit)
    conv_store = (
        UserConversationStore(ttl_seconds=ttl_seconds, clock=clock)
        if clock is not None
        else UserConversationStore(ttl_seconds=ttl_seconds)
    )
    status_uc = CheckSystemStatusUseCase(
        vector_store=vstore,  # type: ignore[arg-type]
        documents_dir=tmp_path / "docs",
        audit_logs_dir=tmp_path / "logs",
    )
    router = TelegramCommandRouter(
        access_control=AccessControl(
            admins=admins if admins is not None else [1001],
            users=[],
            public_access=public_access,
            audit_port=audit,
        ),
        rate_limiter=UserRateLimiter(requests_per_minute=100, burst_size=100),
        splitter=MessageSplitter(max_message_length=4096),
        status_use_case=status_uc,
        catalog_service=catalog,
        indexing_service=_StubIndexingStatus(),
        chat_service=chat,
        runtime_info=TelegramRuntimeInfo(
            environment="test",
            generation_model="gemini-4-hydrogen",
            embedding_model="spy-deterministic-embed-256d",
            embedding_dimensions=256,
            collection_name="test_guidance_col",
            allow_paid_models=False,
            llm_configured=True,
        ),
        audit_port=audit,
        conversation_store=conv_store,
    )
    return router, embedder, llm, repo, conv_store


@pytest.mark.unit
def test_conversational_intent_classifier_matrix_preserves_literal_explanatory_and_vague() -> None:
    """Valida a classificação determinística sem falsos positivos em referências ou fora de base."""
    # Consultas literais (incluindo "conteúdo literal")
    for literal_q in (
        "Transcreva a seção 43.10.",
        "Mostre o texto de 43.10(a)(1).",
        "Qual é o conteúdo literal de 43.13?",
        "Transcreva literalmente os itens 43.10(a)(1) e 43.10(a)(99).",
        "RBAC 43.999",
    ):
        dec = classify_conversational_intent(literal_q)
        assert dec.intent == ConversationIntent.LITERAL_QUERY, literal_q
        assert dec.generate_explanation is False

    # Consultas explicativas com tema ou referência concreta
    for exp_q in (
        "Explique a seção 43.10.",
        "Pode explicar a seção 43.10?",
        "Quem pode executar manutenção no RBAC 43?",
        "Qual é a diferença entre manutenção preventiva e alteração?",
        "Trabalho com manutenção: quem pode aprovar o retorno ao serviço?",
    ):
        dec = classify_conversational_intent(exp_q)
        assert dec.intent == ConversationIntent.EXPLANATORY_QUERY, exp_q
        assert dec.generate_explanation is True

    # Perguntas vagas / orientação -> NEEDS_CLARIFICATION (zero recuperação)
    for vague_q in (
        "Qual seção se usa para o meu caso? Trabalho com aeronavegabilidade e preciso saber "
        "das coisas.",
        "Trabalho com aeronavegabilidade. Qual seção devo consultar?",
        "Oi",
        "Bom dia, pode me ajudar?",
        "Preciso de ajuda",
        "Não sei o que perguntar",
        "Me fale sobre o RBAC 43",
        "aeronavegabilidade",
        "O que significa essa exigência?",
        "Pode explicar essa seção?",
        "Qual é a diferença entre esses dois requisitos?",
    ):
        dec = classify_conversational_intent(vague_q)
        assert dec.intent == ConversationIntent.NEEDS_CLARIFICATION, vague_q
        assert dec.generate_explanation is False
        assert dec.guidance_message is not None

    # Perguntas de ranking / panorama geral multi-RBAC -> OUT_OF_CATALOG (0 embeddings, 0 Gemini)
    for ranking_q in (
        "Quais RBACs são mais utilizadas no setor aeronáutico?",
        "Qual é o RBAC mais importante da ANAC?",
        "Quais RBACs existem na ANAC?",
        "Liste todos os RBACs da ANAC",
    ):
        dec = classify_conversational_intent(ranking_q)
        assert dec.intent == ConversationIntent.OUT_OF_CATALOG, ranking_q
        assert dec.generate_explanation is False

    # Perguntas concretas fora da base NÃO são desviadas para orientação — vão ao gate fail-closed
    for oos_q in (
        "Qual a receita de bolo de chocolate?",
        "Qual a validade da licença de piloto comercial?",
    ):
        dec = classify_conversational_intent(oos_q)
        assert dec.intent != ConversationIntent.NEEDS_CLARIFICATION, oos_q


@pytest.mark.unit
async def test_conversational_guidance_end_to_end_cases_1_to_10(tmp_path: Path) -> None:
    """Valida todos os 10 casos obrigatórios ponta a ponta com SSOT real e PR #21 intacto."""
    now_holder = [1000.0]

    def _clock() -> float:
        return now_holder[0]

    router, embedder, llm, repo, _store = _build_full_ssot_telegram_stack(
        tmp_path,
        clock=_clock,
        ttl_seconds=600.0,
    )
    expected_a1 = (
        "(1) Peça com limite de vida significa qualquer peça para a qual esteja especificado "
        "um limite obrigatório de substituição."
    )
    expected_a2 = (
        "(2) Situação de vida significa o número acumulado de horas, ciclos ou qualquer outro "
        "limite de utilização de uma peça com limite de vida."
    )

    try:
        # --- CASO 9: Comandos /start e /ajuda (e /help) ---
        assert "/ajuda" in SUPPORTED_COMMANDS
        r_start = await router.handle_message(user_id=5001, text="/start", update_id=1)
        assert "Perfil ativo: USER" in r_start.messages[0]
        assert "Consulta literal" in r_start.messages[0]
        assert "Pergunta explicativa" in r_start.messages[0]
        assert "Orientação" in r_start.messages[0]
        assert "/ajuda" in r_start.messages[0]

        r_ajuda = await router.handle_message(user_id=5001, text="/ajuda", update_id=2)
        r_help = await router.handle_message(user_id=5001, text="/help", update_id=3)
        assert r_ajuda.messages == r_help.messages
        help_text = r_ajuda.messages[0]
        assert "Consultas literais" in help_text
        assert "Perguntas explicativas" in help_text
        assert "Pedir orientação" in help_text
        assert "43.10(a)(1)" in help_text
        assert "/rbac" in help_text and "/explicar" in help_text
        assert embedder.query_calls == []
        assert llm.calls == []

        # --- CASO 1: Consulta literal específica (granularidade exata PR #21, 0 Gemini) ---
        r_lit = await router.handle_message(
            user_id=5001,
            text="Mostre o texto de 43.10(a)(1).",
            update_id=4,
        )
        joined_lit = "".join(strip_pagination_header(m) for m in r_lit.messages)
        assert extract_literal_excerpts_from_text(joined_lit) == (expected_a1,)
        assert "Situação de vida" not in joined_lit
        assert "=== EXPLICAÇÃO (GERADA POR IA) ===" not in joined_lit
        assert EXPLANATION_NOT_OFFICIAL_NOTICE not in joined_lit
        assert llm.calls == []

        # --- CASO 2 & 8: Follow-up dêitico ("O que significa essa exigência?") ---
        # Explica com base no SSOT aprovado SEM repetir o texto literal integral já exibido!
        r_deictic = await router.handle_message(
            user_id=5001,
            text="O que significa essa exigência?",
            update_id=5,
        )
        joined_deictic = "".join(strip_pagination_header(m) for m in r_deictic.messages)
        assert len(llm.calls) == 1
        assert "<<<INÍCIO_TRANSCRIÇÃO_LITERAL>>>" not in joined_deictic
        assert "Peça com limite de vida significa" not in joined_deictic
        assert "=== EXPLICAÇÃO (GERADA POR IA) ===" in joined_deictic
        assert "não constitui transcrição oficial" in joined_deictic
        assert "=== REFERÊNCIA ===" in joined_deictic
        assert "Seção 43.10 / Parágrafo (a) / Item (1)" in joined_deictic
        assert "/rbac 43.10(a)(1)" in joined_deictic

        # /explicar continua entregando texto oficial literal + explicação separada com nota!
        r_cmd_exp = await router.handle_message(
            user_id=5001,
            text="/explicar 43.10(a)(1)",
            update_id=55,
        )
        joined_cmd_exp = "".join(strip_pagination_header(m) for m in r_cmd_exp.messages)
        assert len(llm.calls) == 2
        assert extract_literal_excerpts_from_text(joined_cmd_exp) == (expected_a1,)
        assert "=== EXPLICAÇÃO (GERADA POR IA) ===" in joined_cmd_exp
        assert EXPLANATION_NOT_OFFICIAL_NOTICE in joined_cmd_exp
        exp_section = joined_cmd_exp.split("=== EXPLICAÇÃO (GERADA POR IA) ===", 1)[1]
        assert "<<<INÍCIO_TRANSCRIÇÃO_LITERAL>>>" not in exp_section

        # --- Ranking / panorama multi-RBAC -> limitação honesta do catálogo (0 emb, 0 Gemini) ---
        emb_before_rank = len(embedder.query_calls)
        llm_before_rank = len(llm.calls)
        r_rank = await router.handle_message(
            user_id=5001,
            text="Quais RBACs são mais utilizadas no setor aeronáutico?",
            update_id=56,
        )
        rank_text = "\n".join(r_rank.messages)
        assert "Não consigo ranquear" in rank_text
        assert "ANAC RBAC 43" in rank_text
        assert "43.1-I" not in rank_text
        assert "<<<INÍCIO_TRANSCRIÇÃO_LITERAL>>>" not in rank_text
        assert len(embedder.query_calls) == emb_before_rank
        assert len(llm.calls) == llm_before_rank

        # --- CASO 3: Pergunta vaga -> pede esclarecimento com 0 embeddings e 0 Gemini ---
        embed_before_vague = len(embedder.query_calls)
        llm_before_vague = len(llm.calls)
        r_vague = await router.handle_message(
            user_id=5002,
            text=(
                "Qual seção se usa para o meu caso? Trabalho com aeronavegabilidade "
                "e preciso saber das coisas."
            ),
            update_id=6,
        )
        vague_reply = "\n".join(r_vague.messages)
        assert "Execução de manutenção ou alteração" in vague_reply
        assert "Inspeções" in vague_reply
        assert "Registros e documentação de manutenção" in vague_reply
        assert "Aprovação para retorno ao serviço" in vague_reply
        assert "Controle de componentes com limite de vida" in vague_reply
        assert "não regula integralmente" in vague_reply
        assert "<<<INÍCIO_TRANSCRIÇÃO_LITERAL>>>" not in vague_reply
        assert len(embedder.query_calls) == embed_before_vague
        assert len(llm.calls) == llm_before_vague

        # --- Opções 1 a 5: todas fazem sub-orientação curta SEM busca ampla (0 emb, 0 Gemini) ---
        for opt_idx, (opt_num, expected_cue) in enumerate(
            (
                ("1", "execução de manutenção ou alteração"),
                ("2", "inspeções"),
                ("3", "registros e documentação de manutenção"),
                ("4", "aprovação para retorno ao serviço"),
                ("5", "componentes com limite de vida"),
            ),
            start=60,
        ):
            opt_uid = 5200 + opt_idx
            await router.handle_message(
                user_id=opt_uid,
                text="Trabalho com aeronavegabilidade. Qual seção devo consultar?",
                update_id=opt_idx * 10,
            )
            r_opt = await router.handle_message(
                user_id=opt_uid,
                text=opt_num,
                update_id=opt_idx * 10 + 1,
            )
            opt_reply = r_opt.messages[0]
            assert expected_cue in opt_reply.lower()
            assert "<<<INÍCIO_TRANSCRIÇÃO_LITERAL>>>" not in opt_reply
        assert len(embedder.query_calls) == embed_before_vague
        assert len(llm.calls) == llm_before_vague

        # --- CASO 10: Isolamento de contexto entre usuários ---
        # Usuário 5002 está aguardando esclarecimento, mas Usuário 5003 envia "5" ou "essa seção":
        # Usuário 5003 JAMAIS herda o contexto de 5002 ou 5001!
        embed_before_u3 = len(embedder.query_calls)
        llm_before_u3 = len(llm.calls)
        r_u3_digit = await router.handle_message(user_id=5003, text="5", update_id=7)
        assert "qual atividade" in r_u3_digit.messages[0].lower()
        r_u3_deictic = await router.handle_message(
            user_id=5003,
            text="Pode explicar essa seção?",
            update_id=8,
        )
        assert (
            "não identifiquei" in r_u3_deictic.messages[0].lower()
            or "descreva" in r_u3_deictic.messages[0].lower()
        )
        assert len(embedder.query_calls) == embed_before_u3
        assert len(llm.calls) == llm_before_u3

        # --- CASO 4a: Usuário 5002 escolhe opção 5 (0 emb/0 LLM) e depois detalha o ponto ---
        r_opt5_narrow = await router.handle_message(user_id=5002, text="5", update_id=9)
        assert "componentes com limite de vida" in r_opt5_narrow.messages[0].lower()
        assert "/rbac 43.10" in r_opt5_narrow.messages[0]
        assert "<<<INÍCIO_TRANSCRIÇÃO_LITERAL>>>" not in r_opt5_narrow.messages[0]
        assert len(embedder.query_calls) == embed_before_u3
        assert len(llm.calls) == llm_before_u3

        # Agora usuário 5002 fornece detalhe -> explicação curta + referência (sem textão literal!)
        r_followup = await router.handle_message(
            user_id=5002,
            text="Definições de peça com limite de vida",
            update_id=90,
        )
        joined_followup = "".join(strip_pagination_header(m) for m in r_followup.messages)
        assert "=== EXPLICAÇÃO (GERADA POR IA) ===" in joined_followup
        assert "não constitui transcrição oficial" in joined_followup
        assert "=== REFERÊNCIA ===" in joined_followup
        assert "43.10" in joined_followup
        assert "<<<INÍCIO_TRANSCRIÇÃO_LITERAL>>>" not in joined_followup
        assert "/rbac 43.10" in joined_followup
        assert len(llm.calls) == llm_before_u3 + 1

        # --- CASO 4b: Opção 4 com direcionamento específico + opção 6 (sem e com descrição) ---
        await router.handle_message(
            user_id=5007,
            text="Trabalho com aeronavegabilidade. Qual seção devo consultar?",
            update_id=91,
        )
        r_opt4_narrow = await router.handle_message(user_id=5007, text="4", update_id=92)
        assert "aprovação para retorno ao serviço" in r_opt4_narrow.messages[0].lower()
        assert "/rbac 43.5" in r_opt4_narrow.messages[0]
        assert len(llm.calls) == llm_before_u3 + 1

        r_opt6_inline = await router.handle_message(
            user_id=5007,
            text="6 - registros de manutenção obrigatórios",
            update_id=93,
        )
        joined_opt6_inline = "".join(strip_pagination_header(m) for m in r_opt6_inline.messages)
        assert "43.9" in joined_opt6_inline
        assert "=== EXPLICAÇÃO (GERADA POR IA) ===" in joined_opt6_inline
        assert "<<<INÍCIO_TRANSCRIÇÃO_LITERAL>>>" not in joined_opt6_inline
        assert len(llm.calls) == llm_before_u3 + 2

        r_vague2 = await router.handle_message(
            user_id=5004,
            text="Trabalho com aeronavegabilidade. Qual seção devo consultar?",
            update_id=10,
        )
        assert "1." in r_vague2.messages[0]
        # Opção 6 sem detalhe -> pede frase breve sem chamar embeddings nem Gemini
        emb_before_6 = len(embedder.query_calls)
        llm_before_6 = len(llm.calls)
        r_opt6 = await router.handle_message(user_id=5004, text="6", update_id=11)
        assert "Descreva em uma frase" in r_opt6.messages[0]
        assert len(embedder.query_calls) == emb_before_6
        assert len(llm.calls) == llm_before_6

        # Agora usuário 5004 responde com o tema concreto -> explicação curta + referência 43.9
        r_opt6_detail = await router.handle_message(
            user_id=5004,
            text="Registros e documentação de manutenção",
            update_id=12,
        )
        joined_opt6 = "".join(strip_pagination_header(m) for m in r_opt6_detail.messages)
        assert "43.9" in joined_opt6
        assert "=== EXPLICAÇÃO (GERADA POR IA) ===" in joined_opt6
        assert "<<<INÍCIO_TRANSCRIÇÃO_LITERAL>>>" not in joined_opt6
        assert len(llm.calls) == llm_before_6 + 1

        # TTL expirado: se passar mais de 600s, o contexto pendente é descartado!
        await router.handle_message(
            user_id=5005,
            text="Trabalho com aeronavegabilidade. Qual seção devo consultar?",
            update_id=13,
        )
        now_holder[0] += 601.0
        emb_after_ttl = len(embedder.query_calls)
        llm_after_ttl = len(llm.calls)
        r_expired = await router.handle_message(user_id=5005, text="5", update_id=14)
        assert "<<<INÍCIO_TRANSCRIÇÃO_LITERAL>>>" not in r_expired.messages[0]
        assert len(embedder.query_calls) == emb_after_ttl
        assert len(llm.calls) == llm_after_ttl

        # --- CASO 5: Pergunta fora dos documentos aprovados -> fail-closed, 0 Gemini ---
        llm_before_oos = len(llm.calls)
        r_oos = await router.handle_message(
            user_id=5006,
            text="Qual é a receita de bolo de chocolate com cobertura de morango?",
            update_id=15,
        )
        oos_text = "\n".join(r_oos.messages)
        assert "Não foram encontradas evidências normativas suficientes" in oos_text
        assert "<<<INÍCIO_TRANSCRIÇÃO_LITERAL>>>" not in oos_text
        assert len(llm.calls) == llm_before_oos

        # --- CASO 6: Referência estrutural inexistente -> bloqueio PR #21, 0 Gemini ---
        llm_before_missing = len(llm.calls)
        r_missing = await router.handle_message(
            user_id=5006,
            text="Transcreva literalmente o item 43.10(a)(99).",
            update_id=16,
        )
        missing_text = "\n".join(r_missing.messages)
        assert "43.10(a)(99)" in missing_text
        assert "<<<INÍCIO_TRANSCRIÇÃO_LITERAL>>>" not in missing_text
        assert "Peça com limite de vida" not in missing_text
        assert len(llm.calls) == llm_before_missing

        # --- CASO 7: Múltiplas referências sendo uma inválida -> bloqueio integral, 0 Gemini ---
        llm_before_multi = len(llm.calls)
        r_multi_bad = await router.handle_message(
            user_id=5006,
            text="Transcreva literalmente os itens 43.10(a)(1) e 43.10(a)(99).",
            update_id=17,
        )
        multi_bad_text = "\n".join(r_multi_bad.messages)
        assert "43.10(a)(99)" in multi_bad_text
        assert "<<<INÍCIO_TRANSCRIÇÃO_LITERAL>>>" not in multi_bad_text
        assert "Peça com limite de vida" not in multi_bad_text
        assert len(llm.calls) == llm_before_multi

        # Confirma que 2 referências válidas continuam retornando ambas literalmente (PR #21)
        r_multi_ok = await router.handle_message(
            user_id=5006,
            text="Transcreva literalmente os itens 43.10(a)(1) e 43.10(a)(2).",
            update_id=18,
        )
        joined_multi_ok = "".join(strip_pagination_header(m) for m in r_multi_ok.messages)
        assert extract_literal_excerpts_from_text(joined_multi_ok) == (expected_a1, expected_a2)
        assert len(llm.calls) == llm_before_multi

        # Comparação dêitica com 2 referências recém-consultadas pelo mesmo usuário 5006:
        # explica sem repetir as duas transcrições integrais, preservando ambas as referências!
        r_cmp = await router.handle_message(
            user_id=5006,
            text="Qual é a diferença entre esses dois requisitos?",
            update_id=19,
        )
        joined_cmp = "".join(strip_pagination_header(m) for m in r_cmp.messages)
        assert "<<<INÍCIO_TRANSCRIÇÃO_LITERAL>>>" not in joined_cmp
        assert "=== EXPLICAÇÃO (GERADA POR IA) ===" in joined_cmp
        assert "não constitui transcrição oficial" in joined_cmp
        assert "Seção 43.10 / Parágrafo (a) / Item (1)" in joined_cmp
        assert "Seção 43.10 / Parágrafo (a) / Item (2)" in joined_cmp
        assert len(llm.calls) == llm_before_multi + 1
    finally:
        repo.close()


@pytest.mark.unit
async def test_unapproved_document_never_reaches_gemini_even_after_clarification(
    tmp_path: Path,
) -> None:
    """Garante que documento pendente de aprovação nunca chega ao Gemini após esclarecimento."""
    router, _embedder, llm, repo, _store = _build_full_ssot_telegram_stack(
        tmp_path,
        approve_rbac43=False,
    )
    try:
        r_vague = await router.handle_message(
            user_id=6001,
            text="Trabalho com aeronavegabilidade. Qual seção devo consultar?",
            update_id=101,
        )
        assert "5." in r_vague.messages[0]

        r_opt5 = await router.handle_message(user_id=6001, text="5", update_id=102)
        assert "componentes com limite de vida" in r_opt5.messages[0].lower()
        assert llm.calls == []

        r_reply = await router.handle_message(
            user_id=6001,
            text="Definições de peça com limite de vida",
            update_id=103,
        )
        reply_text = "\n".join(r_reply.messages)
        assert "Peça com limite de vida" not in reply_text
        assert "<<<INÍCIO_TRANSCRIÇÃO_LITERAL>>>" not in reply_text
        assert llm.calls == []
    finally:
        repo.close()


@pytest.mark.unit
async def test_literal_queries_work_when_gemini_embeddings_and_generation_fail(
    tmp_path: Path,
) -> None:
    """Caso 9 e B: consultas literais funcionam via SSOT/FTS5 com Gemini em 429/503/timeout."""
    from google.genai import errors as genai_errors

    from rbac_assistant.infrastructure.llm.gemini_client import (
        GeminiClientAdapter,
        GeminiEmbeddingAdapter,
    )
    from tests.unit.test_gemini_client import _FakeGenAISDKClient

    base_router, _spy_emb, _spy_llm, repo, _store = _build_full_ssot_telegram_stack(tmp_path)
    expected_a1 = (
        "(1) Peça com limite de vida significa qualquer peça para a qual esteja especificado "
        "um limite obrigatório de substituição."
    )

    try:
        # Substitui retriever/chat para usar GeminiClientAdapter real sobre SDK falhando em 429/503
        failing_sdk = _FakeGenAISDKClient()
        failing_sdk.models.embed_side_effects = [
            genai_errors.APIError(429, {"error": {"message": "Embedding quota exceeded"}}),
            TimeoutError("embedding socket timeout"),
            genai_errors.APIError(503, {"error": {"message": "Embedding unavailable"}}),
        ]
        failing_sdk.models.generate_side_effects = [
            genai_errors.APIError(429, {"error": {"message": "Generation quota exceeded"}}),
        ]
        gemini_adapter = GeminiClientAdapter(
            max_retries=3,
            generation_max_retries=0,
            query_embedding_max_retries=0,
            retry_min_wait_seconds=0.0,
            retry_max_wait_seconds=0.0,
            sdk_client=failing_sdk,
        )
        emb_adapter = GeminiEmbeddingAdapter(gemini_adapter)

        resilient_retriever = HybridNormativeRetriever(
            repository=repo,
            vector_store=_InMemoryVectorStore(),  # type: ignore[arg-type]
            embedding_port=emb_adapter,  # type: ignore[arg-type]
            collection_name="test_guidance_col",
            validation_service=ValidationService(ValidationPolicy.STRICT_ALL_PAGES),
        )
        resilient_chat = NormativeChatService(
            retriever=resilient_retriever,
            llm=gemini_adapter,
        )
        base_router._chat_service = resilient_chat

        # 1. Consulta literal livre com embedding em 429 -> sucesso via estrutural+FTS5, 0 geração!
        r_lit1 = await base_router.handle_message(
            user_id=7001,
            text="Transcreva literalmente o item 43.10(a)(1).",
            update_id=301,
        )
        joined1 = "".join(strip_pagination_header(m) for m in r_lit1.messages)
        assert extract_literal_excerpts_from_text(joined1) == (expected_a1,)
        assert len(failing_sdk.models.embed_calls) == 1
        assert len(failing_sdk.models.generate_calls) == 0

        # 2. /rbac com embedding em TimeoutError -> sucesso exato, 1 tentativa embed, 0 geração!
        r_lit2 = await base_router.handle_message(
            user_id=7001,
            text="/rbac 43.10(a)(1)",
            update_id=302,
        )
        joined2 = "".join(strip_pagination_header(m) for m in r_lit2.messages)
        assert extract_literal_excerpts_from_text(joined2) == (expected_a1,)
        assert len(failing_sdk.models.embed_calls) == 2
        assert len(failing_sdk.models.generate_calls) == 0

        # 3. Follow-up dêitico com geração em 429 -> exatamente 1 tentativa, sem repetir literal!
        r_deictic_429 = await base_router.handle_message(
            user_id=7001,
            text="O que significa essa exigência?",
            update_id=303,
        )
        deictic_429_text = "\n".join(r_deictic_429.messages)
        assert len(failing_sdk.models.embed_calls) == 3
        assert len(failing_sdk.models.generate_calls) == 1
        assert "temporariamente indisponível" in deictic_429_text.lower()
        assert "<<<INÍCIO_TRANSCRIÇÃO_LITERAL>>>" not in deictic_429_text
        assert "43.10" in deictic_429_text
    finally:
        repo.close()
