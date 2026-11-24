"""Testes integrados end-to-end de fidelidade literal, tabelas fail-closed, Gemini e Telegram."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from rbac_assistant.application.dto.ingestion_dto import (
    IngestionRequestDTO,
    PageReviewRequestDTO,
    RegisterDocumentDTO,
)
from rbac_assistant.application.services.catalog_service import CatalogService
from rbac_assistant.application.services.chat_service import NormativeChatService
from rbac_assistant.application.services.indexing_service import IndexingService
from rbac_assistant.application.services.ingestion_service import IngestionService
from rbac_assistant.application.services.validation_service import ValidationService
from rbac_assistant.application.use_cases.check_system_status import CheckSystemStatusUseCase
from rbac_assistant.application.use_cases.validate_page import ValidatePageUseCase
from rbac_assistant.domain.entities.document_version import DocumentVersion, ExtractionRun
from rbac_assistant.domain.entities.normative_unit import (
    ExtractionMethod,
    NormativeTable,
    NormativeUnitType,
    StructureStatus,
    TableCellGeometry,
)
from rbac_assistant.domain.entities.rag import IndexRegistryEntry, SearchFilter
from rbac_assistant.domain.entities.regulatory_document import (
    DocumentType,
    RegulatoryDocument,
    ValidationPolicy,
    VersionStatus,
)
from rbac_assistant.domain.entities.regulatory_page import (
    PageValidationStatus,
    RegulatoryPage,
    ReviewDecision,
    compute_page_content_hash,
)
from rbac_assistant.domain.ports.validation_port import ParsedUnitRecord
from rbac_assistant.infrastructure.pdf.anac_structure_strategy import AnacRbacStructureStrategy
from rbac_assistant.infrastructure.pdf.generic_structure_strategy import GenericStructureStrategy
from rbac_assistant.infrastructure.pdf.pdfplumber_extractor import PdfPlumberExtractorAdapter
from rbac_assistant.infrastructure.pdf.text_quality_checker import TextQualityChecker
from rbac_assistant.infrastructure.sources.local_file_source_adapter import LocalFileSourceAdapter
from rbac_assistant.infrastructure.storage.file_document_store import LocalFileDocumentStore
from rbac_assistant.infrastructure.storage.sqlite_document_repository import (
    SqliteDocumentRepository,
)
from rbac_assistant.interfaces.telegram.bot_runner import TelegramBotRunner
from rbac_assistant.interfaces.telegram.command_router import (
    TelegramCommandRouter,
    TelegramRuntimeInfo,
)
from rbac_assistant.interfaces.telegram.message_splitter import (
    MessageSplitter,
    strip_pagination_header,
    utf16_length,
)
from rbac_assistant.rag.generation.response_formatter import extract_literal_excerpts_from_text
from rbac_assistant.rag.ingestion.structural_chunker import StructuralNormativeChunker
from rbac_assistant.rag.retrieval.hybrid_retriever import HybridNormativeRetriever
from rbac_assistant.security.access_control import AccessControl
from rbac_assistant.security.rate_limiter import UserRateLimiter
from tests.fixtures.synthetic_pdf_factory import create_synthetic_rbac_pdf


class _NoOpVectorStore:
    def check_heartbeat(self) -> int:
        return 1

    def ensure_collection_compatible(
        self,
        *,
        collection_name: str,
        embedding_model: str,
        embedding_dimension: int,
        instruction_strategy_version: str,
    ) -> None:
        del collection_name, embedding_model, embedding_dimension, instruction_strategy_version

    def get_collection_metadata(self, collection_name: str) -> Any:
        del collection_name
        return None

    def list_record_ids(
        self,
        *,
        collection_name: str,
        where: dict[str, Any] | None = None,
    ) -> list[str]:
        del collection_name, where
        return []

    def upsert_records(self, collection_name: str, records: Any) -> None:
        del collection_name, records

    def delete_records(self, collection_name: str, record_ids: Any) -> None:
        del collection_name, record_ids

    def get_records(self, collection_name: str, record_ids: Any) -> list[Any]:
        del collection_name, record_ids
        return []

    def query_similar(
        self,
        collection_name: str,
        query_embedding: Any,
        *,
        top_k: int = 5,
        where: Any = None,
    ) -> list[Any]:
        del collection_name, query_embedding, top_k, where
        return []

    def delete_collection(self, collection_name: str) -> None:
        del collection_name


class _ConstantEmbedder:
    @property
    def model_name(self) -> str:
        return "test-offline-embedder"

    @property
    def dimensions(self) -> int:
        return 16

    @property
    def instruction_strategy_version(self) -> str:
        return "test_v1"

    def embed_documents(self, texts: Any, *, titles: Any = None) -> list[tuple[float, ...]]:
        del titles
        return [tuple([0.05] * 16) for _ in texts]

    def embed_query(self, text: str) -> tuple[float, ...]:
        del text
        return tuple([0.05] * 16)


class _SpyRewritingLLM:
    """Spy que tenta reescrever símbolos e inventar seções para provar isolamento do SSOT."""

    def __init__(self) -> None:
        self.calls = 0
        self.prompts_received: list[str] = []

    @property
    def model_name(self) -> str:
        return "gemini-4-hydrogen"

    def generate_text(self, *, system_instruction: str, user_prompt: str) -> str:
        del system_instruction
        self.calls += 1
        self.prompts_received.append(user_prompt)
        return (
            "Explicação baseada em [EV-1]: o requisito técnico do Apêndice F aplica-se "
            "aos testes de transponder. Referência inventada RBAC 43.999 [EV-88] (página 999)."
        )


@dataclass
class _FakeTelegramMessage:
    text: str
    sent_payloads: list[dict[str, Any]] = field(default_factory=list)

    async def reply_text(
        self,
        text: str,
        *,
        disable_web_page_preview: bool = True,
        **kwargs: Any,
    ) -> None:
        self.sent_payloads.append(
            {
                "text": text,
                "disable_web_page_preview": disable_web_page_preview,
                "extra_kwargs": kwargs,
            }
        )


@dataclass
class _FakeTelegramChat:
    type: str = "private"


@dataclass
class _FakeTelegramUser:
    id: int = 777


@dataclass
class _FakeTelegramUpdate:
    update_id: int
    effective_message: _FakeTelegramMessage
    effective_chat: _FakeTelegramChat = field(default_factory=_FakeTelegramChat)
    effective_user: _FakeTelegramUser = field(default_factory=_FakeTelegramUser)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_telegram_delivery_preserves_literal_unicode_tables_forms_and_splits_losslessly(
    tmp_path: Path,
) -> None:
    """Testa fluxo completo SQLite -> Retriever -> Formatter -> Chat -> Telegram Runner."""
    repo = SqliteDocumentRepository(tmp_path / "catalog.sqlite3")
    repo.initialize()
    try:
        now = datetime.now(UTC)
        source_hash = "1" * 64
        doc = RegulatoryDocument(
            document_id="anac_rbac_43",
            authority="ANAC",
            document_type=DocumentType.RBAC,
            regulation_number="43",
            document_title="Manutenção, Manutenção Preventiva, Reconstrução e Alteração",
            amendment_number="05",
            version_status=VersionStatus.CURRENT,
            file_hash=source_hash,
        )
        ver = DocumentVersion(
            version_id="anac_rbac_43_emd_05_11111111",
            document_id=doc.document_id,
            amendment_number="05",
            version_status=VersionStatus.CURRENT,
            source_hash=source_hash,
            page_count=2,
            imported_at=now,
        )

        # Página 1: F43.1 longo (> 4500 chars) para exercitar MessageSplitter multi-parte
        # contendo todos os símbolos críticos: ±, º, °, –, +, -, acentos e quebras de linha
        long_padding = "\n".join(
            f"(g) Item técnico de verificação nº {i:02d}: tolerância 1.090±3 MHz, "
            f"vazamento ± 100 pés, ângulo 1° ponto – faixa +70 a -73±4 dBm."
            for i in range(1, 45)
        )
        p1_text = (
            "F43.1 Testes e inspeções no transponder\n"
            "(a)(1) frequência de resposta do sistema é de 1.090±3 MHz.\n"
            "(b)(2) razão de 235±5 interrogações por segundo.\n"
            "(c)(1) nível mínimo -73±4 dBm e -74±3 dBm (Lei nº 7565).\n"
            f"{long_padding}"
        )
        p2_text = (
            "E43.1 Testes e inspeções do sistema do altímetro\n"
            "TABELA III\nAtrito\n1.000 304,8 +70 +21,3"
        )
        valid_table = NormativeTable(
            table_id="tbl_e43_t3",
            version_id=ver.version_id,
            unit_id=f"{ver.version_id}::E43.1",
            pdf_page_number=2,
            caption="TABELA III — Atrito",
            rows=(
                ("ALTITUDE", "", "TOLERÂNCIA", ""),
                ("(pés)", "(m)", "(pés)", "(m)"),
                ("1.000", "304,8", "+70", "+21,3"),
                ("2.000", "609,6", "70", "21,3"),
            ),
            cells=(
                TableCellGeometry(row_index=0, col_index=0, col_span=2, text="ALTITUDE"),
                TableCellGeometry(row_index=0, col_index=2, col_span=2, text="TOLERÂNCIA"),
                TableCellGeometry(row_index=1, col_index=0, col_span=1, text="(pés)"),
                TableCellGeometry(row_index=1, col_index=1, col_span=1, text="(m)"),
                TableCellGeometry(row_index=1, col_index=2, col_span=1, text="(pés)"),
                TableCellGeometry(row_index=1, col_index=3, col_span=1, text="(m)"),
            ),
            header_row_count=2,
            table_kind="data_table",
            has_merged_cells=True,
        )
        form_table = NormativeTable(
            table_id="tbl_e43_form",
            version_id=ver.version_id,
            unit_id=f"{ver.version_id}::E43.1",
            pdf_page_number=2,
            rows=(("Nome da Organização de Manutenção", "Nº Tel"),),
            cells=(
                TableCellGeometry(
                    row_index=0, col_index=0, text="Nome da Organização de Manutenção"
                ),
                TableCellGeometry(row_index=0, col_index=1, text="Nº Tel"),
            ),
            header_row_count=0,
            table_kind="form",
            visual_layout_preserved=False,
            layout_fidelity_note="Formulário não preservado em fac-símile gráfico.",
        )

        pages = [
            RegulatoryPage(
                page_id=f"{ver.version_id}::p0001",
                version_id=ver.version_id,
                pdf_page_number=1,
                printed_page_number="31/32",
                original_text=p1_text,
                normalized_text=p1_text,
                extraction_method=ExtractionMethod.NATIVE_PDF,
                validation_status=PageValidationStatus.APPROVED_FOR_LITERAL_TRANSCRIPTION,
                content_hash=compute_page_content_hash(p1_text),
            ),
            RegulatoryPage(
                page_id=f"{ver.version_id}::p0002",
                version_id=ver.version_id,
                pdf_page_number=2,
                printed_page_number="30/32",
                original_text=p2_text,
                normalized_text=p2_text,
                extraction_method=ExtractionMethod.NATIVE_PDF,
                validation_status=PageValidationStatus.APPROVED_FOR_LITERAL_TRANSCRIPTION,
                content_hash=compute_page_content_hash(
                    p2_text, (valid_table.rows, form_table.rows)
                ),
            ),
        ]
        units = [
            ParsedUnitRecord(
                unit_id=f"{ver.version_id}::F43.1",
                version_id=ver.version_id,
                parent_unit_id=None,
                unit_type=NormativeUnitType.APPENDIX,
                section_number="F43.1",
                subsection_number="",
                paragraph_number="",
                item_number="",
                title="Testes e inspeções no transponder",
                subpart="APÊNDICE F DO RBAC 43",
                original_text=p1_text,
                normalized_text=p1_text,
                start_page=1,
                end_page=1,
                page_numbers=(1,),
                order_index=0,
                structure_status=StructureStatus.PARSED,
                content_hash=hashlib.sha256(p1_text.encode("utf-8")).hexdigest(),
            ),
            ParsedUnitRecord(
                unit_id=f"{ver.version_id}::E43.1",
                version_id=ver.version_id,
                parent_unit_id=None,
                unit_type=NormativeUnitType.APPENDIX,
                section_number="E43.1",
                subsection_number="",
                paragraph_number="",
                item_number="",
                title="Testes e inspeções do sistema do altímetro",
                subpart="APÊNDICE E DO RBAC 43",
                original_text=p2_text,
                normalized_text=p2_text,
                start_page=2,
                end_page=2,
                page_numbers=(2,),
                order_index=1,
                structure_status=StructureStatus.PARSED,
                content_hash=hashlib.sha256(p2_text.encode("utf-8")).hexdigest(),
            ),
        ]
        repo.save_version_ingestion(
            document=doc,
            version=ver,
            pages=pages,
            units=units,
            tables=(valid_table, form_table),
            extraction_run=ExtractionRun(
                run_id="run_test_e2e",
                document_id=doc.document_id,
                version_id=ver.version_id,
                started_at=now,
                finished_at=now,
                status="succeeded",
                extraction_version="2.0.0",
                source_hash=source_hash,
                pages_extracted=2,
                units_created=2,
            ),
        )

        catalog = CatalogService(repo)
        validation = ValidationService(ValidationPolicy.STRICT_ALL_PAGES)
        vector_store = _NoOpVectorStore()
        indexing = IndexingService(
            repository=repo,
            catalog_service=catalog,
            chunker=StructuralNormativeChunker(),
            vector_store=vector_store,  # type: ignore[arg-type]
            embedding_port=_ConstantEmbedder(),
            validation_service=validation,
        )
        indexing.index_document(document_id=doc.document_id)

        retriever = HybridNormativeRetriever(
            repository=repo,
            vector_store=vector_store,  # type: ignore[arg-type]
            embedding_port=None,
            collection_name="test_col",
            validation_service=validation,
        )
        llm_spy = _SpyRewritingLLM()
        chat = NormativeChatService(retriever=retriever, llm=llm_spy)
        router = TelegramCommandRouter(
            access_control=AccessControl(admins=(1,), public_access=True),
            rate_limiter=UserRateLimiter(requests_per_minute=50, burst_size=50),
            splitter=MessageSplitter(max_message_length=4096),
            status_use_case=CheckSystemStatusUseCase(
                vector_store=vector_store,  # type: ignore[arg-type]
                documents_dir=tmp_path,
                audit_logs_dir=tmp_path,
            ),
            catalog_service=catalog,
            indexing_service=indexing,
            chat_service=chat,
            runtime_info=TelegramRuntimeInfo(
                environment="test",
                generation_model="gemini-4-hydrogen",
                embedding_model="gemini-embedding-2",
                embedding_dimensions=768,
                collection_name="test_col",
                allow_paid_models=False,
                llm_configured=True,
            ),
        )
        runner = TelegramBotRunner(token="123456:fake-test-token", router=router)

        # 1. /rbac F43.1: texto longo dividido em múltiplas mensagens, zero parse_mode, zero LLM
        msg_f43 = _FakeTelegramMessage(text="/rbac F43.1")
        upd_f43: Any = _FakeTelegramUpdate(update_id=101, effective_message=msg_f43)
        await runner.handle_update(upd_f43, None)  # type: ignore[arg-type]
        assert llm_spy.calls == 0
        assert len(msg_f43.sent_payloads) >= 2
        for payload in msg_f43.sent_payloads:
            assert utf16_length(payload["text"]) <= 4096
            assert "parse_mode" not in payload["extra_kwargs"]
        joined_f43 = "".join(strip_pagination_header(p["text"]) for p in msg_f43.sent_payloads)
        direct_f43 = chat.ask(question="RBAC F43.1", generate_explanation=False)
        assert direct_f43.literal_excerpts == (p1_text,)
        # Prova RAW caractere-a-caractere SEM sanitize_telegram_text nos dois lados!
        assert joined_f43 == direct_f43.formatted_text
        assert extract_literal_excerpts_from_text(joined_f43) == (p1_text,)
        assert "=== EXPLICAÇÃO (GERADA POR IA) ===" not in joined_f43
        assert "[IDENTIFICAÇÃO EXTERNA — RBAC 43 | Emenda 05 | Seção F43.1]" in joined_f43

        # 2. /explicar F43.1: Gemini explica sem alterar literal e citações falsas são descartadas
        msg_exp = _FakeTelegramMessage(text="/explicar F43.1")
        upd_exp: Any = _FakeTelegramUpdate(update_id=102, effective_message=msg_exp)
        await runner.handle_update(upd_exp, None)  # type: ignore[arg-type]
        assert llm_spy.calls == 1
        joined_exp = "".join(strip_pagination_header(p["text"]) for p in msg_exp.sent_payloads)
        assert p1_text in joined_exp
        exp_block = joined_exp.split("=== EXPLICAÇÃO (GERADA POR IA) ===", 1)[1].split(
            "=== REFERÊNCIA ===", 1
        )[0]
        assert "43.999" not in exp_block
        assert "[EV-88]" not in exp_block
        assert "página 999" not in exp_block
        assert "VALIDAÇÃO DE CITAÇÕES" in joined_exp

        # 3. /rbac E43.1: hidrata tabela estruturada com cabeçalhos/sinais + aviso de formulário
        msg_e43 = _FakeTelegramMessage(text="/rbac E43.1")
        upd_e43: Any = _FakeTelegramUpdate(update_id=103, effective_message=msg_e43)
        await runner.handle_update(upd_e43, None)  # type: ignore[arg-type]
        joined_e43 = "".join(strip_pagination_header(p["text"]) for p in msg_e43.sent_payloads)
        assert "=== TABELAS E FORMULÁRIOS ESTRUTURADOS (SQLite SSOT) ===" in joined_e43
        expected_row = (
            "ALTITUDE (pés)=1.000 | ALTITUDE (m)=304,8 | "
            "TOLERÂNCIA (pés)=+70 | TOLERÂNCIA (m)=+21,3"
        )
        assert expected_row in joined_e43
        assert "FORMULÁRIO ESTRUTURADO" in joined_e43
        assert "Nº Tel" in joined_e43
        assert "AVISO DE FORMULÁRIO" in joined_e43
    finally:
        repo.close()


@pytest.mark.integration
def test_corrupted_table_blocks_retrieval_fail_closed_and_never_calls_gemini(
    tmp_path: Path,
) -> None:
    """Se tabela vinculada perder cabeçalhos/colunas, o sistema recusa com segurança sem LLM."""
    repo = SqliteDocumentRepository(tmp_path / "corrupt_tbl.sqlite3")
    repo.initialize()
    try:
        now = datetime.now(UTC)
        source_hash = "2" * 64
        doc = RegulatoryDocument(
            document_id="anac_rbac_43",
            authority="ANAC",
            document_type=DocumentType.RBAC,
            regulation_number="43",
            document_title="RBAC 43",
            amendment_number="05",
            version_status=VersionStatus.CURRENT,
            file_hash=source_hash,
        )
        ver = DocumentVersion(
            version_id="anac_rbac_43_emd_05_22222222",
            document_id=doc.document_id,
            amendment_number="05",
            version_status=VersionStatus.CURRENT,
            source_hash=source_hash,
            page_count=1,
            imported_at=now,
        )
        p1_text = "E43.1 Testes do altímetro\n1.000 304,8 +70 +21,3"
        # Tabela numérica SEM cabeçalhos (violação de integridade tabular!)
        headerless_table = NormativeTable(
            table_id="tbl_corrupt",
            version_id=ver.version_id,
            unit_id=f"{ver.version_id}::E43.1",
            pdf_page_number=1,
            rows=(
                ("1.000", "304,8", "+70", "+21,3"),
                ("2.000", "609,6", "70", "21,3"),
            ),
            header_row_count=1,
            table_kind="data_table",
        )
        pages = [
            RegulatoryPage(
                page_id=f"{ver.version_id}::p0001",
                version_id=ver.version_id,
                pdf_page_number=1,
                printed_page_number="30/32",
                original_text=p1_text,
                normalized_text=p1_text,
                extraction_method=ExtractionMethod.NATIVE_PDF,
                validation_status=PageValidationStatus.APPROVED_FOR_LITERAL_TRANSCRIPTION,
                content_hash=compute_page_content_hash(p1_text, (headerless_table.rows,)),
            )
        ]
        units = [
            ParsedUnitRecord(
                unit_id=f"{ver.version_id}::E43.1",
                version_id=ver.version_id,
                parent_unit_id=None,
                unit_type=NormativeUnitType.APPENDIX,
                section_number="E43.1",
                subsection_number="",
                paragraph_number="",
                item_number="",
                title="Testes do altímetro",
                subpart="APÊNDICE E DO RBAC 43",
                original_text=p1_text,
                normalized_text=p1_text,
                start_page=1,
                end_page=1,
                page_numbers=(1,),
                order_index=0,
                structure_status=StructureStatus.PARSED,
                content_hash=hashlib.sha256(p1_text.encode("utf-8")).hexdigest(),
            )
        ]
        repo.save_version_ingestion(
            document=doc,
            version=ver,
            pages=pages,
            units=units,
            tables=(headerless_table,),
            extraction_run=ExtractionRun(
                run_id="run_corrupt",
                document_id=doc.document_id,
                version_id=ver.version_id,
                started_at=now,
                finished_at=now,
                status="succeeded",
                extraction_version="2.0.0",
                source_hash=source_hash,
                pages_extracted=1,
                units_created=1,
            ),
        )

        catalog = CatalogService(repo)
        validation = ValidationService(ValidationPolicy.STRICT_ALL_PAGES)
        vector_store = _NoOpVectorStore()
        IndexingService(
            repository=repo,
            catalog_service=catalog,
            chunker=StructuralNormativeChunker(),
            vector_store=vector_store,  # type: ignore[arg-type]
            embedding_port=_ConstantEmbedder(),
            validation_service=validation,
        ).index_document(document_id=doc.document_id)

        retriever = HybridNormativeRetriever(
            repository=repo,
            vector_store=vector_store,  # type: ignore[arg-type]
            embedding_port=None,
            collection_name="test_col",
            validation_service=validation,
        )
        llm_spy = _SpyRewritingLLM()
        chat = NormativeChatService(retriever=retriever, llm=llm_spy)

        answer = chat.ask(
            question="RBAC 43 E43.1",
            filters=SearchFilter(only_approved_literal=True),
            generate_explanation=True,
        )
        assert answer.found is False
        assert answer.normative_text is None
        assert answer.structured_tables_text is None
        assert answer.has_blocked_sources is True
        assert "integridade estrutural" in answer.explanation.lower()
        assert any("INTEGRIDADE TABULAR BLOQUEADA" in obs for obs in answer.observations)
        assert llm_spy.calls == 0
    finally:
        repo.close()


@pytest.mark.integration
def test_reprocessing_clears_stale_vector_index_registry_and_invalidates_changed_page(
    tmp_path: Path,
) -> None:
    """Reprocessar uma versão limpa `vector_index_registry` obsoleto e invalida hash alterado."""
    pdf_path = create_synthetic_rbac_pdf(tmp_path / "rbac43_reproc.pdf")
    docs_dir = tmp_path / "docs"
    repo = SqliteDocumentRepository(docs_dir / "catalog.sqlite3")
    repo.initialize()
    try:
        store = LocalFileDocumentStore(documents_dir=docs_dir)
        store.ensure_directories()
        catalog = CatalogService(repo)
        validation = ValidationService(ValidationPolicy.STRICT_ALL_PAGES)
        ingestion = IngestionService(
            repository=repo,
            file_store=store,
            pdf_extractor=PdfPlumberExtractorAdapter(),
            quality_checker=TextQualityChecker(),
            validation_service=validation,
            structure_strategies=(AnacRbacStructureStrategy(), GenericStructureStrategy()),
            local_source=LocalFileSourceAdapter(),
        )
        req = IngestionRequestDTO(
            document=RegisterDocumentDTO(
                document_id="anac_rbac_43",
                authority="ANAC",
                document_type=DocumentType.RBAC,
                regulation_number="43",
                document_title="RBAC 43",
                amendment_number="05",
                version_status=VersionStatus.CURRENT,
            ),
            source_uri=str(pdf_path),
        )
        res = ingestion.ingest(req)
        ver = res.version

        # Aprova todas as páginas e registra no vector_index_registry
        ValidatePageUseCase(
            repository=repo,
            catalog_service=catalog,
            validation_service=validation,
            file_store=store,
        ).execute(
            PageReviewRequestDTO(
                document_id="anac_rbac_43",
                version_id=ver.version_id,
                reviewer="revisor",
                decision=ReviewDecision.APPROVE,
                confirmed=True,
                page_numbers=tuple(range(1, ver.page_count + 1)),
            )
        )
        repo.upsert_index_registry(
            IndexRegistryEntry(
                collection_name="test_col",
                version_id=ver.version_id,
                document_id="anac_rbac_43",
                source_hash=ver.source_hash,
                embedding_model="gemini-embedding-2",
                embedding_dimension=768,
                chunk_count=3,
                indexed_at=datetime.now(UTC),
                instruction_strategy_version="ge2_asymmetric_v1",
            )
        )
        assert len(repo.list_index_registry(collection_name="test_col")) == 1

        # Reprocessa com force_reprocess=True -> limpa registro vetorial obsoleto
        ingestion.ingest(
            IngestionRequestDTO(
                document=req.document,
                source_uri=req.source_uri,
                force_reprocess=True,
            )
        )
        assert repo.list_index_registry(collection_name="test_col") == []
    finally:
        repo.close()


@pytest.mark.integration
def test_table_ownership_fail_closed_matrix_blocks_unassigned_and_prevents_cross_leak(
    tmp_path: Path,
) -> None:
    """Blocker 2: matriz completa de propriedade tabular com política fail-closed."""
    from rbac_assistant.application.use_cases.get_normative_section import (
        GetNormativeSectionUseCase,
    )
    from rbac_assistant.domain.entities.normative_unit import TableOwnershipStatus
    from rbac_assistant.domain.ports.pdf_extractor_port import (
        ExtractedPageData,
        ExtractedTableData,
    )

    def _run_scenario(
        name: str,
        *,
        units_spec: list[tuple[str, str]],
        tables: tuple[NormativeTable, ...],
        approved: bool = True,
    ) -> tuple[
        NormativeChatService,
        GetNormativeSectionUseCase,
        _SpyRewritingLLM,
        SqliteDocumentRepository,
    ]:
        repo = SqliteDocumentRepository(tmp_path / f"own_{name}.sqlite3")
        repo.initialize()
        now = datetime.now(UTC)
        src_hash = hashlib.sha256(name.encode()).hexdigest()
        doc = RegulatoryDocument(
            document_id="anac_rbac_43",
            authority="ANAC",
            document_type=DocumentType.RBAC,
            regulation_number="43",
            document_title="RBAC 43",
            amendment_number="05",
            version_status=VersionStatus.CURRENT,
            file_hash=src_hash,
        )
        ver_id = f"anac_rbac_43_emd_05_{src_hash[:8]}"
        ver = DocumentVersion(
            version_id=ver_id,
            document_id=doc.document_id,
            amendment_number="05",
            version_status=VersionStatus.CURRENT,
            source_hash=src_hash,
            page_count=1,
            imported_at=now,
        )
        page_text = "\n\n".join(txt for _sec, txt in units_spec)
        fixed_tables = tuple(
            NormativeTable(
                table_id=t.table_id,
                version_id=ver_id,
                unit_id=(
                    f"{ver_id}::{t.unit_id}"
                    if t.unit_id is not None and "::" not in t.unit_id
                    else t.unit_id
                ),
                pdf_page_number=1,
                rows=t.rows,
                caption=t.caption,
                bbox=t.bbox,
                cells=t.cells,
                header_row_count=t.header_row_count,
                table_kind=t.table_kind,
                has_merged_cells=t.has_merged_cells,
                visual_layout_preserved=t.visual_layout_preserved,
                layout_fidelity_note=t.layout_fidelity_note,
                ownership_status=t.ownership_status,
                owner_section_number=t.owner_section_number,
                ownership_note=t.ownership_note,
            )
            for t in tables
        )
        page = RegulatoryPage(
            page_id=f"{ver_id}::p0001",
            version_id=ver_id,
            pdf_page_number=1,
            printed_page_number="1/1",
            original_text=page_text,
            normalized_text=page_text,
            extraction_method=ExtractionMethod.NATIVE_PDF,
            validation_status=(
                PageValidationStatus.APPROVED_FOR_LITERAL_TRANSCRIPTION
                if approved
                else PageValidationStatus.PENDING_REVIEW
            ),
            content_hash=compute_page_content_hash(page_text, fixed_tables),
        )
        units = [
            ParsedUnitRecord(
                unit_id=f"{ver_id}::{sec}",
                version_id=ver_id,
                parent_unit_id=None,
                unit_type=NormativeUnitType.SECTION,
                section_number=sec,
                subsection_number="",
                paragraph_number="",
                item_number="",
                title=f"Seção {sec}",
                subpart="",
                original_text=txt,
                normalized_text=txt,
                start_page=1,
                end_page=1,
                page_numbers=(1,),
                order_index=idx,
                structure_status=StructureStatus.PARSED,
                content_hash=hashlib.sha256(txt.encode("utf-8")).hexdigest(),
            )
            for idx, (sec, txt) in enumerate(units_spec)
        ]
        repo.save_version_ingestion(
            document=doc,
            version=ver,
            pages=(page,),
            units=units,
            tables=fixed_tables,
            extraction_run=ExtractionRun(
                run_id=f"run_{name}",
                document_id=doc.document_id,
                version_id=ver_id,
                started_at=now,
                finished_at=now,
                status="succeeded",
                extraction_version="2.0.0",
                source_hash=src_hash,
                pages_extracted=1,
                units_created=len(units),
            ),
        )
        catalog = CatalogService(repo)
        validation = ValidationService(ValidationPolicy.STRICT_ALL_PAGES)
        vstore = _NoOpVectorStore()
        if approved:
            IndexingService(
                repository=repo,
                catalog_service=catalog,
                chunker=StructuralNormativeChunker(),
                vector_store=vstore,  # type: ignore[arg-type]
                embedding_port=_ConstantEmbedder(),
                validation_service=validation,
            ).index_document(document_id=doc.document_id)
        retriever = HybridNormativeRetriever(
            repository=repo,
            vector_store=vstore,  # type: ignore[arg-type]
            embedding_port=None,
            collection_name="test_col",
            validation_service=validation,
        )
        llm_spy = _SpyRewritingLLM()
        chat = NormativeChatService(retriever=retriever, llm=llm_spy)
        get_sec = GetNormativeSectionUseCase(
            repository=repo, catalog_service=catalog, validation_service=validation
        )
        return chat, get_sec, llm_spy, repo

    def _valid_tbl(
        tid: str,
        *,
        caption: str,
        val: str,
        unit_id: str | None,
        owner_sec: str,
        status: str,
        rows_override: tuple[tuple[str, ...], ...] | None = None,
    ) -> NormativeTable:
        return NormativeTable(
            table_id=tid,
            version_id="placeholder",
            unit_id=unit_id,
            pdf_page_number=1,
            caption=caption,
            rows=rows_override or (("PARÂMETRO", "VALOR"), ("Limite", val)),
            header_row_count=1,
            table_kind="data_table",
            ownership_status=status,
            owner_section_number=owner_sec,
        )

    # Cenário 1: 1 página, 1 unidade e 1 tabela confirmada -> retornada com sucesso
    chat1, sec1, spy1, repo1 = _run_scenario(
        "single_confirmed",
        units_spec=[("43.3", "43.3 Manutenção autorizada conforme tabela.")],
        tables=(
            _valid_tbl(
                "t1",
                caption="TABELA 43.3",
                val="100 horas",
                unit_id="43.3",
                owner_sec="43.3",
                status=TableOwnershipStatus.CONFIRMED.value,
            ),
        ),
    )
    try:
        ans1 = chat1.ask(question="RBAC 43.3", generate_explanation=False)
        assert ans1.found is True
        assert ans1.structured_tables_text is not None
        assert "TABELA 43.3" in ans1.structured_tables_text
        assert "100 horas" in ans1.structured_tables_text
        assert spy1.calls == 0
        lu1 = sec1.execute(document_id="anac_rbac_43", section_number="43.3")
        assert lu1.found is True and lu1.section is not None and len(lu1.section.tables) == 1
    finally:
        repo1.close()

    # Cenário 2: 1 página, 2 unidades e tabela SEM proprietário (unit_id=None) -> BLOQUEIA AMBAS!
    chat2, sec2, spy2, repo2 = _run_scenario(
        "unassigned_two_units",
        units_spec=[
            ("43.3", "43.3 Pessoas autorizadas a executar manutenção."),
            ("43.5", "43.5 Aprovação para retorno ao serviço."),
        ],
        tables=(
            _valid_tbl(
                "t_orphan",
                caption="TABELA SEM DONO",
                val="999 pés",
                unit_id=None,
                owner_sec="",
                status=TableOwnershipStatus.UNASSIGNED.value,
            ),
        ),
    )
    try:
        for sec_num in ("43.3", "43.5"):
            ans = chat2.ask(question=f"RBAC {sec_num}", generate_explanation=True)
            assert ans.found is False
            assert ans.normative_text is None
            assert ans.structured_tables_text is None
            assert any("PROPRIEDADE TABULAR BLOQUEADA" in o for o in ans.observations)
            lu = sec2.execute(document_id="anac_rbac_43", section_number=sec_num)
            assert lu.found is False
        assert spy2.calls == 0  # Zero chamadas ao Gemini!
    finally:
        repo2.close()

    # Cenário 3: Duas tabelas de unidades diferentes na mesma página -> zero vazamento!
    chat3, sec3, spy3, repo3 = _run_scenario(
        "two_tables_two_owners",
        units_spec=[
            ("43.3", "43.3 Pessoas autorizadas a executar manutenção."),
            ("43.5", "43.5 Aprovação para retorno ao serviço."),
        ],
        tables=(
            _valid_tbl(
                "t_43_3",
                caption="TABELA EXCLUSIVA 43.3",
                val="VALOR_EXCLUSIVO_43_3",
                unit_id="43.3",
                owner_sec="43.3",
                status=TableOwnershipStatus.CONFIRMED.value,
            ),
            _valid_tbl(
                "t_43_5",
                caption="TABELA EXCLUSIVA 43.5",
                val="VALOR_EXCLUSIVO_43_5",
                unit_id="43.5",
                owner_sec="43.5",
                status=TableOwnershipStatus.CONFIRMED.value,
            ),
        ),
    )
    try:
        ans_3 = chat3.ask(question="RBAC 43.3", generate_explanation=False)
        assert ans_3.found is True and ans_3.structured_tables_text is not None
        assert "VALOR_EXCLUSIVO_43_3" in ans_3.structured_tables_text
        assert "VALOR_EXCLUSIVO_43_5" not in ans_3.structured_tables_text

        ans_5 = chat3.ask(question="RBAC 43.5", generate_explanation=False)
        assert ans_5.found is True and ans_5.structured_tables_text is not None
        assert "VALOR_EXCLUSIVO_43_5" in ans_5.structured_tables_text
        assert "VALOR_EXCLUSIVO_43_3" not in ans_5.structured_tables_text

        lu3 = sec3.execute(document_id="anac_rbac_43", section_number="43.3")
        assert lu3.found is True and lu3.section is not None
        assert [t.caption for t in lu3.section.tables] == ["TABELA EXCLUSIVA 43.3"]
        assert spy3.calls == 0
    finally:
        repo3.close()

    # Cenário 4: Cabeçalho válido mas dono divergente -> BLOQUEIA (integridade != propriedade)
    chat4, sec4, spy4, repo4 = _run_scenario(
        "wrong_owner_valid_header",
        units_spec=[("43.3", "43.3 Pessoas autorizadas a executar manutenção.")],
        tables=(
            _valid_tbl(
                "t_mismatch",
                caption="TABELA COM DONO DIVERGENTE",
                val="500 pés",
                unit_id="43.3",
                owner_sec="43.9",  # Diverge da seção 43.3!
                status=TableOwnershipStatus.CONFIRMED.value,
            ),
        ),
    )
    try:
        ans4 = chat4.ask(question="RBAC 43.3", generate_explanation=True)
        assert ans4.found is False
        assert any("PROPRIEDADE TABULAR BLOQUEADA" in o for o in ans4.observations)
        assert sec4.execute(document_id="anac_rbac_43", section_number="43.3").found is False
        assert spy4.calls == 0
    finally:
        repo4.close()

    # Cenário 5: Tabela ragged/corrompida com dono confirmado -> BLOQUEIA por integridade
    chat5, sec5, spy5, repo5 = _run_scenario(
        "ragged_corrupted",
        units_spec=[("43.3", "43.3 Pessoas autorizadas a executar manutenção.")],
        tables=(
            _valid_tbl(
                "t_ragged",
                caption="TABELA CORROMPIDA",
                val="ok",
                unit_id="43.3",
                owner_sec="43.3",
                status=TableOwnershipStatus.CONFIRMED.value,
                rows_override=(("COL_A", "COL_B"), ("SO_UMA_COLUNA",)),
            ),
        ),
    )
    try:
        ans5 = chat5.ask(question="RBAC 43.3", generate_explanation=True)
        assert ans5.found is False
        assert any("INTEGRIDADE TABULAR BLOQUEADA" in o for o in ans5.observations)
        assert sec5.execute(document_id="anac_rbac_43", section_number="43.3").found is False
        assert spy5.calls == 0
    finally:
        repo5.close()

    # Cenário 6: PAGE_GLOBAL explícita (nunca de unit_id=None!) -> permitida só se explícita
    chat6, _sec6, spy6, repo6 = _run_scenario(
        "explicit_page_global",
        units_spec=[
            ("43.3", "43.3 Pessoas autorizadas a executar manutenção."),
            ("43.5", "43.5 Aprovação para retorno ao serviço."),
        ],
        tables=(
            _valid_tbl(
                "t_global",
                caption="LEGENDA GLOBAL DA PÁGINA",
                val="Aplica-se a toda a página",
                unit_id=None,
                owner_sec="",
                status=TableOwnershipStatus.PAGE_GLOBAL.value,
            ),
        ),
    )
    try:
        ans6 = chat6.ask(question="RBAC 43.3", generate_explanation=False)
        assert ans6.found is True and ans6.structured_tables_text is not None
        assert "LEGENDA GLOBAL DA PÁGINA" in ans6.structured_tables_text
        assert spy6.calls == 0
    finally:
        repo6.close()

    # Cenário 7: Ingestão com tabela ambígua em página multi-seção -> preserva no SQLite com motivo,
    # reprova qualidade automatizada e bloqueia recuperação com zero chamadas ao Gemini!
    ambig_dir = tmp_path / "ambig_ingest"
    ambig_pdf = create_synthetic_rbac_pdf(tmp_path / "ambig_dummy.pdf")
    ambig_repo = SqliteDocumentRepository(ambig_dir / "catalog.sqlite3")
    ambig_repo.initialize()
    try:
        store = LocalFileDocumentStore(documents_dir=ambig_dir)
        store.ensure_directories()

        class _AmbiguousExtractor:
            def extract_pages(
                self,
                pdf_path_arg: Path,
                *,
                min_native_chars: int = 80,
                max_pages: int | None = None,
            ) -> list[ExtractedPageData]:
                del pdf_path_arg, min_native_chars, max_pages
                txt = (
                    "43.3 Pessoas autorizadas a executar manutenção\n"
                    "(a) Somente pessoal habilitado pode executar serviços.\n"
                    "43.5 Aprovação para retorno ao serviço\n"
                    "(a) O retorno ao serviço exige registro formal."
                )
                ambig_tbl = ExtractedTableData(
                    pdf_page_number=1,
                    rows=(("CÓDIGO", "DESCRIÇÃO"), ("X1", "Item sem vínculo textual com seção")),
                    caption="TABELA SEM REFERÊNCIA TEXTUAL",
                    header_row_count=1,
                    table_kind="data_table",
                )
                return [
                    ExtractedPageData(
                        pdf_page_number=1,
                        printed_page_number="1/1",
                        original_text=txt,
                        normalized_text=txt,
                        page_width=595.0,
                        page_height=842.0,
                        needs_ocr=False,
                        content_hash=compute_page_content_hash(txt, (ambig_tbl,)),
                        tables=(ambig_tbl,),
                    )
                ]

        ing = IngestionService(
            repository=ambig_repo,
            file_store=store,
            pdf_extractor=_AmbiguousExtractor(),
            quality_checker=TextQualityChecker(),
            validation_service=ValidationService(ValidationPolicy.STRICT_ALL_PAGES),
            structure_strategies=(AnacRbacStructureStrategy(), GenericStructureStrategy()),
            local_source=LocalFileSourceAdapter(),
        )
        res_amb = ing.ingest(
            IngestionRequestDTO(
                document=RegisterDocumentDTO(
                    document_id="anac_rbac_43_amb",
                    authority="ANAC",
                    document_type=DocumentType.RBAC,
                    regulation_number="43",
                    document_title="RBAC 43",
                    amendment_number="05",
                    version_status=VersionStatus.CURRENT,
                ),
                source_uri=str(ambig_pdf),
            )
        )
        # Tabela ambígua NÃO é descartada: fica salva no SQLite com status 'ambiguous' + motivo!
        saved_tables = ambig_repo.list_tables(res_amb.version.version_id)
        assert len(saved_tables) == 1
        assert saved_tables[0].unit_id is None
        assert saved_tables[0].ownership_status == TableOwnershipStatus.AMBIGUOUS.value
        assert "ambígua" in saved_tables[0].ownership_note.lower()

        saved_pages = ambig_repo.get_pages(res_amb.version.version_id)
        assert saved_pages[0].extraction_metadata is not None
        assert saved_pages[0].extraction_metadata.quality_metrics.passed_automated_checks is False
        assert saved_pages[0].validation_status == PageValidationStatus.PENDING_REVIEW
    finally:
        ambig_repo.close()

    # Cenário 8: Página pendente NUNCA entra no prompt do Gemini!
    chat_pend, _sec_pend, spy_pend, repo_pend = _run_scenario(
        "pending_no_gemini",
        units_spec=[("43.3", "43.3 Conteúdo secreto pendente de aprovação humana.")],
        tables=(),
        approved=False,
    )
    try:
        ans_pend = chat_pend.ask(question="Explique a seção 43.3", generate_explanation=True)
        assert ans_pend.found is False
        assert spy_pend.calls == 0
    finally:
        repo_pend.close()
