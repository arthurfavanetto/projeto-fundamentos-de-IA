"""Testes de fidelidade normativa contra amostras de referência conferidas manualmente."""

from __future__ import annotations

import json
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
from rbac_assistant.application.use_cases.get_normative_section import GetNormativeSectionUseCase
from rbac_assistant.application.use_cases.ingest_document import IngestDocumentUseCase
from rbac_assistant.application.use_cases.validate_page import ValidatePageUseCase
from rbac_assistant.domain.entities.normative_unit import (
    NormativeUnitType,
    TableOwnershipStatus,
    render_structured_table,
    validate_table_integrity,
    validate_table_ownership,
)
from rbac_assistant.domain.entities.regulatory_document import (
    DocumentType,
    ValidationPolicy,
    VersionStatus,
)
from rbac_assistant.domain.entities.regulatory_page import ReviewDecision
from rbac_assistant.infrastructure.pdf.anac_structure_strategy import AnacRbacStructureStrategy
from rbac_assistant.infrastructure.pdf.generic_structure_strategy import GenericStructureStrategy
from rbac_assistant.infrastructure.pdf.pdfplumber_extractor import PdfPlumberExtractorAdapter
from rbac_assistant.infrastructure.pdf.text_quality_checker import TextQualityChecker
from rbac_assistant.infrastructure.sources.local_file_source_adapter import LocalFileSourceAdapter
from rbac_assistant.infrastructure.storage.file_document_store import LocalFileDocumentStore
from rbac_assistant.infrastructure.storage.sqlite_document_repository import (
    SqliteDocumentRepository,
)
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


class _SpyLLM:
    def __init__(self) -> None:
        self.calls = 0

    @property
    def model_name(self) -> str:
        return "gemini-4-hydrogen"

    def generate_text(self, *, system_instruction: str, user_prompt: str) -> str:
        del system_instruction, user_prompt
        self.calls += 1
        return "Explicação simulada [EV-1]."


def _run_ingestion(base_dir: Path, pdf_path: Path) -> tuple[
    SqliteDocumentRepository,
    GetNormativeSectionUseCase,
    ValidatePageUseCase,
    str,
]:
    repo = SqliteDocumentRepository(base_dir / "catalog.sqlite3")
    repo.initialize()
    store = LocalFileDocumentStore(documents_dir=base_dir)
    store.ensure_directories()
    catalog = CatalogService(repo)
    validation = ValidationService(ValidationPolicy.REVIEW_ON_DEMAND)
    svc = IngestionService(
        repository=repo,
        file_store=store,
        pdf_extractor=PdfPlumberExtractorAdapter(),
        quality_checker=TextQualityChecker(),
        validation_service=validation,
        structure_strategies=(AnacRbacStructureStrategy(), GenericStructureStrategy()),
        local_source=LocalFileSourceAdapter(),
    )
    res = IngestDocumentUseCase(svc).execute(
        IngestionRequestDTO(
            document=RegisterDocumentDTO(
                document_id="anac_rbac_43",
                authority="ANAC",
                document_type=DocumentType.RBAC,
                regulation_number="43",
                document_title="Manutenção, Manutenção Preventiva, Reconstrução e Alteração",
                amendment_number="05",
                version_status=VersionStatus.CURRENT,
            ),
            source_uri=str(pdf_path),
        )
    )
    lookup = GetNormativeSectionUseCase(
        repository=repo,
        catalog_service=catalog,
        validation_service=validation,
    )
    validate_uc = ValidatePageUseCase(
        repository=repo,
        catalog_service=catalog,
        validation_service=validation,
        file_store=store,
    )
    return repo, lookup, validate_uc, res.version.source_hash


def test_synthetic_verbatim_fidelity_preserves_punctuation_accents_and_hierarchy(
    tmp_path: Path,
) -> None:
    """Valida bloqueio pré-aprovação e preservação literal integral após aprovação global em CI."""
    pdf_path = create_synthetic_rbac_pdf(tmp_path / "rbac43_fidelity.pdf")
    repo, lookup, validate_uc, _hash = _run_ingestion(tmp_path / "docs", pdf_path)

    # Antes da aprovação global de 100% das páginas, a consulta de seção deve ser bloqueada
    blocked = lookup.execute(document_id="anac_rbac_43", section_number="43.1")
    assert blocked.found is False
    assert blocked.section is None
    assert blocked.blocked_by_document_approval is True
    assert blocked.pending_pdf_pages == (1, 2, 3)

    # Aprova todas as 3 páginas do PDF sintético em tmp_path
    ver = repo.get_current_version("anac_rbac_43")
    assert ver is not None
    validate_uc.execute(
        PageReviewRequestDTO(
            document_id="anac_rbac_43",
            version_id=ver.version_id,
            reviewer="revisor_teste",
            decision=ReviewDecision.APPROVE,
            confirmed=True,
            page_numbers=tuple(range(1, ver.page_count + 1)),
            comments="Aprovação integral do PDF sintético em tmp_path.",
        )
    )

    sec_1 = lookup.execute(document_id="anac_rbac_43", section_number="43.1")
    assert sec_1.found and sec_1.section is not None
    assert sec_1.section.literal_transcription_allowed is True
    expected_lines = [
        "43.1 Aplicabilidade",
        "(a) Este regulamento estabelece regras de manutenção aeronáutica e inspeção.",
        "(1) aeronave com certificado de aeronavegabilidade brasileiro;",
        "(i) célula, motor, hélice, acessório e componentes.",
    ]
    for line in expected_lines:
        assert line in sec_1.section.original_text


def test_official_rbac43_emd05_verbatim_fidelity_when_pdf_available(
    repo_root: Path,
    tmp_path: Path,
) -> None:
    """Valida fidelidade integral, correções verificadas, tabelas e estrutura do RBAC 43 Emd 05."""
    official_pdf = repo_root / "data" / "raw_pdfs" / "RBAC43EMD05.pdf"
    if not official_pdf.is_file():
        pytest.skip(
            "PDF oficial RBAC43EMD05.pdf não está em data/raw_pdfs/ neste ambiente (gitignored)."
        )

    fixture_path = repo_root / "tests" / "fixtures" / "rbac43_reference_samples.json"
    reference = json.loads(fixture_path.read_text(encoding="utf-8"))

    repo, lookup, _validate_uc, file_hash = _run_ingestion(tmp_path / "official_eval", official_pdf)
    try:
        assert file_hash == reference["expected_sha256"]

        ver = repo.get_current_version("anac_rbac_43")
        assert ver is not None
        assert ver.page_count == reference["expected_total_pages"]

        units = repo.list_units(ver.version_id)
        roots = [u for u in units if u.parent_unit_id is None]
        assert len(roots) == reference["expected_root_sections_count"]

        pages = repo.get_pages(ver.version_id)
        pages_by_num = {p.pdf_page_number: p for p in pages}
        assert len(pages) == 32

        # Todas as 32 páginas passam na verificação automatizada sem anomalias não resolvidas
        total_corrections = 0
        for page in pages:
            assert page.extraction_metadata is not None
            assert page.extraction_metadata.quality_metrics.passed_automated_checks is True, (
                f"Página {page.pdf_page_number} falhou qualidade: "
                f"{page.extraction_metadata.quality_metrics.issues}"
            )
            total_corrections += len(page.extraction_metadata.applied_corrections)
            for anomaly in page.extraction_metadata.glyph_anomalies:
                assert anomaly.get("resolved_by_correction_id") is not None
        assert total_corrections == 6
        assert len(pages_by_num[12].extraction_metadata.applied_corrections) == 1  # type: ignore[union-attr]
        assert len(pages_by_num[21].extraction_metadata.applied_corrections) == 1  # type: ignore[union-attr]
        assert len(pages_by_num[31].extraction_metadata.applied_corrections) == 4  # type: ignore[union-attr]

        # Auditoria das 5 tabelas/formulários (p. 21 formulário + p. 29 TABELA I/II + p. 30 III/IV)
        all_tables = repo.list_tables(ver.version_id)
        assert len(all_tables) == 5
        for tbl in all_tables:
            assert validate_table_integrity(tbl) == ()
            assert validate_table_ownership(tbl) == ()
            assert tbl.unit_id is not None
            assert tbl.ownership_status == TableOwnershipStatus.CONFIRMED.value
            expected_owner = "B43.1" if tbl.pdf_page_number == 21 else "E43.1"
            assert tbl.owner_section_number == expected_owner

        p21_tables = [t for t in all_tables if t.pdf_page_number == 21]
        assert len(p21_tables) == 1
        form_tbl = p21_tables[0]
        assert form_tbl.table_kind == "form"
        assert form_tbl.visual_layout_preserved is False
        assert form_tbl.has_merged_cells is True
        assert form_tbl.rows[4][1] == "Nº Tel"
        assert all("No Tel" not in cell for row in form_tbl.rows for cell in row)
        assert "não é reproduzido em fac-símile" in render_structured_table(form_tbl)

        p29_tables = [t for t in all_tables if t.pdf_page_number == 29]
        assert len(p29_tables) == 2
        assert p29_tables[0].caption == "TABELA I"
        assert len(p29_tables[0].rows) == 25 and len(p29_tables[0].rows[0]) == 6
        assert p29_tables[0].header_row_count == 2
        assert p29_tables[1].caption == "TABELA II — Tolerância do Teste"
        assert len(p29_tables[1].rows) == 7 and len(p29_tables[1].rows[0]) == 3
        assert p29_tables[1].rows[2] == ("Vazamento da caixa", "± 100", "30,48")
        assert p29_tables[1].rows[3] == ("Histerese", "", "")

        p30_tables = [t for t in all_tables if t.pdf_page_number == 30]
        assert len(p30_tables) == 2
        assert p30_tables[0].caption == "TABELA III — Atrito"
        assert p30_tables[0].rows[2] == ("1.000", "304,8", "+70", "+21,3")
        assert p30_tables[1].caption == "TABELA IV — Pressão – diferença de altitude"
        assert p30_tables[1].rows[1][0] == "(polegadas deHg)"
        assert p30_tables[1].rows[7] == ("30,50", "1.032,9", "+531", "161,8")
        assert p30_tables[1].rows[8] == ("30,90", "1.046,4", "+893", "272,2")
        assert p30_tables[1].rows[9] == ("30,99", "1.049,5", "+974", "296,9")

        # Hierarquia corrigida de F43.1: parágrafos (c) e (d) nunca classificados como alíneas!
        f43_subunits = [
            u for u in units if u.section_number == "F43.1" and u.parent_unit_id is not None
        ]
        f43_par_labels = [
            u.paragraph_number for u in f43_subunits if u.unit_type == NormativeUnitType.PARAGRAPH
        ]
        assert f43_par_labels == [
            "(a)",
            "(b)",
            "(c)",
            "(d)",
            "(e)",
            "(f)",
            "(g)",
            "(h)",
            "(i)",
            "(j)",
            "(k)",
        ]
        # D43.1 com subparte correta
        d43_root = next(u for u in roots if u.section_number == "D43.1")
        assert d43_root.subpart == "APÊNDICE D DO RBAC 43"

        for sample in reference["samples"]:
            # 1. Sem aprovação humana das 32 páginas, GetNormativeSectionUseCase permanece bloqueado
            blocked_lookup = lookup.execute(
                document_id="anac_rbac_43",
                section_number=sample["section_number"],
            )
            assert blocked_lookup.found is False
            assert blocked_lookup.section is None
            assert blocked_lookup.blocked_by_document_approval is True
            assert blocked_lookup.pending_pdf_pages == tuple(range(1, ver.page_count + 1))

            # 2. Confere fidelidade literal integral diretamente nas unidades extraídas em tmp_path
            sec_units = repo.get_units_by_section(
                version_id=ver.version_id,
                section_number=sample["section_number"],
            )
            assert sec_units
            root_units = [
                u
                for u in sec_units
                if u.parent_unit_id is None
                and u.unit_type in (NormativeUnitType.SECTION, NormativeUnitType.APPENDIX)
            ]
            primary = root_units[0] if root_units else sec_units[0]
            if "expected_title" in sample:
                assert primary.title == sample["expected_title"]
            pdf_pages = sorted({p for u in sec_units for p in u.page_numbers})
            printed_pages = [
                pages_by_num[p].printed_page_number or f"PDF {p}"
                for p in pdf_pages
                if p in pages_by_num
            ]
            assert pdf_pages == sample["expected_pdf_pages"]
            assert printed_pages == sample["expected_printed_pages"]
            for fragment in sample["expected_verbatim_fragments"]:
                assert fragment in primary.original_text
            for forbidden in sample.get("forbidden_fragments", ()):
                assert forbidden not in primary.original_text
    finally:
        repo.close()


@pytest.mark.asyncio
async def test_official_rbac43_emd05_telegram_end_to_end_in_isolated_tmp_db(
    repo_root: Path,
    tmp_path: Path,
) -> None:
    """Valida gate de 32/32 páginas e entrega literal no Telegram em banco descartável tmp_path."""
    official_pdf = repo_root / "data" / "raw_pdfs" / "RBAC43EMD05.pdf"
    if not official_pdf.is_file():
        pytest.skip(
            "PDF oficial RBAC43EMD05.pdf não está em data/raw_pdfs/ neste ambiente (gitignored)."
        )

    base_dir = tmp_path / "official_e2e_tg"
    repo, _lookup, validate_uc, _file_hash = _run_ingestion(base_dir, official_pdf)
    try:
        ver = repo.get_current_version("anac_rbac_43")
        assert ver is not None
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
        retriever = HybridNormativeRetriever(
            repository=repo,
            vector_store=vector_store,  # type: ignore[arg-type]
            embedding_port=None,
            collection_name="test_col",
            validation_service=validation,
        )
        llm_spy = _SpyLLM()
        chat = NormativeChatService(retriever=retriever, llm=llm_spy)
        router = TelegramCommandRouter(
            access_control=AccessControl(admins=(1,), public_access=True),
            rate_limiter=UserRateLimiter(requests_per_minute=50, burst_size=50),
            splitter=MessageSplitter(max_message_length=4096),
            status_use_case=CheckSystemStatusUseCase(
                vector_store=vector_store,  # type: ignore[arg-type]
                documents_dir=base_dir,
                audit_logs_dir=base_dir,
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

        # 1. Aprova apenas 31/32 páginas (deixando p. 32 pendente) -> TUDO bloqueado, zero LLM
        validate_uc.execute(
            PageReviewRequestDTO(
                document_id="anac_rbac_43",
                version_id=ver.version_id,
                reviewer="revisor_tmp",
                decision=ReviewDecision.APPROVE,
                confirmed=True,
                page_numbers=tuple(range(1, 32)),
                comments="Aprovação parcial 31/32 apenas em banco descartável.",
            )
        )
        partial_resp = await router.handle_message(user_id=99, text="/rbac 43.1", update_id=1)
        assert "indisponível" in partial_resp.messages[0].lower()
        assert "43.1 Aplicabilidade" not in "".join(partial_resp.messages)
        assert llm_spy.calls == 0

        # 2. Aprova a 32ª página em tmp_path e indexa em tmp_path
        validate_uc.execute(
            PageReviewRequestDTO(
                document_id="anac_rbac_43",
                version_id=ver.version_id,
                reviewer="revisor_tmp",
                decision=ReviewDecision.APPROVE,
                confirmed=True,
                page_numbers=(32,),
                comments="Aprovação completa 32/32 apenas em banco descartável.",
            )
        )
        indexing.index_document(document_id="anac_rbac_43")

        # 3. Consulta /rbac F43.1, E43.1, B43.1, 43.13 -> entrega literal exata no Telegram
        for uid, cmd, must_contain, must_not_contain in (
            (
                2,
                "/rbac F43.1",
                (
                    "1.090±3 MHz.",
                    "235±5 interrogações",
                    "-73±4 dBm",
                    "-74±3 dBm",
                    "(k) Registros.",
                ),
                ("1.090+3 MHz", "235+5 interrogações", "-73+4 dBm", "-74+3 dBm"),
            ),
            (
                3,
                "/rbac E43.1",
                (
                    "TABELAS E FORMULÁRIOS ESTRUTURADOS (SQLite SSOT)",
                    "TABELA I",
                    "TABELA II — Tolerância do Teste",
                    "TABELA III — Atrito",
                    "TABELA IV — Pressão – diferença de altitude",
                    "TOLERÂNCIA (pés)=+70",
                    "TOLERÂNCIA (m)=+21,3",
                    "DIFERENÇA DE ALTITUDE (pés)=+974",
                    "(polegadas deHg)",
                ),
                ("±70", "±21,3", "±974"),
            ),
            (
                4,
                "/rbac B43.1",
                (
                    "Nome da Organização de Manutenção Nº Tel",
                    "FORMULÁRIO ESTRUTURADO",
                    "não é reproduzido em fac-símile",
                ),
                ("No Tel",),
            ),
            (
                5,
                "/rbac 43.13",
                ("no art. 120 da Lei nº 7565, de 19 de dezembro de 1986,",),
                ("Lei no 7565",),
            ),
        ):
            resp = await router.handle_message(user_id=99, text=cmd, update_id=uid)
            assert resp.messages
            assert all(utf16_length(m) <= 4096 for m in resp.messages)
            reconstructed = "".join(strip_pagination_header(m) for m in resp.messages)
            for snippet in must_contain:
                assert snippet in reconstructed, f"Esperado {snippet!r} em resposta de {cmd}"
            for bad in must_not_contain:
                assert bad not in reconstructed, f"Proibido {bad!r} em resposta de {cmd}"

            # Prova RAW caractere-a-caractere SEM sanitize_telegram_text nos dois lados!
            direct_ans = chat.ask(
                question=f"RBAC {cmd.split(maxsplit=1)[1]}",
                generate_explanation=False,
            )
            assert reconstructed == direct_ans.formatted_text
            assert "=== EXPLICAÇÃO (GERADA POR IA) ===" not in reconstructed
            extracted_literals = extract_literal_excerpts_from_text(reconstructed)
            assert extracted_literals == direct_ans.literal_excerpts
            sec_num = cmd.split(maxsplit=1)[1]
            sqlite_units = [
                u
                for u in repo.get_units_by_section(
                    version_id=ver.version_id, section_number=sec_num
                )
                if u.parent_unit_id is None
            ]
            assert sqlite_units
            assert sqlite_units[0].original_text in extracted_literals
        assert llm_spy.calls == 0
    finally:
        repo.close()
