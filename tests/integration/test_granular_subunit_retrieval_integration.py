"""Testes integrados (Casos A–G) de recuperação estrutural granular literal no SSOT SQLite.

Cobre integralmente:
- Caso A: Seção inteira (`43.10`)
- Caso B: Parágrafo específico (`43.10(a)`) com seus incisos subordinados, sem `(b)`/`(c)`/`(d)`
- Caso C: Inciso específico (`43.10(a)(1)`) reproduzindo exatamente o defeito real reportado
- Caso D: Alínea específica (`43.10(c)(5)(i)`) sem confundir com inciso `(5)` nem alínea `(ii)`
- Caso E: Unidade inexistente / corrompida / não aprovada -> abstenção segura sem fallback
- Caso F: Fusão RRF (estrutural + FTS5 + ChromaDB) sem expansão e com múltiplas refs na mesma seção
- Caso G: Integridade literal caractere-a-caractere contra o SSOT, ordem, páginas e aprovação

Todos os testes operam exclusivamente em diretórios descartáveis `tmp_path` e nunca alteram
bancos, coleções, volumes Docker ou aprovações de produção.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest
from scripts import query_assistant as query_cli

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
from rbac_assistant.application.use_cases.validate_page import ValidatePageUseCase
from rbac_assistant.domain.entities.document_version import DocumentVersion, ExtractionRun
from rbac_assistant.domain.entities.normative_unit import ExtractionMethod
from rbac_assistant.domain.entities.rag import EvidenceTrustLevel, SearchFilter
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
from rbac_assistant.infrastructure.pdf.anac_structure_strategy import AnacRbacStructureStrategy
from rbac_assistant.infrastructure.pdf.generic_structure_strategy import GenericStructureStrategy
from rbac_assistant.infrastructure.pdf.pdfplumber_extractor import PdfPlumberExtractorAdapter
from rbac_assistant.infrastructure.pdf.text_quality_checker import TextQualityChecker
from rbac_assistant.infrastructure.sources.local_file_source_adapter import LocalFileSourceAdapter
from rbac_assistant.infrastructure.storage.file_document_store import LocalFileDocumentStore
from rbac_assistant.infrastructure.storage.sqlite_document_repository import (
    SqliteDocumentRepository,
)
from rbac_assistant.infrastructure.vector_db.chroma_vector_store import ChromaVectorStore
from rbac_assistant.rag.generation.response_formatter import extract_literal_excerpts_from_text
from rbac_assistant.rag.ingestion.structural_chunker import StructuralNormativeChunker
from rbac_assistant.rag.retrieval.hybrid_retriever import HybridNormativeRetriever
from tests.integration.test_hybrid_rag_pipeline import (
    _DeterministicSemanticEmbedder,
    _RecordingFakeLLM,
)
from tests.unit.test_rag_components import _InMemoryChromaClient


def _build_synthetic_rbac43_10_stack(
    tmp_path: Path,
    *,
    corrupt_subunit_a1: bool = False,
    approve_all_pages: bool = True,
) -> tuple[
    SqliteDocumentRepository,
    HybridNormativeRetriever,
    NormativeChatService,
    _RecordingFakeLLM,
    ChromaVectorStore,
    _DeterministicSemanticEmbedder,
    str,
]:
    """Constrói banco SQLite + FTS5 + ChromaDB em `tmp_path` com estrutura real da seção 43.10."""
    docs_dir = tmp_path / "documents"
    docs_dir.mkdir(parents=True, exist_ok=True)
    repo = SqliteDocumentRepository(docs_dir / "catalog.sqlite3")
    repo.initialize()

    p9_text = "\n".join(
        [
            "43.9 Conteúdo, forma e disposição de registros de manutenção",
            "(a) Cada pessoa que executar manutenção deve efetuar o registro correspondente.",
            "43.10 Controle de peças com limite de vida",
            (
                "(a) Definições usadas nesta seção. Para os propósitos desta seção se aplicam "
                "as seguintes"
            ),
            "definições:",
            (
                "(1) Peça com limite de vida significa qualquer peça ou parte para a qual um "
                "limite obrigatório"
            ),
            (
                "de substituição é especificado no projeto de tipo, nas instruções de "
                "aeronavegabilidade continuada"
            ),
            "ou no manual de manutenção.",
            (
                "(2) Situação de vida significa número acumulado de ciclos, horas ou qualquer "
                "outro limite"
            ),
            "obrigatório de substituição de uma peça com limite de vida.",
            (
                "(b) Remoção temporária de peças de produtos com certificado de tipo. Quando "
                "uma peça com"
            ),
            (
                "limite de vida é temporariamente removida e reinstalada com o propósito de "
                "realizar manutenção,"
            ),
            "não é requerido qualquer controle segundo o parágrafo (c) desta seção se:",
            "(1) a situação de vida da peça não foi alterada;",
            "(2) a remoção e reinstalação foi realizada em produto com o mesmo número de série; e",
            "(3) o produto não acumulou tempo em serviço enquanto a peça esteve removida.",
            (
                "(c) Controle de peças removidas de produtos com certificado de tipo. "
                "Métodos aceitáveis incluem:"
            ),
            "(1) Sistema de registro de informações.",
            "(2) Etiqueta ou registro preso à peça.",
            "(3) Marca não permanente.",
            "(4) Marca permanente.",
        ]
    )
    p10_text = "\n".join(
        [
            (
                "(5) Segregação. A peça pode ser segregada usando métodos para impedir sua "
                "instalação em"
            ),
            "um produto com certificado de tipo. Tais métodos devem incluir, pelo menos:",
            (
                "(i) a conservação dos registros do número da peça, seu número de série e sua "
                "situação de"
            ),
            "vida; e",
            (
                "(ii) a garantia de que a peça está armazenada em local fisicamente separado "
                "de outras peças"
            ),
            "que sejam correntemente elegíveis para instalação.",
            "(6) Mutilação. A peça pode ser mutilada para impedir sua instalação em um produto com",
            "certificado de tipo.",
            "(7) Outros métodos. Qualquer outro método aprovado pela ANAC.",
            (
                "(d) Transferência de peças com limite de vida. Cada pessoa que remover uma "
                "peça com limite de"
            ),
            (
                "vida de um produto com certificado de tipo e posteriormente a vender deve "
                "transferir o registro."
            ),
        ]
    )

    now = datetime.now(UTC)
    src_hash = hashlib.sha256(b"synth_rbac43_10_granular_test").hexdigest()
    ver_id = "anac_rbac_43_emd_05_5026d2da"
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
        version_id=ver_id,
        document_id=doc.document_id,
        amendment_number="05",
        version_status=VersionStatus.CURRENT,
        source_hash=src_hash,
        page_count=2,
        imported_at=now,
    )
    status_p9 = PageValidationStatus.APPROVED_FOR_LITERAL_TRANSCRIPTION
    status_p10 = (
        PageValidationStatus.APPROVED_FOR_LITERAL_TRANSCRIPTION
        if approve_all_pages
        else PageValidationStatus.PENDING_REVIEW
    )
    pages = [
        RegulatoryPage(
            page_id=f"{ver_id}::p0001",
            version_id=ver_id,
            pdf_page_number=1,
            printed_page_number="9/32",
            original_text=p9_text,
            normalized_text=p9_text,
            extraction_method=ExtractionMethod.NATIVE_PDF,
            validation_status=status_p9,
            content_hash=compute_page_content_hash(p9_text, ()),
        ),
        RegulatoryPage(
            page_id=f"{ver_id}::p0002",
            version_id=ver_id,
            pdf_page_number=2,
            printed_page_number="10/32",
            original_text=p10_text,
            normalized_text=p10_text,
            extraction_method=ExtractionMethod.NATIVE_PDF,
            validation_status=status_p10,
            content_hash=compute_page_content_hash(p10_text, ()),
        ),
    ]
    parsed_units = AnacRbacStructureStrategy().parse(
        version_id=ver_id,
        regulation_number="43",
        pages=pages,
    )
    if corrupt_subunit_a1:
        parsed_units = [
            (
                replace(
                    u,
                    original_text="(1) Texto adulterado incompatível com a unidade-raiz SSOT.",
                )
                if u.section_number == "43.10"
                and u.paragraph_number == "(a)"
                and u.item_number == "(1)"
                else u
            )
            for u in parsed_units
        ]

    repo.save_version_ingestion(
        document=doc,
        version=ver,
        pages=pages,
        units=parsed_units,
        tables=(),
        extraction_run=ExtractionRun(
            run_id="run_granular_test",
            document_id=doc.document_id,
            version_id=ver_id,
            started_at=now,
            finished_at=now,
            status="succeeded",
            extraction_version="2.0.0",
            source_hash=src_hash,
            pages_extracted=2,
            units_created=len(parsed_units),
        ),
    )

    catalog = CatalogService(repo)
    validation = ValidationService(ValidationPolicy.STRICT_ALL_PAGES)
    vector_store = ChromaVectorStore(client=_InMemoryChromaClient())
    embedder = _DeterministicSemanticEmbedder()
    if approve_all_pages:
        IndexingService(
            repository=repo,
            catalog_service=catalog,
            chunker=StructuralNormativeChunker(),
            vector_store=vector_store,
            embedding_port=embedder,
            default_collection_name="test_granular_col",
            validation_service=validation,
        ).index_document(document_id=doc.document_id)

    retriever = HybridNormativeRetriever(
        repository=repo,
        vector_store=vector_store,
        embedding_port=embedder,
        collection_name="test_granular_col",
        validation_service=validation,
        reconstruct_full_section_from_ssot=True,
    )
    llm = _RecordingFakeLLM()
    chat = NormativeChatService(retriever=retriever, llm=llm)
    return repo, retriever, chat, llm, vector_store, embedder, ver_id


@pytest.mark.integration
def test_granular_retrieval_cases_a_through_g_on_synthetic_rbac43_10(tmp_path: Path) -> None:
    """Executa integralmente os Casos A–G (funciona sempre, inclusive no CI sem PDF oficial)."""
    repo, _retriever, chat, llm, vector_store, embedder, ver_id = _build_synthetic_rbac43_10_stack(
        tmp_path
    )
    try:
        units_43_10 = repo.get_units_by_section(version_id=ver_id, section_number="43.10")
        root_43_10 = next(u for u in units_43_10 if u.parent_unit_id is None)
        par_a_units = [u for u in units_43_10 if u.paragraph_number == "(a)"]
        unit_a1 = next(u for u in par_a_units if u.item_number == "(1)")
        unit_a2 = next(u for u in par_a_units if u.item_number == "(2)")
        unit_c5_i = next(
            u for u in units_43_10 if u.paragraph_number == "(c)" and u.item_number == "(5)(i)"
        )
        filt = SearchFilter(
            document_id="anac_rbac_43",
            regulation_number="43",
            amendment_number="05",
            only_approved_literal=True,
        )

        # CASO A — Seção inteira
        ans_a = chat.ask(
            question="Transcreva literalmente a seção 43.10.",
            filters=filt,
            top_k=3,
            generate_explanation=False,
        )
        assert ans_a.found is True and ans_a.is_literal_validated is True
        assert len(ans_a.evidences) == 1
        assert ans_a.evidences[0].unit_id == root_43_10.unit_id
        assert ans_a.evidences[0].original_text == root_43_10.original_text
        assert ans_a.evidences[0].pdf_pages == (1, 2)
        assert extract_literal_excerpts_from_text(ans_a.formatted_text) == (
            root_43_10.original_text,
        )

        # CASO B — Parágrafo 43.10(a) (inclui (a), (1) e (2); exclui (b), (c) e (d))
        ans_b = chat.ask(
            question="Transcreva literalmente o parágrafo 43.10(a).",
            filters=filt,
            top_k=3,
            generate_explanation=False,
        )
        assert ans_b.found is True and ans_b.is_literal_validated is True
        assert len(ans_b.evidences) == 1
        ev_b = ans_b.evidences[0]
        expected_par_a = "\n".join(u.original_text for u in par_a_units)
        assert ev_b.unit_id == par_a_units[0].unit_id
        assert ev_b.paragraph_number == "(a)"
        assert ev_b.item_number == ""
        assert ev_b.pdf_pages == (1,)
        assert ev_b.original_text == expected_par_a
        assert ev_b.original_text in root_43_10.original_text
        assert "(b)" not in ev_b.original_text
        assert "(c)" not in ev_b.original_text
        assert "(d)" not in ev_b.original_text
        assert extract_literal_excerpts_from_text(ans_b.formatted_text) == (expected_par_a,)

        # CASO C — Inciso específico 43.10(a)(1) (reprodução fiel do comando reportado)
        ans_c = chat.ask(
            question="Transcreva literalmente o item 43.10(a)(1) do RBAC n 43, Emenda n 05.",
            filters=filt,
            top_k=3,
            generate_explanation=False,
        )
        assert ans_c.found is True and ans_c.is_literal_validated is True
        assert len(ans_c.evidences) == 1
        ev_c = ans_c.evidences[0]
        assert ev_c.trust_level == EvidenceTrustLevel.VALIDATED_LITERAL
        assert ev_c.unit_id == unit_a1.unit_id
        assert ev_c.paragraph_number == "(a)"
        assert ev_c.item_number == "(1)"
        assert ev_c.pdf_pages == (1,)
        assert ev_c.page_statuses == ((1, PageValidationStatus.APPROVED_FOR_LITERAL_TRANSCRIPTION),)
        assert ev_c.original_text == unit_a1.original_text
        assert "(2)" not in ev_c.original_text
        assert "Situação de vida" not in ev_c.original_text
        assert "(b)" not in ev_c.original_text
        assert "(c)" not in ev_c.original_text
        assert "(d)" not in ev_c.original_text
        assert "43.10 Controle de peças" not in ev_c.original_text
        assert extract_literal_excerpts_from_text(ans_c.formatted_text) == (unit_a1.original_text,)

        # CASO D — Alínea específica 43.10(c)(5)(i)
        ans_d = chat.ask(
            question="Transcreva literalmente o item 43.10(c)(5)(i).",
            filters=filt,
            top_k=3,
            generate_explanation=False,
        )
        assert ans_d.found is True and ans_d.is_literal_validated is True
        assert len(ans_d.evidences) == 1
        ev_d = ans_d.evidences[0]
        assert ev_d.unit_id == unit_c5_i.unit_id
        assert ev_d.paragraph_number == "(c)"
        assert ev_d.item_number == "(5)(i)"
        assert ev_d.pdf_pages == (2,)
        assert ev_d.original_text == unit_c5_i.original_text
        assert "(5) Segregação" not in ev_d.original_text
        assert "(ii)" not in ev_d.original_text
        assert "(6)" not in ev_d.original_text
        assert extract_literal_excerpts_from_text(ans_d.formatted_text) == (
            unit_c5_i.original_text,
        )

        # CASO E — Unidades inexistentes: abstenção segura sem fallback para seção-pai e 0 LLM
        assert (
            repo.find_chunks_by_structure(
                section_number="43.10",
                regulation_number="43",
                paragraph_number="(a)",
                item_number="(99)",
                filters=filt,
            )
            == []
        )
        for missing_q in (
            "Transcreva literalmente o item 43.10(a)(99) do RBAC n 43, Emenda n 05.",
            "Transcreva literalmente o parágrafo 43.10(z).",
            "Transcreva literalmente o item 43.10(c)(5)(ix).",
            "Transcreva literalmente o item 43.10(a)(1)(i).",
        ):
            ans_e = chat.ask(
                question=missing_q,
                filters=filt,
                top_k=3,
                generate_explanation=True,
            )
            assert ans_e.found is False, f"Deveria abster-se para: {missing_q}"
            assert ans_e.is_literal_validated is False
            assert ans_e.normative_text is None
            assert ans_e.evidences == ()
            assert "43.10 Controle de peças" not in ans_e.formatted_text

        # CASO F — Fusão RRF: múltiplas referências na mesma seção + anti-expansão lexical/vetorial
        ans_multi = chat.ask(
            question="Transcreva literalmente os itens 43.10(a)(1) e 43.10(a)(2) do RBAC 43.",
            filters=filt,
            top_k=5,
            generate_explanation=False,
        )
        assert ans_multi.found is True
        assert [ev.unit_id for ev in ans_multi.evidences] == [unit_a1.unit_id, unit_a2.unit_id]
        assert extract_literal_excerpts_from_text(ans_multi.formatted_text) == (
            unit_a1.original_text,
            unit_a2.original_text,
        )

        # Mesmo em lexical_only ou vector_only (enable_structural=False), não expande para seção
        for mode_kwargs in (
            {"enable_structural": False, "enable_lexical": True, "enable_vector": False},
            {"enable_structural": False, "enable_lexical": False, "enable_vector": True},
        ):
            mode_ret = HybridNormativeRetriever(
                repository=repo,
                vector_store=vector_store,
                embedding_port=embedder,
                collection_name="test_granular_col",
                validation_service=ValidationService(ValidationPolicy.STRICT_ALL_PAGES),
                **mode_kwargs,
            )
            res_mode = mode_ret.retrieve(
                query="Transcreva literalmente o item 43.10(a)(1) do RBAC 43.",
                filters=filt,
                top_k=3,
            )
            for ev in res_mode.evidences:
                if ev.section_number == "43.10":
                    assert ev.unit_id == unit_a1.unit_id
                    assert ev.original_text == unit_a1.original_text
                    assert "(2)" not in ev.original_text

        assert llm.calls == []
    finally:
        repo.close()

    # Caso E/G adicional: subunidade corrompida (não verificável na raiz) -> BLOQUEIO FAIL-CLOSED
    repo_c, _ret_c, chat_c, llm_c, _vs_c, _emb_c, _vid_c = _build_synthetic_rbac43_10_stack(
        tmp_path / "corrupted",
        corrupt_subunit_a1=True,
    )
    try:
        ans_corrupt = chat_c.ask(
            question="Transcreva literalmente o item 43.10(a)(1) do RBAC 43.",
            generate_explanation=True,
        )
        assert ans_corrupt.found is False
        assert ans_corrupt.is_literal_validated is False
        assert ans_corrupt.normative_text is None
        assert ans_corrupt.has_blocked_sources is True
        assert any("GRANULARIDADE NORMATIVA BLOQUEADA" in o for o in ans_corrupt.observations)
        assert llm_c.calls == []
    finally:
        repo_c.close()

    # Caso E adicional: versão com página pendente -> bloqueio global preservado (0 LLM)
    repo_p, _ret_p, chat_p, llm_p, _vs_p, _emb_p, _vid_p = _build_synthetic_rbac43_10_stack(
        tmp_path / "pending",
        approve_all_pages=False,
    )
    try:
        ans_pend = chat_p.ask(
            question="Transcreva literalmente o item 43.10(a)(1) do RBAC 43.",
            generate_explanation=True,
        )
        assert ans_pend.found is False
        assert ans_pend.normative_text is None
        assert ans_pend.has_blocked_sources is True
        assert llm_p.calls == []
    finally:
        repo_p.close()


@pytest.mark.integration
def test_official_rbac43_emd05_exact_subunit_retrieval_cases_a_through_g_in_tmp_db(
    repo_root: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Valida Casos A–G e o comando CLI exato contra o PDF oficial RBAC43EMD05.pdf em tmp_path."""
    official_pdf = repo_root / "data" / "raw_pdfs" / "RBAC43EMD05.pdf"
    if not official_pdf.is_file():
        pytest.skip(
            "PDF oficial RBAC43EMD05.pdf não está em data/raw_pdfs/ neste ambiente (gitignored)."
        )

    docs_dir = tmp_path / "official_rbac43_granular"
    repo = SqliteDocumentRepository(docs_dir / "catalog.sqlite3")
    repo.initialize()
    try:
        store = LocalFileDocumentStore(documents_dir=docs_dir)
        store.ensure_directories()
        validation = ValidationService(ValidationPolicy.STRICT_ALL_PAGES)
        catalog = CatalogService(repo)
        ingestion = IngestionService(
            repository=repo,
            file_store=store,
            pdf_extractor=PdfPlumberExtractorAdapter(),
            quality_checker=TextQualityChecker(),
            validation_service=validation,
            structure_strategies=(AnacRbacStructureStrategy(), GenericStructureStrategy()),
            local_source=LocalFileSourceAdapter(),
        )
        ing_res = ingestion.ingest(
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
                source_uri=str(official_pdf),
            )
        )
        ver_id = ing_res.version.version_id
        assert ver_id == "anac_rbac_43_emd_05_5026d2da"

        ValidatePageUseCase(
            repository=repo,
            catalog_service=catalog,
            validation_service=validation,
            file_store=store,
        ).execute(
            PageReviewRequestDTO(
                document_id="anac_rbac_43",
                version_id=ver_id,
                reviewer="revisor_tmp",
                decision=ReviewDecision.APPROVE,
                confirmed=True,
                page_numbers=tuple(range(1, 33)),
                comments="Aprovação descartável em tmp_path para validação granular.",
            )
        )

        vector_store = ChromaVectorStore(client=_InMemoryChromaClient())
        embedder = _DeterministicSemanticEmbedder()
        IndexingService(
            repository=repo,
            catalog_service=catalog,
            chunker=StructuralNormativeChunker(),
            vector_store=vector_store,
            embedding_port=embedder,
            default_collection_name="tmp_official_rbac43_col",
            validation_service=validation,
        ).index_document(document_id="anac_rbac_43")

        retriever = HybridNormativeRetriever(
            repository=repo,
            vector_store=vector_store,
            embedding_port=embedder,
            collection_name="tmp_official_rbac43_col",
            validation_service=validation,
            reconstruct_full_section_from_ssot=True,
        )
        llm = _RecordingFakeLLM()
        chat = NormativeChatService(retriever=retriever, llm=llm)

        units_43_10 = repo.get_units_by_section(version_id=ver_id, section_number="43.10")
        root_43_10 = next(u for u in units_43_10 if u.parent_unit_id is None)
        par_a_units = [u for u in units_43_10 if u.paragraph_number == "(a)"]
        unit_a1 = next(u for u in par_a_units if u.item_number == "(1)")
        unit_c5_i = next(
            u for u in units_43_10 if u.paragraph_number == "(c)" and u.item_number == "(5)(i)"
        )
        filt = SearchFilter(
            document_id="anac_rbac_43",
            regulation_number="43",
            amendment_number="05",
            only_approved_literal=True,
        )

        # Caso A — Seção 43.10 completa (páginas 9 e 10)
        ans_a = chat.ask(
            question="Transcreva literalmente a seção 43.10.",
            filters=filt,
            top_k=3,
            generate_explanation=False,
        )
        assert ans_a.found is True and len(ans_a.evidences) == 1
        assert ans_a.evidences[0].original_text == root_43_10.original_text
        assert ans_a.evidences[0].pdf_pages == (9, 10)

        # Caso B — Parágrafo 43.10(a) (página 9 apenas; exclui (b), (c), (d))
        ans_b = chat.ask(
            question="Transcreva literalmente o parágrafo 43.10(a).",
            filters=filt,
            top_k=3,
            generate_explanation=False,
        )
        assert ans_b.found is True and len(ans_b.evidences) == 1
        assert ans_b.evidences[0].pdf_pages == (9,)
        assert ans_b.evidences[0].original_text == "\n".join(u.original_text for u in par_a_units)
        assert "(b)" not in ans_b.evidences[0].original_text

        # Caso C — Consulta real exata: 43.10(a)(1) (página 9 apenas; somente inciso (1))
        ans_c = chat.ask(
            question="Transcreva literalmente o item 43.10(a)(1) do RBAC n 43, Emenda n 05.",
            filters=filt,
            top_k=3,
            generate_explanation=False,
        )
        assert ans_c.found is True and ans_c.is_literal_validated is True
        assert len(ans_c.evidences) == 1
        assert ans_c.evidences[0].unit_id == unit_a1.unit_id
        assert ans_c.evidences[0].pdf_pages == (9,)
        assert ans_c.evidences[0].original_text == unit_a1.original_text
        assert extract_literal_excerpts_from_text(ans_c.formatted_text) == (unit_a1.original_text,)
        assert "(2) Situação de vida" not in ans_c.formatted_text
        assert "(b) Remoção temporária" not in ans_c.formatted_text

        # Caso D — Alínea 43.10(c)(5)(i) (página 10 apenas; somente alínea (i))
        ans_d = chat.ask(
            question="Transcreva literalmente o item 43.10(c)(5)(i).",
            filters=filt,
            top_k=3,
            generate_explanation=False,
        )
        assert ans_d.found is True and len(ans_d.evidences) == 1
        assert ans_d.evidences[0].unit_id == unit_c5_i.unit_id
        assert ans_d.evidences[0].pdf_pages == (10,)
        assert ans_d.evidences[0].original_text == unit_c5_i.original_text
        assert "(ii)" not in ans_d.evidences[0].original_text
        assert "(5) Segregação" not in ans_d.evidences[0].original_text

        # Caso E — Unidade inexistente no RBAC 43.10 -> abstenção segura (0 LLM)
        ans_e = chat.ask(
            question="Transcreva literalmente o item 43.10(a)(99) do RBAC n 43, Emenda n 05.",
            filters=filt,
            top_k=3,
            generate_explanation=True,
        )
        assert ans_e.found is False
        assert ans_e.normative_text is None
        assert llm.calls == []

        # Validação end-to-end da CLI `scripts/query_assistant.py --no-llm --debug` em tmp_path
        rc = query_cli.main(
            [
                "--question",
                "Transcreva literalmente o item 43.10(a)(1) do RBAC n 43, Emenda n 05.",
                "--document-id",
                "anac_rbac_43",
                "--regulation",
                "43",
                "--amendment",
                "05",
                "--only-approved",
                "--top-k",
                "3",
                "--no-llm",
                "--debug",
                "--documents-dir",
                str(docs_dir),
                "--audit-logs-dir",
                str(tmp_path / "logs"),
            ]
        )
        assert rc == 0
        cli_out = capsys.readouterr().out
        assert extract_literal_excerpts_from_text(cli_out) == (unit_a1.original_text,)
        assert "Seção 43.10(a)(1)" in cli_out
        assert "PDF p. [9]" in cli_out
        assert "(2) Situação de vida" not in cli_out
        assert "(b) Remoção temporária" not in cli_out
    finally:
        repo.close()


@pytest.mark.integration
def test_fail_closed_multiple_explicit_references_and_missing_root_unit_cases_a_through_g(
    tmp_path: Path,
) -> None:
    """Testes de regressão obrigatórios (PR #21 revisão — Casos A, B, C, D, E, F e G)."""
    from rbac_assistant.application.use_cases.get_normative_section import (
        GetNormativeSectionUseCase,
    )

    repo, _ret, chat, llm, vector_store, embedder, ver_id = _build_synthetic_rbac43_10_stack(
        tmp_path / "base_multi"
    )
    try:
        units_43_10 = repo.get_units_by_section(version_id=ver_id, section_number="43.10")
        par_a_units = [u for u in units_43_10 if u.paragraph_number == "(a)"]
        unit_a1 = next(u for u in par_a_units if u.item_number == "(1)")
        unit_a2 = next(u for u in par_a_units if u.item_number == "(2)")
        filt = SearchFilter(
            document_id="anac_rbac_43",
            regulation_number="43",
            amendment_number="05",
            only_approved_literal=True,
        )

        # Caso A: 43.10(a)(1) válida + 43.10(a)(99) inexistente -> bloqueio integral, 0 LLM
        ans_a = chat.ask(
            question="Transcreva literalmente os itens 43.10(a)(1) e 43.10(a)(99).",
            filters=filt,
            top_k=5,
            generate_explanation=True,
        )
        assert ans_a.found is False
        assert ans_a.is_literal_validated is False
        assert ans_a.normative_text is None
        assert ans_a.evidences == ()
        assert ans_a.has_blocked_sources is True
        assert "Peça com limite de vida" not in ans_a.formatted_text
        assert "43.10(a)(99)" in ans_a.explanation
        assert any("CONSULTA PARCIAL BLOQUEADA" in o for o in ans_a.observations)
        assert ans_a.diagnostics is not None
        assert [r.canonical_label for r in ans_a.diagnostics.resolved_references] == [
            "43.10(a)(1)",
            "43.10(a)(99)",
        ]
        assert any(
            "explicit_ref_not_found(43.10(a)(99))" in d.reason
            for d in ans_a.diagnostics.evidence_decisions
        )
        assert any(
            "partial_multi_ref_blocked" in d.reason
            for d in ans_a.diagnostics.evidence_decisions
        )
        assert llm.calls == []

        # Caso C: Duas referências válidas na mesma seção -> ambas preservadas separadamente
        ans_c = chat.ask(
            question="Transcreva literalmente os itens 43.10(a)(1) e 43.10(a)(2).",
            filters=filt,
            top_k=5,
            generate_explanation=False,
        )
        assert ans_c.found is True
        assert ans_c.is_literal_validated is True
        assert ans_c.normative_text is not None
        assert len(ans_c.evidences) == 2
        assert [ev.unit_id for ev in ans_c.evidences] == [unit_a1.unit_id, unit_a2.unit_id]
        assert extract_literal_excerpts_from_text(ans_c.formatted_text) == (
            unit_a1.original_text,
            unit_a2.original_text,
        )
        assert "(b) Remoção temporária" not in ans_c.formatted_text
        assert llm.calls == []

        # Caso D: 43.10(a)(1) + 43.10(a)(99) usando SOMENTE recuperação lexical (FTS5) -> bloqueio
        lex_ret = HybridNormativeRetriever(
            repository=repo,
            vector_store=vector_store,
            embedding_port=embedder,
            collection_name="test_granular_col",
            validation_service=ValidationService(ValidationPolicy.STRICT_ALL_PAGES),
            enable_structural=False,
            enable_lexical=True,
            enable_vector=False,
        )
        llm_d = _RecordingFakeLLM()
        chat_d = NormativeChatService(retriever=lex_ret, llm=llm_d)
        ans_d = chat_d.ask(
            question="Transcreva literalmente os itens 43.10(a)(1) e 43.10(a)(99).",
            filters=filt,
            top_k=5,
            generate_explanation=True,
        )
        assert ans_d.found is False
        assert ans_d.normative_text is None
        assert ans_d.evidences == ()
        assert ans_d.has_blocked_sources is True
        assert "Peça com limite de vida" not in ans_d.formatted_text
        assert "43.10(a)(99)" in ans_d.explanation
        assert llm_d.calls == []

        # Caso E: 43.10(a)(1) + 43.10(a)(99) usando SOMENTE recuperação vetorial -> bloqueio
        vec_ret = HybridNormativeRetriever(
            repository=repo,
            vector_store=vector_store,
            embedding_port=embedder,
            collection_name="test_granular_col",
            validation_service=ValidationService(ValidationPolicy.STRICT_ALL_PAGES),
            enable_structural=False,
            enable_lexical=False,
            enable_vector=True,
        )
        llm_e = _RecordingFakeLLM()
        chat_e = NormativeChatService(retriever=vec_ret, llm=llm_e)
        ans_e = chat_e.ask(
            question="Transcreva literalmente os itens 43.10(a)(1) e 43.10(a)(99).",
            filters=filt,
            top_k=5,
            generate_explanation=True,
        )
        assert ans_e.found is False
        assert ans_e.normative_text is None
        assert ans_e.evidences == ()
        assert ans_e.has_blocked_sources is True
        assert "Peça com limite de vida" not in ans_e.formatted_text
        assert "43.10(a)(99)" in ans_e.explanation
        assert llm_e.calls == []
    finally:
        repo.close()

    # Caso B1: Uma referência válida (43.10(a)(2)) + outra com divergência literal (43.10(a)(1))
    repo_b1, _r_b1, chat_b1, llm_b1, _vs_b1, _emb_b1, _vid_b1 = _build_synthetic_rbac43_10_stack(
        tmp_path / "case_b_corrupt",
        corrupt_subunit_a1=True,
    )
    try:
        ans_b1 = chat_b1.ask(
            question="Transcreva literalmente os itens 43.10(a)(2) e 43.10(a)(1).",
            generate_explanation=True,
        )
        assert ans_b1.found is False
        assert ans_b1.normative_text is None
        assert ans_b1.evidences == ()
        assert ans_b1.has_blocked_sources is True
        assert "Situação de vida" not in ans_b1.formatted_text
        assert "43.10(a)(1)" in ans_b1.explanation
        assert llm_b1.calls == []
    finally:
        repo_b1.close()

    # Caso B2: Uma referência válida (43.10(a)(2)) + outra ambígua/duplicada (43.10(a)(1))
    repo_b2, _r_b2, chat_b2, llm_b2, _vs_b2, _emb_b2, vid_b2 = _build_synthetic_rbac43_10_stack(
        tmp_path / "case_b_ambig"
    )
    try:
        # Insere duplicata ambígua de 43.10(a)(1) diretamente na base descartável
        with repo_b2._connect() as conn:
            row = conn.execute(
                "SELECT * FROM normative_units "
                "WHERE version_id = ? AND section_number = '43.10' "
                "AND paragraph_number = '(a)' AND item_number = '(1)';",
                (vid_b2,),
            ).fetchone()
            assert row is not None
            cols = list(row.keys())
            vals = [row[c] for c in cols]
            vals[cols.index("unit_id")] = f"{row['unit_id']}_dup"
            vals[cols.index("order_index")] = int(row["order_index"])
            placeholders = ",".join("?" for _ in cols)
            conn.execute(
                f"INSERT INTO normative_units ({', '.join(cols)}) VALUES ({placeholders});",
                vals,
            )
            conn.commit()

        ans_b2 = chat_b2.ask(
            question="Transcreva literalmente os itens 43.10(a)(2) e 43.10(a)(1).",
            generate_explanation=True,
        )
        assert ans_b2.found is False
        assert ans_b2.normative_text is None
        assert ans_b2.evidences == ()
        assert ans_b2.has_blocked_sources is True
        assert "Situação de vida" not in ans_b2.formatted_text
        assert "43.10(a)(1)" in ans_b2.explanation
        assert ans_b2.diagnostics is not None
        assert any(
            d.reason == "excluded:explicit_ref_ambiguous(43.10(a)(1))"
            for d in ans_b2.diagnostics.evidence_decisions
        )
        assert any(
            d.reason == "excluded:partial_multi_ref_blocked"
            for d in ans_b2.diagnostics.evidence_decisions
        )
        assert llm_b2.calls == []
    finally:
        repo_b2.close()

    # Casos F e G: Seção SEM unidade-raiz (`parent_unit_id IS NULL`) no SQLite
    repo_nr, _r_nr, chat_nr, llm_nr, vs_nr, emb_nr, vid_nr = _build_synthetic_rbac43_10_stack(
        tmp_path / "no_root"
    )
    try:
        # Remove exclusivamente o registro-raiz da seção 43.10 mantendo suas subunidades e chunks
        with repo_nr._connect() as conn:
            conn.execute("PRAGMA foreign_keys = OFF;")
            conn.execute(
                "DELETE FROM normative_units "
                "WHERE version_id = ? AND section_number = '43.10' AND parent_unit_id IS NULL;",
                (vid_nr,),
            )
            conn.commit()

        # Garante que restam apenas subunidades de 43.10 (zero unidade-raiz)
        rem_units = repo_nr.get_units_by_section(version_id=vid_nr, section_number="43.10")
        assert len(rem_units) > 0
        assert all(u.parent_unit_id is not None for u in rem_units)

        # Caso F1: Consulta explícita por seção completa sem unidade-raiz -> abstenção segura
        ans_f1 = chat_nr.ask(
            question="Transcreva literalmente a seção 43.10.",
            generate_explanation=True,
        )
        assert ans_f1.found is False
        assert ans_f1.is_literal_validated is False
        assert ans_f1.normative_text is None
        assert ans_f1.evidences == ()
        assert ans_f1.has_blocked_sources is True
        assert "(a) Definições usadas nesta seção" not in ans_f1.formatted_text
        assert llm_nr.calls == []

        # Caso F2a: Consulta lexical-only atingindo 43.10 sem raiz -> bloqueio total
        ret_f2_lex = HybridNormativeRetriever(
            repository=repo_nr,
            vector_store=vs_nr,
            embedding_port=emb_nr,
            collection_name="test_granular_col",
            validation_service=ValidationService(ValidationPolicy.STRICT_ALL_PAGES),
            enable_structural=False,
            enable_lexical=True,
            enable_vector=False,
        )
        llm_f2_lex = _RecordingFakeLLM()
        ans_f2a = NormativeChatService(retriever=ret_f2_lex, llm=llm_f2_lex).ask(
            question="O que significa peça com limite de vida e segregação?",
            generate_explanation=True,
        )
        assert ans_f2a.found is False
        assert ans_f2a.is_literal_validated is False
        assert ans_f2a.normative_text is None
        assert ans_f2a.evidences == ()
        assert ans_f2a.has_blocked_sources is True
        assert "(a) Definições usadas nesta seção" not in ans_f2a.formatted_text
        assert llm_f2_lex.calls == []

        # Caso F2b: Em busca híbrida aberta, 43.10 sem raiz é excluída (nunca retorna subunidade)
        llm_f2_hyb = _RecordingFakeLLM()
        ans_f2b = NormativeChatService(retriever=_r_nr, llm=llm_f2_hyb).ask(
            question="O que significa peça com limite de vida e segregação?",
            generate_explanation=False,
        )
        assert all(ev.section_number != "43.10" for ev in ans_f2b.evidences)
        assert "(a) Definições usadas nesta seção" not in ans_f2b.formatted_text
        assert ans_f2b.diagnostics is not None
        assert any(
            d.section_number == "43.10"
            and d.included is False
            and d.reason == "excluded:missing_root_unit"
            for d in ans_f2b.diagnostics.evidence_decisions
        )

        # Caso F3: Mesmo com reconstruct_full_section_from_ssot=False, bloqueia sem raiz
        ret_no_recon = HybridNormativeRetriever(
            repository=repo_nr,
            vector_store=vs_nr,
            embedding_port=emb_nr,
            collection_name="test_granular_col",
            validation_service=ValidationService(ValidationPolicy.STRICT_ALL_PAGES),
            reconstruct_full_section_from_ssot=False,
        )
        llm_f3 = _RecordingFakeLLM()
        ans_f3 = NormativeChatService(retriever=ret_no_recon, llm=llm_f3).ask(
            question="Transcreva literalmente a seção 43.10.",
            generate_explanation=True,
        )
        assert ans_f3.found is False
        assert ans_f3.normative_text is None
        assert ans_f3.evidences == ()
        assert llm_f3.calls == []

        # Caso F4: GetNormativeSectionUseCase também bloqueia sem unidade-raiz
        sec_uc = GetNormativeSectionUseCase(
            repository=repo_nr,
            catalog_service=CatalogService(repo_nr),
            validation_service=ValidationService(ValidationPolicy.STRICT_ALL_PAGES),
        )
        sec_res = sec_uc.execute(document_id="anac_rbac_43", section_number="43.10")
        assert sec_res.found is False
        assert sec_res.section is None
        assert "unidade-raiz" in sec_res.message.lower()

        # Caso G: Consulta por subunidade (43.10(a)(1)) sem unidade-raiz -> bloqueio literal, 0 LLM
        ans_g = chat_nr.ask(
            question="Transcreva literalmente o item 43.10(a)(1).",
            generate_explanation=True,
        )
        assert ans_g.found is False
        assert ans_g.is_literal_validated is False
        assert ans_g.normative_text is None
        assert ans_g.evidences == ()
        assert ans_g.has_blocked_sources is True
        assert "Peça com limite de vida" not in ans_g.formatted_text
        assert llm_nr.calls == []
    finally:
        repo_nr.close()
