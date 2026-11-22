"""Fixture determinística multi-documento (100% SINTÉTICA) para avaliação controlada do RAG.

IMPORTANTE:
- Todos os textos regulatórios abaixo são FICTÍCIOS/SINTÉTICOS criados exclusivamente para
  testar busca estrutural, lexical (FTS5), vetorial (ChromaDB), fusão RRF, reranking
  determinístico, filtros de regulamento/emenda, ambiguidade de versão, bloqueio de
  OCR/reprovação e validação de citações.
- `DeterministicEvalSemanticEmbedder` é um embedder MECÂNICO/SINTÉTICO para testes
  reproduzíveis no CI; seus números NÃO representam a qualidade real do `gemini-embedding-2`.
- Para benchmark manual com `gemini-embedding-2` real (Free Tier Standard, fora do CI),
  `build_synthetic_eval_environment` aceita um `embedding_port` opcional mantendo exatamente
  o mesmo corpus sintético, consultas, filtros e evidências esperadas.
- Nenhum PDF oficial real da ANAC é lido ou ingerido.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar

from rbac_assistant.application.services.catalog_service import CatalogService
from rbac_assistant.application.services.indexing_service import IndexingService
from rbac_assistant.application.services.validation_service import ValidationService
from rbac_assistant.domain.entities.document_version import DocumentVersion, ExtractionRun
from rbac_assistant.domain.entities.normative_unit import (
    ExtractionMethod,
    NormativeUnitType,
    StructureStatus,
)
from rbac_assistant.domain.entities.rag import RetrievalSource
from rbac_assistant.domain.entities.regulatory_document import (
    DocumentStatus,
    DocumentType,
    RegulatoryDocument,
    ValidationPolicy,
    VersionStatus,
)
from rbac_assistant.domain.entities.regulatory_page import (
    ExtractionMetadata,
    PageValidationStatus,
    QualityMetrics,
    RegulatoryPage,
)
from rbac_assistant.domain.ports.rag_ports import EmbeddingPort
from rbac_assistant.domain.ports.validation_port import ParsedUnitRecord
from rbac_assistant.infrastructure.storage.sqlite_document_repository import (
    SqliteDocumentRepository,
)
from rbac_assistant.infrastructure.vector_db.chroma_vector_store import ChromaVectorStore
from rbac_assistant.rag.evaluation.rag_evaluator import EvaluationCase, ExpectedTarget
from rbac_assistant.rag.ingestion.structural_chunker import StructuralNormativeChunker
from rbac_assistant.rag.retrieval.hybrid_retriever import HybridNormativeRetriever
from tests.unit.test_rag_components import _InMemoryChromaClient

EVAL_DATASET_PATH = Path(__file__).resolve().parent / "rag_eval_dataset.json"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalize(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


class DeterministicEvalSemanticEmbedder:
    """Embedder vetorial determinístico (768d normalizado) apenas para testes mecânicos no CI.

    AVISO METODOLÓGICO:
    - Utiliza hashing determinístico com sinal + pequeno dicionário de sinônimos sintéticos.
    - NÃO é um modelo neural de linguagem e NÃO representa o desempenho real do
      `gemini-embedding-2`.
    - Espelha o contrato de entrada do `GeminiEmbeddingAdapter`, combinando `title` e `text`
      em memória durante `embed_documents` (sem alterar o texto persistido no SQLite).
    """

    _CONCEPT_MAP: ClassVar[dict[str, tuple[str, float]]] = {
        # Pessoas autorizadas a executar manutenção (43.3)
        "mecanico": ("c_exec_manutencao", 2.6),
        "habilitado": ("c_exec_manutencao", 2.2),
        "autorizadas": ("c_exec_manutencao", 2.0),
        "autorizado": ("c_exec_manutencao", 2.0),
        "executar": ("c_exec_manutencao", 1.8),
        "profissional": ("c_exec_manutencao", 2.2),
        "tecnico": ("c_exec_manutencao", 1.8),
        "piloto": ("c_exec_manutencao", 1.8),
        # Retorno ao serviço (43.7 / 145.201)
        "retorno": ("c_retorno_servico", 2.8),
        "liberar": ("c_retorno_servico", 2.8),
        "liberacao": ("c_retorno_servico", 2.8),
        "voar": ("c_retorno_servico", 2.4),
        "voo": ("c_retorno_servico", 2.2),
        "inspetor": ("c_retorno_servico", 2.4),
        "aprovar": ("c_retorno_servico", 2.0),
        "prerrogativas": ("c_prerrogativas_om", 3.0),
        "poderes": ("c_prerrogativas_om", 3.0),
        "componentes": ("c_prerrogativas_om", 2.6),
        "artigos": ("c_prerrogativas_om", 2.2),
        # Registros de manutenção (43.9)
        "registros": ("c_registro_manutencao", 2.4),
        "registro": ("c_registro_manutencao", 2.4),
        "caderneta": ("c_registro_manutencao", 3.0),
        "bordo": ("c_registro_manutencao", 2.6),
        "anotacao": ("c_registro_manutencao", 3.0),
        "rubrica": ("c_registro_manutencao", 2.8),
        "assinatura": ("c_registro_manutencao", 2.8),
        "licenca": ("c_registro_manutencao", 2.4),
        "conclusao": ("c_registro_manutencao", 2.2),
        "finalizar": ("c_registro_manutencao", 2.2),
        "lanco": ("c_registro_manutencao", 2.2),
        # Guarda / retenção documental 5 anos (145.219)
        "guarda": ("c_retencao_om", 3.2),
        "arquivamento": ("c_retencao_om", 3.2),
        "retencao": ("c_retencao_om", 3.2),
        "arquivos": ("c_retencao_om", 2.6),
        "cinco": ("c_retencao_om", 2.4),
        "anos": ("c_retencao_om", 2.4),
        "prazo": ("c_retencao_om", 2.0),
        "periodo": ("c_retencao_om", 2.2),
        # Oficina / Organização de Manutenção (145)
        "oficina": ("c_om_145", 2.6),
        "hangar": ("c_om_145", 2.6),
        "homologada": ("c_om_145", 2.6),
        "certificada": ("c_om_145", 1.8),
        "organizacao": ("c_om_145", 1.8),
        # Ferramentas calibradas e equipamentos (145.109)
        "ferramentas": ("c_ferramental_145", 3.0),
        "ferramental": ("c_ferramental_145", 3.0),
        "instrumentos": ("c_ferramental_145", 3.0),
        "calibradas": ("c_ferramental_145", 3.0),
        "aferidos": ("c_ferramental_145", 3.0),
        "equipamentos": ("c_ferramental_145", 2.8),
        "materiais": ("c_ferramental_145", 2.2),
        # Grandes reparos e manutenção preventiva (Apêndice A)
        "pneu": ("c_apendice_a", 3.0),
        "pneus": ("c_apendice_a", 3.0),
        "bateria": ("c_apendice_a", 3.0),
        "baterias": ("c_apendice_a", 3.0),
        "trocar": ("c_apendice_a", 2.6),
        "troca": ("c_apendice_a", 2.6),
        "simples": ("c_apendice_a", 2.4),
        "preventiva": ("c_apendice_a", 2.2),
        "classificacao": ("c_apendice_a", 2.6),
        "conserto": ("c_apendice_a", 2.4),
        "estrutural": ("c_apendice_a", 3.0),
        "fuselagem": ("c_apendice_a", 3.0),
        "celula": ("c_apendice_a", 3.0),
        "reparo": ("c_apendice_a", 2.4),
        # Aplicabilidade / Experimentais (43.1)
        "experimentais": ("c_aplicabilidade_43", 3.0),
        "aplicabilidade": ("c_aplicabilidade_43", 2.2),
        "exclusao": ("c_aplicabilidade_43", 2.6),
        # Definições / Responsável técnico (145.3)
        "definicoes": ("c_definicoes_145", 3.0),
        "responsavel": ("c_definicoes_145", 2.4),
        # Vigência Resolução 700 (Art. 2º)
        "vigor": ("c_vigencia_res", 3.2),
        "vigencia": ("c_vigencia_res", 3.2),
        "revogacao": ("c_vigencia_res", 2.8),
        "entra": ("c_vigencia_res", 2.4),
    }

    _STOPWORDS: ClassVar[frozenset[str]] = frozenset(
        {
            "a",
            "o",
            "e",
            "de",
            "do",
            "da",
            "dos",
            "das",
            "em",
            "no",
            "na",
            "nos",
            "nas",
            "para",
            "por",
            "com",
            "um",
            "uma",
            "que",
            "qual",
            "quais",
            "quem",
            "como",
            "onde",
            "quando",
            "sobre",
            "se",
            "ao",
            "os",
            "as",
            "ou",
            "sao",
            "esta",
            "deve",
            "devo",
            "pode",
            "podem",
            "saber",
            "preciso",
            "conta",
            "depois",
            "novo",
        }
    )

    def __init__(self, *, dimensions: int = 768) -> None:
        self._dimensions = dimensions
        self._token_hash_cache: dict[str, tuple[int, float]] = {}
        self._concept_hash_cache: dict[str, tuple[int, float]] = {}

    @property
    def model_name(self) -> str:
        return "eval-deterministic-embed-v1"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def instruction_strategy_version(self) -> str:
        return "ge2_asymmetric_v1"

    def _encode(self, text: str) -> tuple[float, ...]:
        ascii_text = _normalize(text)
        tokens = [
            tok for tok in re.findall(r"[a-z0-9]+", ascii_text) if tok not in self._STOPWORDS
        ]
        vec = [0.0] * self._dimensions
        tok_cache = self._token_hash_cache
        concept_cache = self._concept_hash_cache
        dims = self._dimensions
        for tok in tokens:
            tok_entry = tok_cache.get(tok)
            if tok_entry is None:
                tok_digest = hashlib.sha256(f"tok:{tok}".encode()).digest()
                tok_entry = (
                    int.from_bytes(tok_digest[:2], "big") % dims,
                    1.0 if (tok_digest[2] & 1) == 0 else -1.0,
                )
                tok_cache[tok] = tok_entry
            vec[tok_entry[0]] += tok_entry[1]
            mapped = self._CONCEPT_MAP.get(tok)
            if mapped is not None:
                concept_name, weight = mapped
                c_entry = concept_cache.get(concept_name)
                if c_entry is None:
                    c_digest = hashlib.sha256(f"concept:{concept_name}".encode()).digest()
                    c_entry = (
                        int.from_bytes(c_digest[:2], "big") % dims,
                        1.0 if (c_digest[2] & 1) == 0 else -1.0,
                    )
                    concept_cache[concept_name] = c_entry
                vec[c_entry[0]] += c_entry[1] * weight

        norm = math.sqrt(math.sumprod(vec, vec)) or 1.0
        inv_norm = 1.0 / norm
        return tuple(v * inv_norm for v in vec)

    def embed_documents(
        self,
        texts: Sequence[str],
        *,
        titles: Sequence[str | None] | None = None,
    ) -> list[tuple[float, ...]]:
        # Espelha o contrato do GeminiEmbeddingAdapter combinando título + corpo em memória
        if titles is not None:
            combined = [
                f"{title}\n{body}" if title else body
                for body, title in zip(texts, titles, strict=True)
            ]
        else:
            combined = list(texts)
        return [self._encode(item) for item in combined]

    def embed_query(self, text: str) -> tuple[float, ...]:
        return self._encode(text)


def load_eval_dataset(path: Path = EVAL_DATASET_PATH) -> tuple[EvaluationCase, ...]:
    """Carrega os casos controlados de avaliação com granularidade normativa completa."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    cases: list[EvaluationCase] = []
    for item in raw["cases"]:
        targets = tuple(
            ExpectedTarget(
                regulation_number=str(t["regulation_number"]),
                amendment_number=str(t["amendment_number"]),
                section_number=str(t["section_number"]),
                subsection_number=str(t.get("subsection_number", "")),
                paragraph_number=str(t.get("paragraph_number", "")),
                alinea=str(t.get("alinea", "")),
                item_number=str(t.get("item_number", "")),
                subitem_number=str(t.get("subitem_number", "")),
                normative_unit_id=str(t.get("normative_unit_id", "")),
                expected_pdf_pages=tuple(int(p) for p in t.get("expected_pdf_pages", ())),
                expected_text_snippets=tuple(str(s) for s in t.get("expected_text_snippets", ())),
            )
            for t in item.get("expected_targets", ())
        )
        cases.append(
            EvaluationCase(
                case_id=str(item["case_id"]),
                category=str(item["category"]),
                structural_subset=str(item.get("structural_subset", "natural_language_no_ref")),
                query=str(item["query"]),
                expected_targets=targets,
                expected_behavior=str(item["expected_behavior"]),
                is_exact_structural=bool(item.get("is_exact_structural", False)),
                expected_regulation=item.get("expected_regulation"),
                expected_amendment=item.get("expected_amendment"),
                regulation_filter=item.get("regulation_filter"),
                amendment_filter=item.get("amendment_filter"),
                document_type_filter=item.get("document_type_filter"),
                only_current_version=bool(item.get("only_current_version", True)),
            )
        )
    return tuple(cases)


def _make_page(
    *,
    version_id: str,
    pdf_page_number: int,
    printed_page: str,
    text: str,
    status: PageValidationStatus,
    method: ExtractionMethod = ExtractionMethod.NATIVE_PDF,
) -> RegulatoryPage:
    return RegulatoryPage(
        page_id=f"{version_id}_p{pdf_page_number}",
        version_id=version_id,
        pdf_page_number=pdf_page_number,
        printed_page_number=printed_page,
        original_text=text,
        normalized_text=_normalize(text),
        extraction_method=method,
        validation_status=status,
        content_hash=_sha(text),
        extraction_metadata=ExtractionMetadata(
            extraction_method=method,
            extraction_timestamp=datetime.now(UTC),
            extraction_version="synthetic_eval_1.0",
            content_hash=_sha(text),
            quality_metrics=QualityMetrics(
                char_count=len(text),
                line_count=max(1, text.count("\n") + 1),
                word_count=len(text.split()),
                has_replacement_chars=False,
                has_mojibake_signals=False,
                has_portuguese_accents=True,
                unbalanced_parentheses=False,
                is_empty_or_incomplete=False,
                passed_automated_checks=True,
            ),
        ),
    )


def _make_unit(
    *,
    version_id: str,
    section_number: str,
    title: str,
    text: str,
    page_numbers: tuple[int, ...],
    order_index: int,
    parent_unit_id: str | None = None,
    unit_id_override: str | None = None,
    paragraph_number: str = "",
    item_number: str = "",
    unit_type: NormativeUnitType = NormativeUnitType.SECTION,
) -> ParsedUnitRecord:
    uid = unit_id_override or f"{version_id}_u_{order_index}_{section_number.replace(' ', '_')}"
    return ParsedUnitRecord(
        unit_id=uid,
        version_id=version_id,
        parent_unit_id=parent_unit_id,
        unit_type=unit_type,
        section_number=section_number,
        subsection_number="",
        paragraph_number=paragraph_number,
        item_number=item_number,
        title=title,
        subpart="Sintética",
        original_text=text,
        normalized_text=_normalize(text),
        start_page=min(page_numbers),
        end_page=max(page_numbers),
        page_numbers=page_numbers,
        order_index=order_index,
        structure_status=StructureStatus.PARSED,
        content_hash=_sha(text),
    )


def build_synthetic_eval_environment(
    tmp_path: Path,
    *,
    embedding_port: EmbeddingPort | None = None,
    collection_name: str = "eval_synthetic_ge2_768_v1",
) -> tuple[
    SqliteDocumentRepository,
    ChromaVectorStore,
    EmbeddingPort,
    Callable[..., HybridNormativeRetriever],
]:
    """Constrói base SQLite + ChromaDB em memória com 4 documentos sintéticos controlados."""
    repo = SqliteDocumentRepository(tmp_path / "eval_catalog.sqlite3", reuse_connection=True)
    repo.initialize()
    catalog = CatalogService(repo)
    validation = ValidationService(policy=ValidationPolicy.REVIEW_ON_DEMAND)
    vector_store = ChromaVectorStore(client=_InMemoryChromaClient())
    embedder: EmbeddingPort = embedding_port or DeterministicEvalSemanticEmbedder(dimensions=768)
    now = datetime.now(UTC)

    def _save_bundle(
        *,
        doc: RegulatoryDocument,
        ver: DocumentVersion,
        pages: Sequence[RegulatoryPage],
        units: Sequence[ParsedUnitRecord],
    ) -> None:
        run = ExtractionRun(
            run_id=f"run_{ver.version_id}",
            document_id=doc.document_id,
            version_id=ver.version_id,
            started_at=now,
            finished_at=now,
            status="succeeded",
            extraction_version="synthetic_eval_1.0",
            source_hash=ver.source_hash,
            pages_extracted=len(pages),
            units_created=len(units),
        )
        repo.save_version_ingestion(
            document=doc,
            version=ver,
            pages=pages,
            units=units,
            tables=(),
            extraction_run=run,
        )

    # ------------------------------------------------------------------
    # Documento 1: RBAC 43 Sintético (Emenda 05 vigente + Emenda 04 histórica)
    # ------------------------------------------------------------------
    doc_43 = RegulatoryDocument(
        document_id="eval_rbac_43",
        authority="ANAC",
        document_type=DocumentType.RBAC,
        regulation_number="43",
        document_title="Manutenção, Manutenção Preventiva, Reconstrução e Alteração (Sintético)",
        document_status=DocumentStatus.ACTIVE,
        version_status=VersionStatus.CURRENT,
        amendment_number="05",
        official_url="https://example.test/eval/rbac-43",
    )

    p43_2 = (
        "43.1 Aplicabilidade\n"
        "(a) Este regulamento estabelece regras que regem a manutenção, manutenção preventiva, "
        "reconstrução e alteração de aeronaves civis registradas no Brasil, com exclusão de "
        "aeronaves experimentais.\n\n"
        "43.3 Pessoas autorizadas a executar manutenção, manutenção preventiva, reconstrução "
        "e alteração\n"
        "(a) Exceto como previsto nesta seção, nenhuma pessoa pode executar manutenção em "
        "aeronave civil sem habilitação.\n"
        "(a)(1) O mecânico de manutenção aeronáutica habilitado pode executar manutenção "
        "conforme suas habilitações na licença ANAC."
    )
    p43_3 = (
        "(b) Uma organização de manutenção certificada pode executar manutenção, manutenção "
        "preventiva, reconstrução e alteração conforme suas especificações operativas.\n"
        "(c) O titular de licença de piloto privado pode executar manutenção preventiva em "
        "aeronave de sua propriedade.\n\n"
        "43.7 Pessoas autorizadas a aprovar o retorno ao serviço após manutenção\n"
        "(a) Somente o inspetor certificado, a organização de manutenção aprovada ou o titular "
        "de certificado de tipo podem aprovar o retorno ao serviço de aeronave após manutenção."
    )
    p43_4 = (
        "43.9 Conteúdo, formato e disposição dos registros de manutenção\n"
        "(a) Cada pessoa que executa manutenção deve registrar na caderneta de manutenção:\n"
        "(a)(1) Descrição detalhada do trabalho executado e dados técnicos aprovados;\n"
        "(a)(1)(i) Data de conclusão do serviço, nome completo, assinatura e número da licença "
        "ANAC do responsável técnico."
    )
    p43_5 = (
        "43.11 Conteúdo, formato e disposição dos registros de inspeções de 100 horas e anuais\n"
        "(a) A pessoa que aprova ou reprova uma aeronave em inspeção de 100 horas ou inspeção "
        "anual deve emitir lista formal de discrepâncias de aeronavegabilidade."
    )
    p43_6 = (
        "43.13 Regras de execução de manutenção (página OCR pendente)\n"
        "(a) Cada pessoa executando manutenção deve utilizar métodos, técnicas e práticas "
        "aceitáveis previstos no manual do fabricante e ferramentas calibradas."
    )
    p43_7 = (
        "43.15 Regras adicionais de execução para inspeções em helicópteros\n"
        "(a) Inspeções periódicas de rotores principais e rotores de cauda exigem verificação "
        "dimensional específica (página reprovada na revisão humana)."
    )
    p43_8 = (
        "Apêndice A Grandes alterações, grandes reparos e manutenção preventiva\n"
        "(a) Classifica-se como grande reparo qualquer reparo estrutural na célula ou fuselagem "
        "que afete resistência primária.\n"
        "(c) Classifica-se como manutenção preventiva a troca de pneus e a substituição simples "
        "de baterias de aeronave."
    )

    v43_05 = DocumentVersion(
        version_id="eval_rbac_43_v05",
        document_id="eval_rbac_43",
        amendment_number="05",
        version_status=VersionStatus.CURRENT,
        source_hash=_sha("eval_rbac_43_v05"),
        page_count=7,
        pdf_filename="eval_rbac_43_emd05.pdf",
        official_url="https://example.test/eval/rbac-43-emd05",
    )
    pages_43_05 = [
        _make_page(
            version_id=v43_05.version_id,
            pdf_page_number=2,
            printed_page="2/10",
            text=p43_2,
            status=PageValidationStatus.APPROVED_FOR_LITERAL_TRANSCRIPTION,
        ),
        _make_page(
            version_id=v43_05.version_id,
            pdf_page_number=3,
            printed_page="3/10",
            text=p43_3,
            status=PageValidationStatus.APPROVED_FOR_LITERAL_TRANSCRIPTION,
        ),
        _make_page(
            version_id=v43_05.version_id,
            pdf_page_number=4,
            printed_page="4/10",
            text=p43_4,
            status=PageValidationStatus.APPROVED_FOR_LITERAL_TRANSCRIPTION,
        ),
        _make_page(
            version_id=v43_05.version_id,
            pdf_page_number=5,
            printed_page="5/10",
            text=p43_5,
            status=PageValidationStatus.NATIVE_QUALITY_CHECKED,
        ),
        _make_page(
            version_id=v43_05.version_id,
            pdf_page_number=6,
            printed_page="6/10",
            text=p43_6,
            status=PageValidationStatus.PENDING_MANUAL_REVIEW,
            method=ExtractionMethod.OCR_FALLBACK,
        ),
        _make_page(
            version_id=v43_05.version_id,
            pdf_page_number=7,
            printed_page="7/10",
            text=p43_7,
            status=PageValidationStatus.REJECTED,
        ),
        _make_page(
            version_id=v43_05.version_id,
            pdf_page_number=8,
            printed_page="8/10",
            text=p43_8,
            status=PageValidationStatus.APPROVED_FOR_LITERAL_TRANSCRIPTION,
        ),
    ]
    u43_3_id = "eval_rbac_43_v05_u_2_43.3"
    u43_9_id = "eval_rbac_43_v05_u_4_43.9"
    units_43_05 = [
        _make_unit(
            version_id=v43_05.version_id,
            section_number="43.1",
            title="Aplicabilidade",
            text=(
                "43.1 Aplicabilidade\n"
                "(a) Este regulamento estabelece regras que regem a manutenção, manutenção "
                "preventiva, reconstrução e alteração de aeronaves civis registradas no Brasil, "
                "com exclusão de aeronaves experimentais."
            ),
            page_numbers=(2,),
            order_index=1,
        ),
        _make_unit(
            version_id=v43_05.version_id,
            section_number="43.3",
            title="Pessoas autorizadas a executar manutenção",
            text=(
                "43.3 Pessoas autorizadas a executar manutenção, manutenção preventiva, "
                "reconstrução e alteração\n"
                "(a) Exceto como previsto nesta seção, nenhuma pessoa pode executar manutenção em "
                "aeronave civil sem habilitação.\n"
                "(a)(1) O mecânico de manutenção aeronáutica habilitado pode executar manutenção "
                "conforme suas habilitações na licença ANAC.\n"
                "(b) Uma organização de manutenção certificada pode executar manutenção, "
                "manutenção preventiva, reconstrução e alteração conforme suas especificações.\n"
                "(c) O titular de licença de piloto privado pode executar manutenção preventiva "
                "em aeronave de sua propriedade."
            ),
            page_numbers=(2, 3),
            order_index=2,
            unit_id_override=u43_3_id,
        ),
        # Unidades filhas hierárquicas (parágrafo/item/subitem) espelhando AnacRbacStructureStrategy
        _make_unit(
            version_id=v43_05.version_id,
            parent_unit_id=u43_3_id,
            unit_id_override="eval_rbac_43_v05_u_2_43.3_a_1",
            unit_type=NormativeUnitType.ALINEA,
            section_number="43.3",
            paragraph_number="(a)",
            item_number="(1)",
            title="Pessoas autorizadas — 43.3(a)(1)",
            text=(
                "(a)(1) O mecânico de manutenção aeronáutica habilitado pode executar manutenção "
                "conforme suas habilitações na licença ANAC."
            ),
            page_numbers=(2,),
            order_index=21,
        ),
        _make_unit(
            version_id=v43_05.version_id,
            section_number="43.7",
            title="Pessoas autorizadas a aprovar o retorno ao serviço",
            text=(
                "43.7 Pessoas autorizadas a aprovar o retorno ao serviço após manutenção\n"
                "(a) Somente o inspetor certificado, a organização de manutenção aprovada ou o "
                "titular de certificado de tipo podem aprovar o retorno ao serviço de aeronave "
                "após manutenção."
            ),
            page_numbers=(3,),
            order_index=3,
        ),
        _make_unit(
            version_id=v43_05.version_id,
            section_number="43.9",
            title="Conteúdo, formato e disposição dos registros de manutenção",
            text=p43_4,
            page_numbers=(4,),
            order_index=4,
            unit_id_override=u43_9_id,
        ),
        _make_unit(
            version_id=v43_05.version_id,
            parent_unit_id=u43_9_id,
            unit_id_override="eval_rbac_43_v05_u_4_43.9_a",
            unit_type=NormativeUnitType.PARAGRAPH,
            section_number="43.9",
            paragraph_number="(a)",
            title="Registros de manutenção — 43.9(a)",
            text=(
                "(a) Cada pessoa que executa manutenção deve registrar na caderneta de manutenção:"
            ),
            page_numbers=(4,),
            order_index=41,
        ),
        _make_unit(
            version_id=v43_05.version_id,
            parent_unit_id=u43_9_id,
            unit_id_override="eval_rbac_43_v05_u_4_43.9_a_1_i",
            unit_type=NormativeUnitType.SUBITEM,
            section_number="43.9",
            paragraph_number="(a)",
            item_number="(1)",
            title="Registros de manutenção — 43.9(a)(1)(i)",
            text=(
                "(a)(1)(i) Data de conclusão do serviço, nome completo, assinatura e número da "
                "licença ANAC do responsável técnico."
            ),
            page_numbers=(4,),
            order_index=42,
        ),
        _make_unit(
            version_id=v43_05.version_id,
            section_number="43.11",
            title="Registros de inspeções de 100 horas e anuais",
            text=p43_5,
            page_numbers=(5,),
            order_index=5,
        ),
        _make_unit(
            version_id=v43_05.version_id,
            section_number="43.13",
            title="Regras de execução de manutenção",
            text=p43_6,
            page_numbers=(6,),
            order_index=6,
        ),
        _make_unit(
            version_id=v43_05.version_id,
            section_number="43.15",
            title="Regras adicionais para helicópteros",
            text=p43_7,
            page_numbers=(7,),
            order_index=7,
        ),
        _make_unit(
            version_id=v43_05.version_id,
            section_number="Apêndice A",
            title="Grandes alterações, grandes reparos e manutenção preventiva",
            text=p43_8,
            page_numbers=(8,),
            order_index=8,
            unit_type=NormativeUnitType.APPENDIX,
        ),
    ]
    _save_bundle(doc=doc_43, ver=v43_05, pages=pages_43_05, units=units_43_05)

    # RBAC 43 Emenda 04 (histórica)
    p43_04_txt = (
        "43.3 Pessoas autorizadas a executar manutenção (Emenda 04 histórica)\n"
        "(a) Sob a redação histórica anterior da Emenda 04, o registro em formulário físico "
        "SEGVOO 001 era obrigatório para toda grande alteração."
    )
    v43_04 = DocumentVersion(
        version_id="eval_rbac_43_v04",
        document_id="eval_rbac_43",
        amendment_number="04",
        version_status=VersionStatus.HISTORICAL,
        source_hash=_sha("eval_rbac_43_v04"),
        page_count=1,
        pdf_filename="eval_rbac_43_emd04.pdf",
        official_url="https://example.test/eval/rbac-43-emd04",
    )
    _save_bundle(
        doc=doc_43,
        ver=v43_04,
        pages=[
            _make_page(
                version_id=v43_04.version_id,
                pdf_page_number=2,
                printed_page="2/8",
                text=p43_04_txt,
                status=PageValidationStatus.APPROVED_FOR_LITERAL_TRANSCRIPTION,
            )
        ],
        units=[
            _make_unit(
                version_id=v43_04.version_id,
                section_number="43.3",
                title="Pessoas autorizadas (Emenda 04 histórica)",
                text=p43_04_txt,
                page_numbers=(2,),
                order_index=1,
            )
        ],
    )

    # ------------------------------------------------------------------
    # Documento 2: RBAC 145 Sintético (Emenda 06 vigente — numeração sobreposta!)
    # ------------------------------------------------------------------
    doc_145 = RegulatoryDocument(
        document_id="eval_rbac_145",
        authority="ANAC",
        document_type=DocumentType.RBAC,
        regulation_number="145",
        document_title="Organizações de Manutenção de Produto Aeronáutico (Sintético)",
        document_status=DocumentStatus.ACTIVE,
        version_status=VersionStatus.CURRENT,
        amendment_number="06",
        official_url="https://example.test/eval/rbac-145",
    )
    v145_06 = DocumentVersion(
        version_id="eval_rbac_145_v06",
        document_id="eval_rbac_145",
        amendment_number="06",
        version_status=VersionStatus.CURRENT,
        source_hash=_sha("eval_rbac_145_v06"),
        page_count=4,
        pdf_filename="eval_rbac_145_emd06.pdf",
        official_url="https://example.test/eval/rbac-145-emd06",
    )
    u145_1 = (
        "145.1 Aplicabilidade\n"
        "(a) Este regulamento estabelece os requisitos para certificação de organizações de "
        "manutenção de produto aeronáutico sediadas no Brasil ou no exterior."
    )
    u145_3 = (
        "145.3 Definições\n"
        "(a) Para fins do RBAC 145, define-se responsável técnico e artigos aeronáuticos "
        "submetidos à manutenção em oficina certificada."
    )
    u145_109 = (
        "145.109 Requisitos de equipamentos, materiais e dados técnicos\n"
        "(a) A organização de manutenção certificada deve dispor de equipamentos, ferramentas "
        "calibradas, materiais e dados técnicos atualizados no hangar."
    )
    u145_201 = (
        "145.201 Prerrogativas e limitações do certificado\n"
        "(a) Uma organização de manutenção homologada possui prerrogativas para aprovar o "
        "retorno ao serviço de componentes e artigos aeronáuticos dentro de seus padrões."
    )
    u145_219 = (
        "145.219 Registros de manutenção e guarda documental\n"
        "(a) A organização de manutenção certificada deve manter os registros de manutenção em "
        "arquivamento seguro pelo prazo mínimo de cinco anos para auditoria da ANAC."
    )
    _save_bundle(
        doc=doc_145,
        ver=v145_06,
        pages=[
            _make_page(
                version_id=v145_06.version_id,
                pdf_page_number=2,
                printed_page="2/12",
                text=f"{u145_1}\n\n{u145_3}",
                status=PageValidationStatus.APPROVED_FOR_LITERAL_TRANSCRIPTION,
            ),
            _make_page(
                version_id=v145_06.version_id,
                pdf_page_number=3,
                printed_page="3/12",
                text=u145_109,
                status=PageValidationStatus.APPROVED_FOR_LITERAL_TRANSCRIPTION,
            ),
            _make_page(
                version_id=v145_06.version_id,
                pdf_page_number=4,
                printed_page="4/12",
                text=u145_201,
                status=PageValidationStatus.APPROVED_FOR_LITERAL_TRANSCRIPTION,
            ),
            _make_page(
                version_id=v145_06.version_id,
                pdf_page_number=5,
                printed_page="5/12",
                text=u145_219,
                status=PageValidationStatus.APPROVED_FOR_LITERAL_TRANSCRIPTION,
            ),
        ],
        units=[
            _make_unit(
                version_id=v145_06.version_id,
                section_number="145.1",
                title="Aplicabilidade OM",
                text=u145_1,
                page_numbers=(2,),
                order_index=1,
            ),
            _make_unit(
                version_id=v145_06.version_id,
                section_number="145.3",
                title="Definições OM",
                text=u145_3,
                page_numbers=(2,),
                order_index=2,
            ),
            _make_unit(
                version_id=v145_06.version_id,
                section_number="145.109",
                title="Requisitos de equipamentos e ferramentas calibradas",
                text=u145_109,
                page_numbers=(3,),
                order_index=3,
            ),
            _make_unit(
                version_id=v145_06.version_id,
                section_number="145.201",
                title="Prerrogativas e limitações do certificado",
                text=u145_201,
                page_numbers=(4,),
                order_index=4,
            ),
            _make_unit(
                version_id=v145_06.version_id,
                section_number="145.219",
                title="Registros de manutenção e guarda documental",
                text=u145_219,
                page_numbers=(5,),
                order_index=5,
            ),
        ],
    )

    # ------------------------------------------------------------------
    # Documento 3: RBAC 88 Sintético (versões ambíguas — SEM versão current!)
    # ------------------------------------------------------------------
    doc_88 = RegulatoryDocument(
        document_id="eval_rbac_88",
        authority="ANAC",
        document_type=DocumentType.RBAC,
        regulation_number="88",
        document_title="Regulamento Sintético de Versões Ambíguas",
        document_status=DocumentStatus.ACTIVE,
        version_status=VersionStatus.HISTORICAL,
        amendment_number="01",
        official_url="https://example.test/eval/rbac-88",
    )
    for amd in ("01", "02"):
        v88 = DocumentVersion(
            version_id=f"eval_rbac_88_v{amd}",
            document_id="eval_rbac_88",
            amendment_number=amd,
            version_status=VersionStatus.HISTORICAL,
            source_hash=_sha(f"eval_rbac_88_v{amd}"),
            page_count=1,
            pdf_filename=f"eval_rbac_88_emd{amd}.pdf",
            official_url=f"https://example.test/eval/rbac-88-emd{amd}",
        )
        t88 = f"88.5 Regra eletrônica fictícia da Emenda {amd} (versão histórica)."
        _save_bundle(
            doc=doc_88,
            ver=v88,
            pages=[
                _make_page(
                    version_id=v88.version_id,
                    pdf_page_number=1,
                    printed_page="1/2",
                    text=t88,
                    status=PageValidationStatus.APPROVED_FOR_LITERAL_TRANSCRIPTION,
                )
            ],
            units=[
                _make_unit(
                    version_id=v88.version_id,
                    section_number="88.5",
                    title=f"Regra eletrônica Emenda {amd}",
                    text=t88,
                    page_numbers=(1,),
                    order_index=1,
                )
            ],
        )

    # ------------------------------------------------------------------
    # Documento 4: Resolução 700 Sintética (tipo RESOLUCAO)
    # ------------------------------------------------------------------
    doc_res = RegulatoryDocument(
        document_id="eval_res_700",
        authority="ANAC",
        document_type=DocumentType.RESOLUCAO,
        regulation_number="700",
        document_title="Resolução Sintética 700 — Publicação Normativa",
        document_status=DocumentStatus.ACTIVE,
        version_status=VersionStatus.CURRENT,
        amendment_number="00",
        official_url="https://example.test/eval/res-700",
    )
    v_res = DocumentVersion(
        version_id="eval_res_700_v00",
        document_id="eval_res_700",
        amendment_number="00",
        version_status=VersionStatus.CURRENT,
        source_hash=_sha("eval_res_700_v00"),
        page_count=1,
        pdf_filename="eval_res_700.pdf",
        official_url="https://example.test/eval/res-700",
    )
    art1 = (
        "Art. 1º Fica aprovada a Emenda 05 ao RBAC 43 sintético para validação da arquitetura "
        "multidocumento de normas da ANAC."
    )
    art2 = (
        "Art. 2º Esta Resolução entra em vigor trinta dias após sua publicação oficial, "
        "revogando disposições anteriores em contrário."
    )
    _save_bundle(
        doc=doc_res,
        ver=v_res,
        pages=[
            _make_page(
                version_id=v_res.version_id,
                pdf_page_number=1,
                printed_page="1/1",
                text=f"{art1}\n\n{art2}",
                status=PageValidationStatus.APPROVED_FOR_LITERAL_TRANSCRIPTION,
            )
        ],
        units=[
            _make_unit(
                version_id=v_res.version_id,
                section_number="Art. 1º",
                title="Aprovação da Emenda 05",
                text=art1,
                page_numbers=(1,),
                order_index=1,
                unit_type=NormativeUnitType.SECTION,
            ),
            _make_unit(
                version_id=v_res.version_id,
                section_number="Art. 2º",
                title="Vigência e revogação",
                text=art2,
                page_numbers=(1,),
                order_index=2,
                unit_type=NormativeUnitType.SECTION,
            ),
        ],
    )

    indexing = IndexingService(
        repository=repo,
        catalog_service=catalog,
        chunker=StructuralNormativeChunker(),
        vector_store=vector_store,
        embedding_port=embedder,
        default_collection_name=collection_name,
    )
    for doc_id, ver_id in (
        ("eval_rbac_43", "eval_rbac_43_v05"),
        ("eval_rbac_43", "eval_rbac_43_v04"),
        ("eval_rbac_145", "eval_rbac_145_v06"),
        ("eval_rbac_88", "eval_rbac_88_v01"),
        ("eval_rbac_88", "eval_rbac_88_v02"),
        ("eval_res_700", "eval_res_700_v00"),
    ):
        indexing.index_document(document_id=doc_id, version_id=ver_id)

    def _retriever_factory(
        mode: str,
        max_vector_distance: float = 0.85,
        *,
        lexical_top_k: int = 5,
        semantic_top_k: int = 5,
        rrf_k: int = 60,
        source_weights: Mapping[RetrievalSource, float] | None = None,
        enable_reranker: bool | None = None,
        enable_adaptive_cutoff: bool | None = None,
        adaptive_cutoff_ratio: float = 0.55,
    ) -> HybridNormativeRetriever:
        if mode == "structural_only":
            flags = (True, False, False)
            default_rerank = False
            default_cutoff = False
        elif mode == "lexical_fts5_only":
            flags = (False, True, False)
            default_rerank = False
            default_cutoff = False
        elif mode == "vector_only":
            flags = (False, False, True)
            default_rerank = False
            default_cutoff = False
        elif mode == "hybrid_rrf":
            flags = (True, True, True)
            default_rerank = True
            default_cutoff = True
        else:
            raise ValueError(f"Modo de avaliação desconhecido: {mode}")

        return HybridNormativeRetriever(
            repository=repo,
            vector_store=vector_store,
            embedding_port=embedder,
            collection_name=collection_name,
            validation_service=validation,
            lexical_top_k=lexical_top_k,
            semantic_top_k=semantic_top_k,
            final_top_k=5,
            rrf_k=rrf_k,
            max_vector_distance=max_vector_distance,
            enable_structural=flags[0],
            enable_lexical=flags[1],
            enable_vector=flags[2],
            source_weights=source_weights,
            enable_reranker=default_rerank if enable_reranker is None else enable_reranker,
            enable_adaptive_cutoff=(
                default_cutoff if enable_adaptive_cutoff is None else enable_adaptive_cutoff
            ),
            adaptive_cutoff_ratio=adaptive_cutoff_ratio,
        )

    return repo, vector_store, embedder, _retriever_factory
