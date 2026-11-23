"""Avaliação reproduzível e calibração de limiar sobre corpus regulatório real aprovado.

Garantias metodológicas:
- Distingue explicitamente `corpus_origin` (`official_approved_corpus` vs.
  `synthetic_controlled_fixture`), impedindo que métricas sintéticas sejam apresentadas
  como resultados do documento oficial da ANAC.
- Recusa avaliar corpus real quando o documento não está ingerido ou não possui páginas
  com aprovação humana explícita (`MANUALLY_APPROVED`), nunca aprovando páginas
  automaticamente.
- Mantém `max_vector_distance=0.85` inalterado na configuração: produz apenas uma
  recomendação de calibração transparente e reproduzível quando executado sobre corpus
  real aprovado com `gemini-embedding-2`.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rbac_assistant.application.services.validation_service import ValidationService
from rbac_assistant.domain.entities.regulatory_document import ValidationPolicy
from rbac_assistant.domain.entities.regulatory_page import PageValidationStatus
from rbac_assistant.domain.ports.rag_ports import EmbeddingPort
from rbac_assistant.domain.ports.vector_store_port import VectorStorePort
from rbac_assistant.infrastructure.storage.sqlite_document_repository import (
    SqliteDocumentRepository,
)
from rbac_assistant.rag.evaluation.rag_evaluator import (
    EvaluationCase,
    EvaluationSummaryReport,
    ExpectedTarget,
    RagEvaluator,
    ThresholdSweepPoint,
)
from rbac_assistant.rag.retrieval.hybrid_retriever import HybridNormativeRetriever

DEFAULT_OFFICIAL_DATASET_PATH = Path("config/evaluations/rbac43_emd05_eval_dataset.json")
PROVISIONAL_DEFAULT_MAX_VECTOR_DISTANCE = 0.85


@dataclass(frozen=True, slots=True)
class OfficialCorpusApprovalSummary:
    """Resumo auditável do estado de aprovação humana do corpus real no SQLite."""

    document_id: str
    version_id: str
    amendment_number: str
    source_hash: str
    total_pages: int
    manually_approved_pages: int
    preliminary_native_pages: int
    ocr_pending_pages: int
    rejected_pages: int
    sqlite_chunk_count: int


@dataclass(frozen=True, slots=True)
class ThresholdCalibrationRecommendation:
    """Análise reproduzível de calibração para `max_vector_distance` (sem alterar config)."""

    provisional_configured_threshold: float
    calibration_eligible: bool
    recommended_threshold: float | None
    rationale: str
    evaluated_candidates: tuple[ThresholdSweepPoint, ...]


@dataclass(frozen=True, slots=True)
class ApprovedCorpusEvaluationReport:
    """Relatório completo da avaliação executada sobre corpus real aprovado."""

    corpus_origin: str
    dataset_origin: str
    approval_summary: OfficialCorpusApprovalSummary
    evaluated_cases_count: int
    skipped_unapproved_case_ids: tuple[str, ...]
    summary: EvaluationSummaryReport
    calibration: ThresholdCalibrationRecommendation


def load_evaluation_cases_from_json(
    dataset_path: Path,
) -> tuple[str, str, tuple[EvaluationCase, ...]]:
    """Carrega dataset JSON de avaliação preservando nome e origem (`dataset_origin`)."""
    raw = json.loads(dataset_path.read_text(encoding="utf-8"))
    dataset_name = str(raw.get("dataset_name", dataset_path.stem))
    dataset_origin = str(raw.get("dataset_origin", "unspecified"))
    cases: list[EvaluationCase] = []
    for item in raw.get("cases", []):
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
                expected_pdf_pages=tuple(int(p) for p in t.get("expected_pdf_pages", [])),
                expected_text_snippets=tuple(str(s) for s in t.get("expected_text_snippets", [])),
            )
            for t in item.get("expected_targets", [])
        )
        cases.append(
            EvaluationCase(
                case_id=str(item["case_id"]),
                category=str(item["category"]),
                structural_subset=str(item["structural_subset"]),
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
    return dataset_name, dataset_origin, tuple(cases)


_HUMAN_APPROVED_STATUSES: frozenset[PageValidationStatus] = frozenset(
    {
        PageValidationStatus.MANUALLY_APPROVED,
        PageValidationStatus.APPROVED_FOR_LITERAL_TRANSCRIPTION,
    }
)


def inspect_official_corpus_approval(
    *,
    repository: SqliteDocumentRepository,
    document_id: str,
) -> OfficialCorpusApprovalSummary:
    """Inspeciona o estado real de aprovação humana no SQLite sem alterar nenhum registro."""
    doc = repository.get_document(document_id)
    if doc is None:
        raise ValueError(
            f"Documento '{document_id}' não está ingerido no catálogo SQLite. "
            "Execute `scripts/ingest_regulation.py` antes da avaliação do corpus oficial."
        )
    version = repository.get_current_version(document_id)
    if version is None:
        versions = repository.list_versions(document_id)
        if not versions:
            raise ValueError(f"Nenhuma versão encontrada para '{document_id}' no SQLite.")
        version = versions[0]

    pages = repository.get_pages(version.version_id)
    approved = sum(1 for p in pages if p.validation_status in _HUMAN_APPROVED_STATUSES)
    prelim = sum(
        1 for p in pages if p.validation_status == PageValidationStatus.NATIVE_QUALITY_CHECKED
    )
    ocr_pending = sum(
        1 for p in pages if p.validation_status == PageValidationStatus.PENDING_MANUAL_REVIEW
    )
    rejected = sum(1 for p in pages if p.validation_status == PageValidationStatus.REJECTED)
    chunks = repository.list_chunks(version.version_id)

    return OfficialCorpusApprovalSummary(
        document_id=document_id,
        version_id=version.version_id,
        amendment_number=version.amendment_number,
        source_hash=version.source_hash,
        total_pages=len(pages),
        manually_approved_pages=approved,
        preliminary_native_pages=prelim,
        ocr_pending_pages=ocr_pending,
        rejected_pages=rejected,
        sqlite_chunk_count=len(chunks),
    )


def recommend_threshold_calibration(
    *,
    sweep_points: Sequence[ThresholdSweepPoint],
    corpus_origin: str,
    embedding_backend: str,
    approved_pages_count: int,
    provisional_default: float = PROVISIONAL_DEFAULT_MAX_VECTOR_DISTANCE,
) -> ThresholdCalibrationRecommendation:
    """Gera recomendação reproduzível de limiar vetorial sem modificar `settings.yaml`."""
    eligible = (
        corpus_origin == "official_approved_corpus"
        and embedding_backend == "gemini-embedding-2"
        and approved_pages_count > 0
        and len(sweep_points) > 0
    )

    if not eligible:
        return ThresholdCalibrationRecommendation(
            provisional_configured_threshold=provisional_default,
            calibration_eligible=False,
            recommended_threshold=None,
            rationale=(
                "Calibração definitiva indisponível nesta execução "
                f"(corpus_origin={corpus_origin}, embedding_backend={embedding_backend}, "
                f"approved_pages={approved_pages_count}). "
                f"Mantenha max_vector_distance={provisional_default:.2f} (provisório) até "
                "executar com `gemini-embedding-2` real sobre páginas oficialmente aprovadas "
                "do RBAC 43."
            ),
            evaluated_candidates=tuple(sweep_points),
        )

    # Considera apenas candidatos com taxa de recusa segura = 100% (abstention_rate == 1.0)
    safe_points = [
        pt for pt in sweep_points if pt.semantic_top_k == 5 and pt.hybrid_abstention_rate >= 0.9999
    ]
    if not safe_points:
        return ThresholdCalibrationRecommendation(
            provisional_configured_threshold=provisional_default,
            calibration_eligible=True,
            recommended_threshold=None,
            rationale=(
                "Nenhum limiar avaliado preservou 100% de abstenção segura em consultas sem "
                f"evidência; mantenha max_vector_distance={provisional_default:.2f} e revise "
                "os casos negativos antes de recalibrar."
            ),
            evaluated_candidates=tuple(sweep_points),
        )

    best = max(
        safe_points,
        key=lambda pt: (
            pt.hybrid_recall_at_5,
            pt.hybrid_mrr_at_10,
            pt.hybrid_precision_at_5,
            -pt.hybrid_retrieval_fp,
            -abs(pt.max_vector_distance - provisional_default),
        ),
    )
    return ThresholdCalibrationRecommendation(
        provisional_configured_threshold=provisional_default,
        calibration_eligible=True,
        recommended_threshold=best.max_vector_distance,
        rationale=(
            f"Candidato ótimo no corpus real aprovado com `gemini-embedding-2`: "
            f"max_vector_distance={best.max_vector_distance:.2f} "
            f"(Hybrid Rec@5={best.hybrid_recall_at_5 * 100:.1f}%, "
            f"Prec@5={best.hybrid_precision_at_5 * 100:.1f}%, "
            f"MRR@10={best.hybrid_mrr_at_10:.3f}, Abst={best.hybrid_abstention_rate * 100:.1f}%, "
            f"RetFP={best.hybrid_retrieval_fp}). "
            "`config/settings.yaml` não foi modificado automaticamente."
        ),
        evaluated_candidates=tuple(sweep_points),
    )


def evaluate_approved_corpus(
    *,
    repository: SqliteDocumentRepository,
    vector_store: VectorStorePort,
    embedding_port: EmbeddingPort | None,
    document_id: str = "anac_rbac_43",
    collection_name: str = "regulatory_chunks_ge2_768_v1",
    dataset_path: Path = DEFAULT_OFFICIAL_DATASET_PATH,
    embedding_backend_label: str = "lexical_only",
    default_threshold: float = PROVISIONAL_DEFAULT_MAX_VECTOR_DISTANCE,
    sweep_thresholds: Sequence[float] = (0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95),
    include_ablations: bool = True,
) -> ApprovedCorpusEvaluationReport:
    """Executa avaliação controlada exclusivamente sobre páginas aprovadas no SQLite real."""
    approval = inspect_official_corpus_approval(repository=repository, document_id=document_id)
    if approval.manually_approved_pages == 0:
        raise ValueError(
            f"O documento '{document_id}' (versão {approval.version_id}) possui "
            f"0/{approval.total_pages} páginas aprovadas manualmente. Revise e aprove páginas "
            "explicitamente com `scripts/validate_extraction.py` antes de executar a avaliação "
            "do corpus oficial."
        )
    if approval.sqlite_chunk_count == 0:
        raise ValueError(
            f"O documento '{document_id}' (versão {approval.version_id}) ainda não possui chunks "
            "indexados no SQLite FTS5. Execute `scripts/index_document.py` antes da avaliação."
        )

    dataset_name, dataset_origin, all_cases = load_evaluation_cases_from_json(dataset_path)
    approved_pages = {
        p.pdf_page_number
        for p in repository.get_pages(approval.version_id)
        if p.validation_status in _HUMAN_APPROVED_STATUSES
    }

    # Seleciona casos negativos (sem alvos) e casos cujas seções/páginas alvo estão aprovadas
    eligible_cases: list[EvaluationCase] = []
    skipped_ids: list[str] = []
    for case in all_cases:
        if not case.expected_targets:
            eligible_cases.append(case)
            continue
        all_targets_approved = True
        for target in case.expected_targets:
            units = repository.get_units_by_section(
                version_id=approval.version_id,
                section_number=target.section_number,
            )
            if not units:
                all_targets_approved = False
                break
            unit_pages = {p for u in units for p in u.page_numbers}
            if not unit_pages or not unit_pages.issubset(approved_pages):
                all_targets_approved = False
                break
        if all_targets_approved:
            eligible_cases.append(case)
        else:
            skipped_ids.append(case.case_id)

    if not any(c.expected_targets for c in eligible_cases):
        raise ValueError(
            "Nenhum caso respondível do dataset oficial possui todas as páginas da seção "
            f"aprovadas manualmente em '{approval.version_id}' "
            f"(aprovadas: {sorted(approved_pages)})."
        )

    validation_service = ValidationService(ValidationPolicy.REVIEW_ON_DEMAND)

    def _retriever_factory(
        mode: str,
        max_vector_distance: float = default_threshold,
        *,
        lexical_top_k: int = 5,
        semantic_top_k: int = 5,
        rrf_k: int = 60,
        source_weights: Any = None,
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
            repository=repository,
            vector_store=vector_store,
            embedding_port=embedding_port if flags[2] else None,
            collection_name=collection_name,
            validation_service=validation_service,
            lexical_top_k=lexical_top_k,
            semantic_top_k=semantic_top_k,
            final_top_k=5,
            rrf_k=rrf_k,
            max_vector_distance=max_vector_distance,
            enable_structural=flags[0],
            enable_lexical=flags[1],
            enable_vector=flags[2] and (embedding_port is not None),
            source_weights=source_weights,
            enable_reranker=default_rerank if enable_reranker is None else enable_reranker,
            enable_adaptive_cutoff=(
                default_cutoff if enable_adaptive_cutoff is None else enable_adaptive_cutoff
            ),
            adaptive_cutoff_ratio=adaptive_cutoff_ratio,
        )

    evaluator = RagEvaluator()
    summary = evaluator.run_full_benchmark(
        dataset_name=dataset_name,
        embedding_backend=embedding_backend_label,
        cases=eligible_cases,
        retriever_factory=_retriever_factory,
        default_threshold=default_threshold,
        sweep_thresholds=sweep_thresholds,
        include_ablations=include_ablations,
    )
    calibration = recommend_threshold_calibration(
        sweep_points=summary.threshold_sweep,
        corpus_origin="official_approved_corpus",
        embedding_backend=embedding_backend_label,
        approved_pages_count=approval.manually_approved_pages,
        provisional_default=default_threshold,
    )
    return ApprovedCorpusEvaluationReport(
        corpus_origin="official_approved_corpus",
        dataset_origin=dataset_origin,
        approval_summary=approval,
        evaluated_cases_count=len(eligible_cases),
        skipped_unapproved_case_ids=tuple(skipped_ids),
        summary=summary,
        calibration=calibration,
    )
