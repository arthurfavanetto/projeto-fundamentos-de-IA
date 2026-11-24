"""Motor de busca híbrida (Estrutural + SQLite FTS5 + ChromaDB Vetorial + Fusão RRF).

Responsabilidades:
1. Resolver referências estruturais explícitas (ex.: `RBAC 43.3`, `43.9(a)(1)(i)`, `Art. 1º`).
2. Executar busca lexical SQLite FTS5 (BM25, tolerante a acentos, sem alterar `original_text`).
3. Executar busca semântica vetorial no ChromaDB com filtros de metadados normalizados
   (incluindo `amendment_number_key` para `5` == `05` == `005`) e limiar de distância.
4. Fundir e deduplicar candidatos via Reciprocal Rank Fusion (RRF) rastreável e reranking
   determinístico opcional (sem LLM pago).
5. Hidratar evidências diretamente do SQLite (SSOT), detectar ambiguidade de versão, selecionar
   contexto pertinente (poda adaptativa opcional preservando reconstrução integral) e validar
   ao vivo o estado de revisão de todas as páginas abrangidas.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import structlog

from rbac_assistant.application.services.validation_service import (
    DocumentVersionValidationAssessment,
    ValidationService,
)
from rbac_assistant.domain.entities.document_version import DocumentVersion
from rbac_assistant.domain.entities.normative_unit import (
    NormativeTable,
    StructureStatus,
    validate_table_integrity,
)
from rbac_assistant.domain.entities.rag import (
    EvidenceDecisionDiagnostic,
    EvidenceTrustLevel,
    FusedEvidence,
    HybridRetrievalResult,
    LexicalHitDiagnostic,
    NormativeChunk,
    ResolvedStructuralRef,
    RetrievalCandidate,
    RetrievalDiagnostics,
    RetrievalSource,
    RrfCandidateDiagnostic,
    SearchFilter,
    VectorHitDiagnostic,
)
from rbac_assistant.domain.entities.regulatory_document import (
    DocumentType,
    RegulatoryDocument,
    ValidationPolicy,
    VersionStatus,
)
from rbac_assistant.domain.entities.regulatory_page import PageValidationStatus, RegulatoryPage
from rbac_assistant.domain.exceptions import (
    EmbeddingCompatibilityError,
    LLMServiceError,
    StorageConnectionError,
)
from rbac_assistant.domain.ports.document_repository_port import DocumentRepositoryPort
from rbac_assistant.domain.ports.rag_ports import EmbeddingPort
from rbac_assistant.domain.ports.vector_store_port import (
    MetadataFilterValue,
    VectorStorePort,
)
from rbac_assistant.rag.retrieval.reranker import DeterministicNormativeReranker, RerankInput
from rbac_assistant.rag.retrieval.section_resolver import (
    amendments_match,
    extract_amendment_hint,
    extract_regulation_hint,
    normalize_amendment_number,
    resolve_structural_references,
)

logger = structlog.get_logger(__name__)

_DEFAULT_SOURCE_WEIGHTS: Mapping[RetrievalSource, float] = {
    RetrievalSource.EXACT_STRUCTURAL: 2.0,
    RetrievalSource.LEXICAL_FTS5: 1.0,
    RetrievalSource.SEMANTIC_VECTOR: 1.0,
}


@dataclass(slots=True)
class _RrfAccumulator:
    chunk_id: str
    score: float = 0.0
    sources: list[RetrievalSource] | None = None
    structural_rank: int | None = None
    lexical_rank: int | None = None
    vector_rank: int | None = None
    bm25_score: float | None = None
    vector_distance: float | None = None
    rerank_score: float | None = None

    def add(
        self,
        candidate: RetrievalCandidate,
        rrf_k: int,
        weights: Mapping[RetrievalSource, float] = _DEFAULT_SOURCE_WEIGHTS,
    ) -> None:
        if self.sources is None:
            self.sources = []
        weight = weights.get(candidate.source, _DEFAULT_SOURCE_WEIGHTS[candidate.source])
        self.score += weight / float(rrf_k + candidate.rank)
        if candidate.source not in self.sources:
            self.sources.append(candidate.source)
        if candidate.source == RetrievalSource.EXACT_STRUCTURAL:
            if self.structural_rank is None or candidate.rank < self.structural_rank:
                self.structural_rank = candidate.rank
        elif candidate.source == RetrievalSource.LEXICAL_FTS5:
            if self.lexical_rank is None or candidate.rank < self.lexical_rank:
                self.lexical_rank = candidate.rank
            if self.bm25_score is None or candidate.raw_score > self.bm25_score:
                self.bm25_score = candidate.raw_score
        elif candidate.source == RetrievalSource.SEMANTIC_VECTOR and (
            self.vector_rank is None or candidate.rank < self.vector_rank
        ):
            self.vector_rank = candidate.rank


def fuse_candidates_rrf(
    candidates: Sequence[RetrievalCandidate],
    *,
    rrf_k: int = 60,
    source_weights: Mapping[RetrievalSource, float] | None = None,
) -> list[_RrfAccumulator]:
    """Combina candidatos de múltiplas fontes usando Reciprocal Rank Fusion (RRF)."""
    weights = source_weights or _DEFAULT_SOURCE_WEIGHTS
    by_chunk: dict[str, _RrfAccumulator] = {}
    for cand in candidates:
        acc = by_chunk.get(cand.chunk_id)
        if acc is None:
            acc = _RrfAccumulator(chunk_id=cand.chunk_id)
            by_chunk[cand.chunk_id] = acc
        acc.add(cand, rrf_k, weights)

    return sorted(
        by_chunk.values(),
        key=lambda item: (-item.score, item.chunk_id),
    )


class HybridNormativeRetriever:
    """Recuperador híbrido com bloqueio global pré-recuperação, SSOT SQLite, ChromaDB e RRF."""

    def __init__(
        self,
        *,
        repository: DocumentRepositoryPort,
        vector_store: VectorStorePort,
        embedding_port: EmbeddingPort | None,
        collection_name: str,
        validation_service: ValidationService | None = None,
        lexical_top_k: int = 5,
        semantic_top_k: int = 5,
        final_top_k: int = 5,
        rrf_k: int = 60,
        max_vector_distance: float = 0.85,
        reconstruct_full_section_from_ssot: bool = True,
        require_vector_search: bool = False,
        enable_structural: bool = True,
        enable_lexical: bool = True,
        enable_vector: bool = True,
        source_weights: Mapping[RetrievalSource, float] | None = None,
        enable_reranker: bool = True,
        enable_adaptive_cutoff: bool = True,
        adaptive_cutoff_ratio: float = 0.55,
    ) -> None:
        self._repository = repository
        self._vector_store = vector_store
        self._embedding_port = embedding_port
        self._collection_name = collection_name
        self._validation_service = validation_service or ValidationService(
            policy=ValidationPolicy.STRICT_ALL_PAGES
        )
        self._lexical_top_k = lexical_top_k
        self._semantic_top_k = semantic_top_k
        self._final_top_k = final_top_k
        self._rrf_k = rrf_k
        self._max_vector_distance = max_vector_distance
        self._reconstruct_full_section = reconstruct_full_section_from_ssot
        self._require_vector_search = require_vector_search
        self._enable_structural = enable_structural
        self._enable_lexical = enable_lexical
        self._enable_vector = enable_vector
        self._source_weights = dict(source_weights or _DEFAULT_SOURCE_WEIGHTS)
        self._enable_reranker = enable_reranker
        self._enable_adaptive_cutoff = enable_adaptive_cutoff
        self._adaptive_cutoff_ratio = adaptive_cutoff_ratio
        self._reranker = DeterministicNormativeReranker()

    def retrieve(
        self,
        *,
        query: str,
        filters: SearchFilter | None = None,
        top_k: int | None = None,
    ) -> HybridRetrievalResult:
        """Executa busca híbrida estritamente limitada a versões globalmente aprovadas."""
        clean_query = query.strip()
        if not clean_query:
            return HybridRetrievalResult(
                query=query,
                warnings=("Consulta vazia.",),
                diagnostics=RetrievalDiagnostics(
                    query=query,
                    max_vector_distance=self._max_vector_distance,
                    reranker_enabled=self._enable_reranker,
                    adaptive_cutoff_enabled=self._enable_adaptive_cutoff,
                ),
            )

        resolved_refs = resolve_structural_references(clean_query)
        base_filters = self._merge_query_hints(filters, clean_query, resolved_refs)
        limit_final = top_k or self._final_top_k

        # BLOQUEIO GLOBAL PRÉ-RECUPERAÇÃO (AO VIVO NO SQLITE):
        # Avalia todas as versões candidatas antes de acionar busca estrutural, FTS5,
        # embeddings ou ChromaDB. Nunca confia em metadados antigos de índices!
        all_versions = self._repository.list_all_versions()
        total_chunks = self._repository.count_chunks()
        if not all_versions and total_chunks == 0:
            return HybridRetrievalResult(
                query=clean_query,
                index_empty=True,
                warnings=(
                    "Nenhum documento regulatório está indexado na base no momento. "
                    "Execute a ingestão e a indexação antes de consultar o assistente.",
                ),
                diagnostics=RetrievalDiagnostics(
                    query=clean_query,
                    effective_filters=self._serialize_filters(base_filters),
                    resolved_references=resolved_refs,
                    max_vector_distance=self._max_vector_distance,
                    reranker_enabled=self._enable_reranker,
                    adaptive_cutoff_enabled=self._enable_adaptive_cutoff,
                ),
            )

        all_pages_cache: dict[str, list[RegulatoryPage]] = {}
        doc_assessments: dict[str, DocumentVersionValidationAssessment] = {}
        (
            approved_version_ids,
            blocked_versions,
            matching_any_scope_approved_count,
        ) = self._resolve_globally_approved_versions(
            all_versions=all_versions,
            effective_filters=base_filters,
            all_pages_cache=all_pages_cache,
            doc_assessments=doc_assessments,
        )

        # Curto-circuito ANTES de qualquer motor de busca (estrutural, lexical ou vetorial):
        # Se nenhuma versão globalmente aprovada corresponder ao escopo, não executa buscas!
        if not approved_version_ids:
            warnings_empty: list[str] = []
            blocked_by_approval = False
            blocked_vids = tuple(v.version_id for _, v, _ in blocked_versions)
            if blocked_versions:
                blocked_by_approval = True
                warnings_empty.append(
                    "BLOQUEIO GLOBAL DE APROVAÇÃO DOCUMENTAL: A(s) versão(ões) documental(is) "
                    "correspondente(s) ainda não tiveram todas as páginas revisadas e aprovadas "
                    "para transcrição literal e foram excluídas antes da recuperação."
                )
            else:
                ambiguity_warnings = self._detect_missing_current_version_ambiguity(
                    effective_filters=base_filters,
                    resolved_refs=resolved_refs,
                    doc_assessments=doc_assessments,
                    all_pages_cache=all_pages_cache,
                )
                warnings_empty.extend(ambiguity_warnings)
                if (
                    not ambiguity_warnings
                    and total_chunks == 0
                    and matching_any_scope_approved_count == 0
                ):
                    return HybridRetrievalResult(
                        query=clean_query,
                        resolved_references=resolved_refs,
                        index_empty=True,
                        warnings=(
                            "Nenhum documento regulatório aprovado está indexado na base "
                            "no momento.",
                        ),
                        diagnostics=RetrievalDiagnostics(
                            query=clean_query,
                            effective_filters=self._serialize_filters(base_filters),
                            resolved_references=resolved_refs,
                            max_vector_distance=self._max_vector_distance,
                            reranker_enabled=self._enable_reranker,
                            adaptive_cutoff_enabled=self._enable_adaptive_cutoff,
                        ),
                    )

            return HybridRetrievalResult(
                query=clean_query,
                resolved_references=resolved_refs,
                evidences=(),
                blocked_evidences=(),
                blocked_by_document_approval=blocked_by_approval,
                blocked_version_ids=blocked_vids,
                warnings=tuple(dict.fromkeys(warnings_empty)),
                diagnostics=RetrievalDiagnostics(
                    query=clean_query,
                    effective_filters=self._serialize_filters(
                        replace(base_filters, allowed_version_ids=())
                    ),
                    resolved_references=resolved_refs,
                    max_vector_distance=self._max_vector_distance,
                    version_ambiguity_detected=any(
                        "AMBIGUIDADE DE VERSÃO" in w for w in warnings_empty
                    ),
                    reranker_enabled=self._enable_reranker,
                    adaptive_cutoff_enabled=self._enable_adaptive_cutoff,
                ),
            )

        effective_filters = replace(
            base_filters,
            only_approved_literal=True,
            include_preliminary_native=False,
            allowed_version_ids=tuple(approved_version_ids),
        )

        if total_chunks == 0:
            return HybridRetrievalResult(
                query=clean_query,
                resolved_references=resolved_refs,
                index_empty=True,
                warnings=(
                    "Nenhum documento regulatório está indexado na base no momento. "
                    "Execute a indexação dos documentos aprovados antes de consultar.",
                ),
                diagnostics=RetrievalDiagnostics(
                    query=clean_query,
                    effective_filters=self._serialize_filters(effective_filters),
                    resolved_references=resolved_refs,
                    max_vector_distance=self._max_vector_distance,
                    reranker_enabled=self._enable_reranker,
                    adaptive_cutoff_enabled=self._enable_adaptive_cutoff,
                ),
            )

        candidates: list[RetrievalCandidate] = []
        warnings: list[str] = []
        structural_hit_ids: list[str] = []
        lexical_diagnostics: list[LexicalHitDiagnostic] = []
        vector_diagnostics: list[VectorHitDiagnostic] = []
        vector_distance_by_chunk: dict[str, float] = {}
        chroma_where_used: dict[str, MetadataFilterValue] | None = None

        # 1. Busca estrutural determinística exata restrita a versões globalmente aprovadas
        structural_hits = 0
        if self._enable_structural:
            for ref in resolved_refs:
                if not ref.section_number:
                    continue
                matched_chunks = self._repository.find_chunks_by_structure(
                    section_number=ref.section_number,
                    regulation_number=ref.regulation_number,
                    paragraph_number=ref.paragraph_number,
                    item_number=ref.item_number,
                    subitem_number=ref.subitem_number,
                    filters=effective_filters,
                    limit=limit_final,
                )
                for rank, chunk in enumerate(matched_chunks, start=1):
                    structural_hits += 1
                    structural_hit_ids.append(chunk.chunk_id)
                    candidates.append(
                        RetrievalCandidate(
                            chunk_id=chunk.chunk_id,
                            source=RetrievalSource.EXACT_STRUCTURAL,
                            rank=rank,
                            raw_score=1.0 / rank,
                        )
                    )

        # 2. Busca lexical SQLite FTS5 (BM25) restrita a versões globalmente aprovadas
        lexical_pairs: list[tuple[str, float]] = []
        if self._enable_lexical:
            lexical_pairs = self._repository.search_chunks_fts(
                query=clean_query,
                filters=effective_filters,
                limit=max(self._lexical_top_k, limit_final),
            )
            for rank, (chunk_id, bm25_score) in enumerate(lexical_pairs, start=1):
                lexical_diagnostics.append(
                    LexicalHitDiagnostic(chunk_id=chunk_id, rank=rank, bm25_score=bm25_score)
                )
                candidates.append(
                    RetrievalCandidate(
                        chunk_id=chunk_id,
                        source=RetrievalSource.LEXICAL_FTS5,
                        rank=rank,
                        raw_score=bm25_score,
                    )
                )

        # 3. Busca semântica vetorial no ChromaDB restrita a versões globalmente aprovadas
        vector_hits_count = 0
        vector_available = True
        if self._enable_vector and self._embedding_port is not None:
            try:
                self._vector_store.ensure_collection_compatible(
                    collection_name=self._collection_name,
                    embedding_model=self._embedding_port.model_name,
                    embedding_dimension=self._embedding_port.dimensions,
                    instruction_strategy_version=self._embedding_port.instruction_strategy_version,
                )
                query_vector = self._embedding_port.embed_query(clean_query)
                chroma_where_used = self._build_chroma_where(effective_filters)
                vector_hits = self._vector_store.query_similar(
                    collection_name=self._collection_name,
                    query_embedding=query_vector,
                    top_k=max(self._semantic_top_k, limit_final),
                    where=chroma_where_used,
                )
                accepted_rank = 0
                for raw_rank, hit in enumerate(vector_hits, start=1):
                    cosine_sim = max(-1.0, min(1.0, 1.0 - hit.distance))
                    if hit.distance <= self._max_vector_distance:
                        accepted_rank += 1
                        vector_hits_count += 1
                        vector_distance_by_chunk[hit.record_id] = hit.distance
                        vector_diagnostics.append(
                            VectorHitDiagnostic(
                                chunk_id=hit.record_id,
                                rank=raw_rank,
                                distance=hit.distance,
                                cosine_similarity=cosine_sim,
                                accepted=True,
                            )
                        )
                        candidates.append(
                            RetrievalCandidate(
                                chunk_id=hit.record_id,
                                source=RetrievalSource.SEMANTIC_VECTOR,
                                rank=accepted_rank,
                                raw_score=1.0 / (1.0 + max(0.0, hit.distance)),
                            )
                        )
                    else:
                        vector_diagnostics.append(
                            VectorHitDiagnostic(
                                chunk_id=hit.record_id,
                                rank=raw_rank,
                                distance=hit.distance,
                                cosine_similarity=cosine_sim,
                                accepted=False,
                                rejection_reason=(
                                    f"distance {hit.distance:.4f} > "
                                    f"max_vector_distance {self._max_vector_distance:.4f}"
                                ),
                            )
                        )
            except EmbeddingCompatibilityError:
                raise
            except (StorageConnectionError, LLMServiceError) as exc:
                vector_available = False
                if self._require_vector_search:
                    raise
                logger.warning(
                    "hybrid_vector_search_degraded",
                    collection=self._collection_name,
                    error=str(exc),
                )
                warnings.append(
                    "Busca vetorial indisponível nesta consulta "
                    f"({type(exc).__name__}); recuperação executada apenas com busca "
                    "estrutural e lexical (SQLite FTS5)."
                )

        serialized_chroma_where = tuple(
            (str(k), str(v)) for k, v in (chroma_where_used or {}).items()
        )

        if not candidates:
            ambiguity_warnings = self._detect_missing_current_version_ambiguity(
                effective_filters=effective_filters,
                resolved_refs=resolved_refs,
                doc_assessments=doc_assessments,
                all_pages_cache=all_pages_cache,
            )
            warnings.extend(ambiguity_warnings)
            diagnostics = RetrievalDiagnostics(
                query=clean_query,
                effective_filters=self._serialize_filters(effective_filters),
                chroma_where_filter=serialized_chroma_where,
                resolved_references=resolved_refs,
                structural_hit_ids=tuple(structural_hit_ids),
                lexical_hits=tuple(lexical_diagnostics),
                vector_hits=tuple(vector_diagnostics),
                rrf_candidates=(),
                evidence_decisions=(),
                max_vector_distance=self._max_vector_distance,
                version_ambiguity_detected=bool(ambiguity_warnings),
                reranker_enabled=self._enable_reranker,
                adaptive_cutoff_enabled=self._enable_adaptive_cutoff,
            )
            return HybridRetrievalResult(
                query=clean_query,
                resolved_references=resolved_refs,
                lexical_hits_count=len(lexical_pairs),
                vector_hits_count=vector_hits_count,
                structural_hits_count=structural_hits,
                vector_search_available=vector_available,
                warnings=tuple(warnings),
                diagnostics=diagnostics,
            )

        # 4. Fusão e deduplicação por RRF
        fused = fuse_candidates_rrf(
            candidates,
            rrf_k=self._rrf_k,
            source_weights=self._source_weights,
        )
        for acc in fused:
            acc.vector_distance = vector_distance_by_chunk.get(acc.chunk_id)

        if structural_hits > 0:
            structural_only = [acc for acc in fused if acc.structural_rank is not None]
            if structural_only:
                fused = structural_only

        chunk_ids = [item.chunk_id for item in fused]
        approved_version_set = set(approved_version_ids)
        chunks_by_id = {
            c.chunk_id: c
            for c in self._repository.get_chunks_by_ids(chunk_ids)
            if c.version_id in approved_version_set
        }

        # 4.1. Reranking determinístico leve opcional (sem LLM externo; nunca sobrepõe filtros)
        if self._enable_reranker and len(fused) > 1:
            rerank_inputs = [
                RerankInput(
                    chunk_id=acc.chunk_id,
                    chunk=chunks_by_id[acc.chunk_id],
                    rrf_score=acc.score,
                    structural_rank=acc.structural_rank,
                    lexical_rank=acc.lexical_rank,
                    vector_rank=acc.vector_rank,
                    bm25_score=acc.bm25_score,
                    vector_distance=acc.vector_distance,
                )
                for acc in fused
                if acc.chunk_id in chunks_by_id
            ]
            rerank_scores = self._reranker.score_candidates(
                query=clean_query,
                resolved_refs=resolved_refs,
                candidates=rerank_inputs,
            )
            for acc in fused:
                acc.rerank_score = rerank_scores.get(acc.chunk_id)
            fused.sort(
                key=lambda item: (
                    -(item.rerank_score if item.rerank_score is not None else item.score),
                    -item.score,
                    item.chunk_id,
                )
            )

        rrf_diagnostics = tuple(
            RrfCandidateDiagnostic(
                chunk_id=acc.chunk_id,
                rrf_score=acc.score,
                structural_rank=acc.structural_rank,
                lexical_rank=acc.lexical_rank,
                vector_rank=acc.vector_rank,
                sources=tuple(acc.sources or ()),
                rerank_score=acc.rerank_score,
            )
            for acc in fused
        )

        usable_evidences: list[FusedEvidence] = []
        blocked_integrity_evidences: list[FusedEvidence] = []
        evidence_decisions: list[EvidenceDecisionDiagnostic] = []
        seen_sections: set[tuple[str, str]] = set()
        # Cache efêmero restrito a uma única chamada `retrieve()` (zero estado persistente).
        docs_cache: dict[str, RegulatoryDocument | None] = {}
        versions_cache: dict[str, DocumentVersion | None] = {}
        all_tables_cache: dict[str, list[NormativeTable]] = {}

        for acc in fused:
            hydrated_chunk = chunks_by_id.get(acc.chunk_id)
            if hydrated_chunk is None:
                continue
            section_key = (hydrated_chunk.version_id, hydrated_chunk.section_number.lower())
            if self._reconstruct_full_section and section_key in seen_sections:
                evidence_decisions.append(
                    EvidenceDecisionDiagnostic(
                        chunk_id=hydrated_chunk.chunk_id,
                        document_id=hydrated_chunk.document_id,
                        version_id=hydrated_chunk.version_id,
                        regulation_number=hydrated_chunk.regulation_number,
                        amendment_number=hydrated_chunk.amendment_number,
                        section_number=hydrated_chunk.section_number,
                        pdf_pages=hydrated_chunk.pdf_pages,
                        trust_level=hydrated_chunk.validation_status,
                        included=False,
                        reason="excluded:duplicate_section_reconstructed",
                    )
                )
                continue

            evidence = self._hydrate_evidence_from_ssot(
                chunk=hydrated_chunk,
                acc=acc,
                docs_cache=docs_cache,
                versions_cache=versions_cache,
                all_pages_cache=all_pages_cache,
                all_tables_cache=all_tables_cache,
                doc_assessments=doc_assessments,
            )
            if evidence is None:
                continue
            if not self._matches_post_hydration_filters(evidence, effective_filters):
                evidence_decisions.append(
                    EvidenceDecisionDiagnostic(
                        chunk_id=evidence.chunk_id,
                        document_id=evidence.document_id,
                        version_id=evidence.version_id,
                        regulation_number=evidence.regulation_number,
                        amendment_number=evidence.amendment_number,
                        section_number=evidence.section_number,
                        pdf_pages=evidence.pdf_pages,
                        trust_level=evidence.trust_level.value,
                        included=False,
                        reason="excluded:post_hydration_filter",
                    )
                )
                continue

            seen_sections.add(section_key)
            if (
                evidence.trust_level == EvidenceTrustLevel.VALIDATED_LITERAL
                and evidence.literal_transcription_allowed
            ):
                usable_evidences.append(evidence)
                evidence_decisions.append(
                    EvidenceDecisionDiagnostic(
                        chunk_id=evidence.chunk_id,
                        document_id=evidence.document_id,
                        version_id=evidence.version_id,
                        regulation_number=evidence.regulation_number,
                        amendment_number=evidence.amendment_number,
                        section_number=evidence.section_number,
                        pdf_pages=evidence.pdf_pages,
                        trust_level=evidence.trust_level.value,
                        included=True,
                        reason="included:validated_literal",
                    )
                )
            else:
                is_table_block = bool(
                    evidence.warning_message
                    and "INTEGRIDADE TABULAR BLOQUEADA" in evidence.warning_message
                )
                if is_table_block:
                    blocked_integrity_evidences.append(evidence)
                    if evidence.warning_message:
                        warnings.append(evidence.warning_message)
                evidence_decisions.append(
                    EvidenceDecisionDiagnostic(
                        chunk_id=evidence.chunk_id,
                        document_id=evidence.document_id,
                        version_id=evidence.version_id,
                        regulation_number=evidence.regulation_number,
                        amendment_number=evidence.amendment_number,
                        section_number=evidence.section_number,
                        pdf_pages=evidence.pdf_pages,
                        trust_level=evidence.trust_level.value,
                        included=False,
                        reason=(
                            "excluded:table_integrity_failed"
                            if is_table_block
                            else "excluded:global_document_approval_required"
                        ),
                    )
                )

            if len(usable_evidences) >= limit_final:
                break

        usable_evidences.sort(
            key=lambda ev: (
                -(ev.rerank_score if ev.rerank_score is not None else ev.rrf_score),
                -ev.rrf_score,
            )
        )

        # 4.2. Seleção adaptativa de contexto pertinente
        if self._enable_adaptive_cutoff and structural_hits == 0 and len(usable_evidences) > 1:
            top_ev = usable_evidences[0]
            top_metric = (
                top_ev.rerank_score if top_ev.rerank_score is not None else top_ev.rrf_score
            )
            min_metric = top_metric * self._adaptive_cutoff_ratio
            kept: list[FusedEvidence] = [top_ev]
            pruned_ids: set[str] = set()
            for ev in usable_evidences[1:]:
                cand_metric = ev.rerank_score if ev.rerank_score is not None else ev.rrf_score
                if cand_metric >= min_metric:
                    kept.append(ev)
                else:
                    pruned_ids.add(ev.chunk_id)

            if pruned_ids:
                usable_evidences = kept
                evidence_decisions = [
                    (
                        EvidenceDecisionDiagnostic(
                            chunk_id=d.chunk_id,
                            document_id=d.document_id,
                            version_id=d.version_id,
                            regulation_number=d.regulation_number,
                            amendment_number=d.amendment_number,
                            section_number=d.section_number,
                            pdf_pages=d.pdf_pages,
                            trust_level=d.trust_level,
                            included=False,
                            reason=(
                                f"excluded:adaptive_context_cutoff"
                                f"(ratio<{self._adaptive_cutoff_ratio:.2f})"
                            ),
                        )
                        if d.chunk_id in pruned_ids and d.included
                        else d
                    )
                    for d in evidence_decisions
                ]

        # 5. Detecção explícita de ambiguidade de versão (nunca assume maior emenda como vigente)
        ambiguity_warnings = self._detect_result_version_ambiguity(
            evidences=usable_evidences,
            effective_filters=effective_filters,
        )
        if not usable_evidences:
            ambiguity_warnings.extend(
                self._detect_missing_current_version_ambiguity(
                    effective_filters=effective_filters,
                    resolved_refs=resolved_refs,
                    doc_assessments=doc_assessments,
                    all_pages_cache=all_pages_cache,
                )
            )
        warnings.extend(ambiguity_warnings)

        diagnostics = RetrievalDiagnostics(
            query=clean_query,
            effective_filters=self._serialize_filters(effective_filters),
            chroma_where_filter=serialized_chroma_where,
            resolved_references=resolved_refs,
            structural_hit_ids=tuple(structural_hit_ids),
            lexical_hits=tuple(lexical_diagnostics),
            vector_hits=tuple(vector_diagnostics),
            rrf_candidates=rrf_diagnostics,
            evidence_decisions=tuple(evidence_decisions),
            max_vector_distance=self._max_vector_distance,
            version_ambiguity_detected=bool(ambiguity_warnings),
            reranker_enabled=self._enable_reranker,
            adaptive_cutoff_enabled=self._enable_adaptive_cutoff,
        )

        return HybridRetrievalResult(
            query=clean_query,
            resolved_references=resolved_refs,
            evidences=tuple(usable_evidences),
            blocked_evidences=tuple(blocked_integrity_evidences),
            lexical_hits_count=len(lexical_pairs),
            vector_hits_count=vector_hits_count,
            structural_hits_count=structural_hits,
            vector_search_available=vector_available,
            warnings=tuple(dict.fromkeys(warnings)),
            diagnostics=diagnostics,
        )

    def _resolve_globally_approved_versions(
        self,
        *,
        all_versions: Sequence[DocumentVersion],
        effective_filters: SearchFilter,
        all_pages_cache: dict[str, list[RegulatoryPage]],
        doc_assessments: dict[str, DocumentVersionValidationAssessment],
    ) -> tuple[
        list[str],
        list[tuple[RegulatoryDocument, DocumentVersion, DocumentVersionValidationAssessment]],
        int,
    ]:
        """Resolve ao vivo no SQLite as versões elegíveis (100% das páginas aprovadas)."""
        catalog_docs = {doc.document_id: doc for doc in self._repository.search_documents()}
        approved_ids: list[str] = []
        blocked: list[
            tuple[RegulatoryDocument, DocumentVersion, DocumentVersionValidationAssessment]
        ] = []
        total_approved_any_scope = 0

        for version in all_versions:
            doc = catalog_docs.get(version.document_id)
            if doc is None:
                continue
            if not self._matches_catalog_scope(doc, version, effective_filters):
                continue

            pages = self._repository.get_pages(version.version_id)
            all_pages_cache[version.version_id] = pages
            assessment = self._validation_service.evaluate_document_version(
                version=version,
                pages=pages,
            )
            doc_assessments[version.version_id] = assessment
            if assessment.literal_transcription_allowed:
                approved_ids.append(version.version_id)
                total_approved_any_scope += 1
            else:
                blocked.append((doc, version, assessment))

        return approved_ids, blocked, total_approved_any_scope

    @staticmethod
    def _matches_catalog_scope(
        document: RegulatoryDocument,
        version: DocumentVersion,
        filters: SearchFilter,
    ) -> bool:
        if filters.document_id and document.document_id != filters.document_id:
            return False
        if filters.authority and document.authority.lower() != filters.authority.lower():
            return False
        if filters.document_type is not None and document.document_type != filters.document_type:
            return False
        if (
            filters.regulation_number
            and document.regulation_number.strip().lower()
            != filters.regulation_number.strip().lower()
        ):
            return False
        if filters.version_id and version.version_id != filters.version_id:
            return False
        if filters.allowed_version_ids is not None and (
            version.version_id not in filters.allowed_version_ids
        ):
            return False
        if filters.amendment_number and not amendments_match(
            version.amendment_number, filters.amendment_number
        ):
            return False
        if (
            not filters.amendment_number
            and not filters.version_id
            and filters.version_status is not None
            and version.version_status != filters.version_status
        ):
            return False
        return not (
            not filters.amendment_number
            and not filters.version_id
            and filters.version_status is None
            and filters.only_current_version
            and version.version_status != VersionStatus.CURRENT
        )

    @staticmethod
    def _merge_query_hints(
        filters: SearchFilter | None,
        query: str,
        resolved_refs: Sequence[ResolvedStructuralRef],
    ) -> SearchFilter:
        base = filters or SearchFilter()

        amendment = base.amendment_number
        only_current = base.only_current_version
        if not amendment:
            for ref in resolved_refs:
                if ref.amendment_number:
                    amendment = ref.amendment_number
                    only_current = False
                    break
            if not amendment:
                hinted_amd = extract_amendment_hint(query)
                if hinted_amd:
                    amendment = hinted_amd
                    only_current = False

        regulation_number = base.regulation_number
        document_type = base.document_type
        if not regulation_number and not base.document_id:
            hinted_type_str, hinted_reg = extract_regulation_hint(query)
            if hinted_reg:
                regulation_number = hinted_reg
                if document_type is None and hinted_type_str is not None:
                    try:
                        document_type = DocumentType(hinted_type_str)
                    except ValueError:
                        document_type = None
            else:
                ref_regs = {
                    ref.regulation_number.strip()
                    for ref in resolved_refs
                    if ref.regulation_number and ref.regulation_number.strip()
                }
                if len(ref_regs) == 1:
                    regulation_number = next(iter(ref_regs))

        # Política global obrigatória: sempre exige aprovação integral e bloqueia preliminares
        return SearchFilter(
            document_id=base.document_id,
            authority=base.authority,
            document_type=document_type,
            regulation_number=regulation_number,
            amendment_number=amendment,
            version_id=base.version_id,
            version_status=base.version_status,
            only_current_version=only_current,
            only_approved_literal=True,
            include_preliminary_native=False,
            allowed_version_ids=base.allowed_version_ids,
        )

    @staticmethod
    def _build_chroma_where(
        filters: SearchFilter,
    ) -> dict[str, MetadataFilterValue] | None:
        where: dict[str, MetadataFilterValue] = {}
        if filters.document_id:
            where["document_id"] = filters.document_id
        if filters.authority:
            where["authority"] = filters.authority
        if filters.document_type is not None:
            where["document_type"] = filters.document_type.value
        if filters.regulation_number:
            where["regulation_number"] = filters.regulation_number.strip()
        if filters.version_id:
            where["version_id"] = filters.version_id
        elif filters.allowed_version_ids is not None:
            if len(filters.allowed_version_ids) == 1:
                where["version_id"] = filters.allowed_version_ids[0]
            elif len(filters.allowed_version_ids) > 1:
                where["version_id"] = list(filters.allowed_version_ids)
        if filters.amendment_number:
            # Aplica filtragem normalizada diretamente no ChromaDB ("5" == "05" == "005")
            # evitando desperdício de janela top_k com emendas concorrentes.
            where["amendment_number_key"] = normalize_amendment_number(filters.amendment_number)
        elif filters.version_status is not None:
            where["version_status"] = filters.version_status.value
        elif filters.only_current_version and not filters.version_id:
            where["version_status"] = VersionStatus.CURRENT.value
        return where or None

    @staticmethod
    def _serialize_filters(filters: SearchFilter) -> tuple[tuple[str, str], ...]:
        items: list[tuple[str, str]] = []
        if filters.document_id:
            items.append(("document_id", filters.document_id))
        if filters.authority:
            items.append(("authority", filters.authority))
        if filters.document_type is not None:
            items.append(("document_type", filters.document_type.value))
        if filters.regulation_number:
            items.append(("regulation_number", filters.regulation_number))
        if filters.amendment_number:
            items.append(("amendment_number", filters.amendment_number))
            items.append(
                ("amendment_number_key", normalize_amendment_number(filters.amendment_number))
            )
        if filters.version_id:
            items.append(("version_id", filters.version_id))
        if filters.allowed_version_ids is not None:
            items.append(("allowed_version_ids", ",".join(filters.allowed_version_ids)))
        if filters.version_status is not None:
            items.append(("version_status", filters.version_status.value))
        items.append(("only_current_version", str(filters.only_current_version)))
        items.append(("only_approved_literal", str(filters.only_approved_literal)))
        items.append(("include_preliminary_native", str(filters.include_preliminary_native)))
        return tuple(items)

    def _detect_result_version_ambiguity(
        self,
        *,
        evidences: Sequence[FusedEvidence],
        effective_filters: SearchFilter,
    ) -> list[str]:
        """Detecta quando múltiplas versões do mesmo documento coexistem sem filtro explícito."""
        if effective_filters.version_id or effective_filters.amendment_number:
            return []

        versions_by_doc: dict[str, set[str]] = defaultdict(set)
        doc_labels: dict[str, str] = {}
        for ev in evidences:
            versions_by_doc[ev.document_id].add(
                f"Emenda {ev.amendment_number} ({ev.version_status})"
            )
            doc_labels[ev.document_id] = f"{ev.document_type} {ev.regulation_number}"

        ambiguity_warnings: list[str] = []
        for doc_id, labels in versions_by_doc.items():
            if len(labels) > 1:
                sorted_labels = ", ".join(sorted(labels))
                ambiguity_warnings.append(
                    f"AMBIGUIDADE DE VERSÃO: Múltiplas versões de {doc_labels[doc_id]} "
                    f"corresponderam à consulta ({sorted_labels}). Especifique a emenda "
                    "explicitamente (ex.: 'Emenda 05' ou --amendment) — o sistema não assume "
                    "automaticamente que a maior emenda é a vigente."
                )
                continue

            db_versions = self._repository.list_versions(doc_id)
            current_versions = [
                v for v in db_versions if v.version_status == VersionStatus.CURRENT
            ]
            if len(current_versions) > 1:
                amd_list = ", ".join(f"Emenda {v.amendment_number}" for v in current_versions)
                ambiguity_warnings.append(
                    f"AMBIGUIDADE DE VERSÃO: O catálogo SQLite possui múltiplas versões marcadas "
                    f"como vigentes para {doc_labels[doc_id]} ({amd_list}). Corrija a vigência "
                    "no catálogo ou especifique a emenda explicitamente."
                )

        return ambiguity_warnings

    def _detect_missing_current_version_ambiguity(
        self,
        *,
        effective_filters: SearchFilter,
        resolved_refs: Sequence[ResolvedStructuralRef],
        doc_assessments: dict[str, DocumentVersionValidationAssessment] | None = None,
        all_pages_cache: dict[str, list[RegulatoryPage]] | None = None,
    ) -> list[str]:
        """Alerta quando existem versões aprovadas não vigentes, mas nenhuma marcada current."""
        if (
            not effective_filters.only_current_version
            or effective_filters.amendment_number
            or effective_filters.version_id
        ):
            return []

        target_regs: set[str] = set()
        if effective_filters.regulation_number:
            target_regs.add(effective_filters.regulation_number.strip().lower())
        for ref in resolved_refs:
            if ref.regulation_number:
                target_regs.add(ref.regulation_number.strip().lower())

        docs = self._repository.search_documents()
        warnings: list[str] = []
        for doc in docs:
            if effective_filters.document_id and doc.document_id != effective_filters.document_id:
                continue
            if target_regs and doc.regulation_number.strip().lower() not in target_regs:
                continue
            versions = self._repository.list_versions(doc.document_id)
            approved_versions: list[DocumentVersion] = []
            for ver in versions:
                assessment = (
                    doc_assessments.get(ver.version_id) if doc_assessments is not None else None
                )
                if assessment is None:
                    pages = (
                        all_pages_cache.get(ver.version_id)
                        if all_pages_cache is not None
                        else None
                    )
                    if pages is None:
                        pages = self._repository.get_pages(ver.version_id)
                        if all_pages_cache is not None:
                            all_pages_cache[ver.version_id] = pages
                    assessment = self._validation_service.evaluate_document_version(
                        version=ver,
                        pages=pages,
                    )
                    if doc_assessments is not None:
                        doc_assessments[ver.version_id] = assessment
                if assessment.literal_transcription_allowed:
                    approved_versions.append(ver)

            if not approved_versions:
                continue
            has_current = any(
                v.version_status == VersionStatus.CURRENT for v in approved_versions
            )
            if not has_current:
                available = ", ".join(
                    f"Emenda {v.amendment_number} ({v.version_status.value})"
                    for v in approved_versions
                )
                warnings.append(
                    f"AMBIGUIDADE DE VERSÃO: {doc.document_type.value} {doc.regulation_number} "
                    f"não possui versão marcada como vigente (current) no catálogo; versões "
                    f"disponíveis: {available}. Informe a emenda explicitamente — o sistema não "
                    "assume automaticamente que a maior emenda é a vigente."
                )
        return warnings

    def _hydrate_evidence_from_ssot(
        self,
        *,
        chunk: NormativeChunk,
        acc: _RrfAccumulator,
        docs_cache: dict[str, RegulatoryDocument | None] | None = None,
        versions_cache: dict[str, DocumentVersion | None] | None = None,
        all_pages_cache: dict[str, list[RegulatoryPage]] | None = None,
        all_tables_cache: dict[str, list[NormativeTable]] | None = None,
        doc_assessments: dict[str, DocumentVersionValidationAssessment] | None = None,
    ) -> FusedEvidence | None:
        """Recupera texto literal e valida aprovação global ao vivo diretamente do SQLite (SSOT)."""
        if docs_cache is not None and chunk.document_id in docs_cache:
            document = docs_cache[chunk.document_id]
        else:
            document = self._repository.get_document(chunk.document_id)
            if docs_cache is not None:
                docs_cache[chunk.document_id] = document

        if versions_cache is not None and chunk.version_id in versions_cache:
            version = versions_cache[chunk.version_id]
        else:
            version = self._repository.get_version(chunk.version_id)
            if versions_cache is not None:
                versions_cache[chunk.version_id] = version

        if document is None or version is None:
            return None

        if all_pages_cache is not None and version.version_id in all_pages_cache:
            all_version_pages = all_pages_cache[version.version_id]
        else:
            all_version_pages = self._repository.get_pages(version.version_id)
            if all_pages_cache is not None:
                all_pages_cache[version.version_id] = all_version_pages

        doc_assessment = (
            doc_assessments.get(version.version_id) if doc_assessments is not None else None
        )
        if doc_assessment is None:
            doc_assessment = self._validation_service.evaluate_document_version(
                version=version,
                pages=all_version_pages,
            )
            if doc_assessments is not None:
                doc_assessments[version.version_id] = doc_assessment

        # Defesa em profundidade: nunca hidrata evidência de versão sem aprovação global
        if not doc_assessment.literal_transcription_allowed:
            return None

        units = self._repository.get_units_by_section(
            version_id=version.version_id,
            section_number=chunk.section_number,
        )
        root_units = [u for u in units if u.parent_unit_id is None]
        primary_unit = root_units[0] if root_units else (units[0] if units else None)

        if self._reconstruct_full_section and primary_unit is not None:
            original_text = primary_unit.original_text
            page_numbers = sorted({p for u in units for p in u.page_numbers})
            structure_status = (
                StructureStatus.PARSED
                if all(
                    u.structure_status == StructureStatus.PARSED
                    for u in (root_units or [primary_unit])
                )
                else StructureStatus.PENDING_VALIDATION
            )
            title = primary_unit.title or chunk.title
        else:
            original_text = chunk.original_text
            page_numbers = list(chunk.pdf_pages)
            structure_status = StructureStatus.PARSED
            title = chunk.title

        pages_by_num = {p.pdf_page_number: p for p in all_version_pages}
        pages = [pages_by_num[p_num] for p_num in page_numbers if p_num in pages_by_num]

        assessment = self._validation_service.evaluate_section(
            section_number=chunk.section_number,
            pages=pages,
            structure_status=structure_status,
            document_assessment=doc_assessment,
        )

        # Hidrata tabelas/formulários vinculados à seção e valida integridade (fail-closed)
        if all_tables_cache is not None and version.version_id in all_tables_cache:
            version_tables = all_tables_cache[version.version_id]
        else:
            version_tables = self._repository.list_tables(version.version_id)
            if all_tables_cache is not None:
                all_tables_cache[version.version_id] = version_tables
        page_num_set = set(page_numbers)
        raw_page_tables = [t for t in version_tables if t.pdf_page_number in page_num_set]
        section_tables: list[NormativeTable] = []
        for tbl in raw_page_tables:
            if primary_unit is not None and tbl.unit_id is not None:
                if tbl.unit_id == primary_unit.unit_id:
                    section_tables.append(tbl)
            else:
                section_tables.append(tbl)

        table_integrity_issues: list[str] = []
        for tbl in section_tables:
            table_integrity_issues.extend(validate_table_integrity(tbl))

        rejected_pages = tuple(
            p.pdf_page_number
            for p in pages
            if p.validation_status == PageValidationStatus.REJECTED
        )
        page_statuses = tuple((p.pdf_page_number, p.validation_status) for p in pages)
        printed_pages = tuple(
            p.printed_page_number or f"PDF {p.pdf_page_number}" for p in pages
        ) or chunk.printed_pages

        effective_warning = assessment.warning_message
        effective_allowed = assessment.literal_transcription_allowed
        if table_integrity_issues:
            trust_level = EvidenceTrustLevel.BLOCKED_UNVERIFIED
            effective_allowed = False
            effective_warning = (
                f"INTEGRIDADE TABULAR BLOQUEADA ({chunk.section_number}): "
                "A estrutura tabular desta seção apresentou inconsistência e foi bloqueada "
                "para evitar apresentação de valores sem cabeçalhos ou associação confiável "
                f"({'; '.join(table_integrity_issues[:3])})."
            )
            hydrated_tables: tuple[NormativeTable, ...] = ()
        elif assessment.literal_transcription_allowed:
            trust_level = EvidenceTrustLevel.VALIDATED_LITERAL
            hydrated_tables = tuple(section_tables)
        elif rejected_pages:
            trust_level = EvidenceTrustLevel.BLOCKED_REJECTED
            hydrated_tables = ()
        elif assessment.ocr_blocked_pdf_pages:
            trust_level = EvidenceTrustLevel.BLOCKED_OCR_PENDING
            hydrated_tables = ()
        else:
            trust_level = EvidenceTrustLevel.BLOCKED_UNVERIFIED
            hydrated_tables = ()

        return FusedEvidence(
            chunk_id=chunk.chunk_id,
            unit_id=chunk.unit_id,
            document_id=document.document_id,
            version_id=version.version_id,
            authority=document.authority,
            document_type=document.document_type.value,
            regulation_number=document.regulation_number,
            document_title=document.document_title,
            amendment_number=version.amendment_number,
            version_status=version.version_status.value,
            official_url=version.official_url or document.official_url,
            section_number=chunk.section_number,
            subsection_number=chunk.subsection_number,
            paragraph_number=chunk.paragraph_number,
            item_number=chunk.item_number,
            title=title,
            original_text=original_text,
            pdf_pages=tuple(page_numbers) or chunk.pdf_pages,
            printed_pages=printed_pages,
            page_statuses=page_statuses,
            trust_level=trust_level,
            literal_transcription_allowed=effective_allowed,
            rrf_score=acc.score,
            matched_sources=tuple(acc.sources or ()),
            pending_pdf_pages=assessment.pending_pdf_pages,
            ocr_blocked_pdf_pages=assessment.ocr_blocked_pdf_pages,
            rejected_pdf_pages=rejected_pages,
            lexical_rank=acc.lexical_rank,
            vector_rank=acc.vector_rank,
            structural_rank=acc.structural_rank,
            rerank_score=acc.rerank_score,
            warning_message=effective_warning,
            tables=hydrated_tables,
        )

    @staticmethod
    def _matches_post_hydration_filters(
        evidence: FusedEvidence,
        filters: SearchFilter,
    ) -> bool:
        if filters.document_id and evidence.document_id != filters.document_id:
            return False
        if filters.authority and evidence.authority.lower() != filters.authority.lower():
            return False
        if filters.document_type and evidence.document_type != filters.document_type.value:
            return False
        if (
            filters.regulation_number
            and evidence.regulation_number.lower() != filters.regulation_number.strip().lower()
        ):
            return False
        if filters.version_id and evidence.version_id != filters.version_id:
            return False
        if filters.allowed_version_ids is not None and (
            evidence.version_id not in filters.allowed_version_ids
        ):
            return False
        if filters.amendment_number and not amendments_match(
            evidence.amendment_number, filters.amendment_number
        ):
            return False
        if (
            not filters.amendment_number
            and not filters.version_id
            and filters.version_status is not None
            and evidence.version_status != filters.version_status.value
        ):
            return False
        return not (
            not filters.amendment_number
            and not filters.version_id
            and filters.version_status is None
            and filters.only_current_version
            and evidence.version_status != VersionStatus.CURRENT.value
        )
