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

import re
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
    TableOwnershipStatus,
    resolve_section_tables,
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
from rbac_assistant.domain.ports.validation_port import ParsedUnitRecord
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
from rbac_assistant.rag.retrieval.subunit_selector import (
    SubunitSelectionStatus,
    select_structural_units,
)

logger = structlog.get_logger(__name__)

_EXPLICIT_REF_PREFIX_RE = re.compile(
    r"\b(?:rbac|se[çc][ãa]o|sec\.?|ap[êe]ndice|art\.?|artigo|par[áa]grafo|inciso|item|al[íi]nea)\b",
    re.IGNORECASE,
)
_APPENDIX_ALIAS_SEC_RE = re.compile(
    r"^(?:ap[êe]ndice\s+[a-z0-9]+|[a-z]\d{1,3}\.\d+)$",
    re.IGNORECASE,
)

_DEFAULT_SOURCE_WEIGHTS: Mapping[RetrievalSource, float] = {
    RetrievalSource.EXACT_STRUCTURAL: 2.0,
    RetrievalSource.LEXICAL_FTS5: 1.0,
    RetrievalSource.SEMANTIC_VECTOR: 1.0,
}


@dataclass(slots=True)
class _RrfAccumulator:
    chunk_id: str
    target_ref: ResolvedStructuralRef | None = None
    target_key: str = ""
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
            if candidate.target_ref is not None and self.target_ref is None:
                self.target_ref = candidate.target_ref
                self.target_key = candidate.target_ref.target_key
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
    """Combina candidatos de múltiplas fontes via RRF preservando referências granulares.

    Quando um mesmo `chunk_id` (âncora de seção) é atingido por múltiplas referências
    estruturais distintas (ex.: `43.10(a)(1)` e `43.10(a)(2)`), cada referência mantém
    seu próprio acumulador. Hits lexicais/vetoriais no mesmo `chunk_id` reforçam o score
    dos alvos estruturais específicos sem criar um candidato paralelo da seção inteira.
    """
    weights = source_weights or _DEFAULT_SOURCE_WEIGHTS
    by_key: dict[tuple[str, str], _RrfAccumulator] = {}
    structural_keys_by_chunk: dict[str, list[tuple[str, str]]] = defaultdict(list)

    # 1º passo: registra todos os candidatos estruturais com sua chave granular específica
    for cand in candidates:
        if cand.target_ref is None:
            continue
        t_key = cand.target_ref.target_key
        full_key = (cand.chunk_id, t_key)
        acc = by_key.get(full_key)
        if acc is None:
            acc = _RrfAccumulator(
                chunk_id=cand.chunk_id,
                target_ref=cand.target_ref,
                target_key=t_key,
            )
            by_key[full_key] = acc
            structural_keys_by_chunk[cand.chunk_id].append(full_key)
        acc.add(cand, rrf_k, weights)

    # 2º passo: funde candidatos lexicais/vetoriais (ou estruturais sem target_ref)
    for cand in candidates:
        if cand.target_ref is not None:
            continue
        existing_structural = structural_keys_by_chunk.get(cand.chunk_id)
        if existing_structural:
            for full_key in existing_structural:
                by_key[full_key].add(cand, rrf_k, weights)
            continue
        full_key = (cand.chunk_id, "")
        acc = by_key.get(full_key)
        if acc is None:
            acc = _RrfAccumulator(chunk_id=cand.chunk_id)
            by_key[full_key] = acc
        acc.add(cand, rrf_k, weights)

    return sorted(
        by_key.values(),
        key=lambda item: (-item.score, item.chunk_id, item.target_key),
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
        units_cache: dict[tuple[str, str], list[ParsedUnitRecord]] = {}
        approved_version_regs = self._map_approved_version_regulations(approved_version_ids)

        # Avaliação estrutural determinística no SSOT de TODAS as referências explícitas
        # (independente do modo estrutural/lexical/vetorial): garante que qualquer referência
        # ausente, ambígua ou não verificável bloqueie a consulta inteira (zero resposta parcial).
        resolved_explicit_count, unresolved_explicit_refs = (
            self._evaluate_explicit_references_against_ssot(
                resolved_refs=resolved_refs,
                approved_version_ids=approved_version_ids,
                approved_version_regs=approved_version_regs,
                units_cache=units_cache,
            )
        )
        for u_ref, u_status, u_reason in unresolved_explicit_refs:
            if u_status == SubunitSelectionStatus.NOT_FOUND:
                warnings.append(
                    f"UNIDADE NORMATIVA NÃO ENCONTRADA ({u_ref.canonical_label}): {u_reason}"
                )
            elif not u_ref.has_subunit_target and "Unidade-raiz" in u_reason:
                warnings.append(
                    f"UNIDADE-RAIZ AUSENTE NO SSOT ({u_ref.canonical_label}): {u_reason}"
                )
            else:
                warnings.append(
                    f"GRANULARIDADE NORMATIVA BLOQUEADA ({u_ref.canonical_label}): {u_reason}"
                )
        if unresolved_explicit_refs and resolved_explicit_count > 0:
            missing_desc = ", ".join(r.canonical_label for r, _, _ in unresolved_explicit_refs)
            warnings.append(
                "CONSULTA PARCIAL BLOQUEADA (FAIL-CLOSED): A consulta solicitou múltiplas "
                f"referências estruturais explícitas, mas [{missing_desc}] não pôde(puderam) "
                "ser validada(s) no SSOT. Nenhuma transcrição parcial será apresentada."
            )

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
                    leaf_number=ref.leaf_number,
                    filters=effective_filters,
                    limit=limit_final,
                )

                # Defesa em profundidade: mesmo que o repositório retorne um chunk-raiz como
                # âncora, valida explicitamente no SSOT (`normative_units`) antes de aceitar.
                verified_anchor_chunks: list[NormativeChunk] = []
                seen_ver_sec_for_ref: set[tuple[str, str]] = set()
                for chunk in matched_chunks:
                    ver_sec_key = (chunk.version_id, chunk.section_number.lower())
                    if ver_sec_key in seen_ver_sec_for_ref and ref.has_subunit_target:
                        continue
                    sec_units = self._get_section_units_cached(
                        version_id=chunk.version_id,
                        section_number=chunk.section_number,
                        units_cache=units_cache,
                    )
                    sel = select_structural_units(sec_units, ref)
                    if not sel.found:
                        continue
                    if ref.has_subunit_target:
                        seen_ver_sec_for_ref.add(ver_sec_key)
                    verified_anchor_chunks.append(chunk)

                for rank, chunk in enumerate(verified_anchor_chunks, start=1):
                    structural_hits += 1
                    structural_hit_ids.append(chunk.chunk_id)
                    candidates.append(
                        RetrievalCandidate(
                            chunk_id=chunk.chunk_id,
                            source=RetrievalSource.EXACT_STRUCTURAL,
                            rank=rank,
                            raw_score=1.0 / rank,
                            target_ref=ref,
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
            blocked_subunit_empty = self._build_blocked_subunit_evidences(
                unresolved_subunit_refs=unresolved_explicit_refs,
                chunks_by_id={},
            )
            empty_decisions = tuple(
                EvidenceDecisionDiagnostic(
                    chunk_id=f"unresolved::{u_ref.canonical_label}",
                    document_id="",
                    version_id="",
                    regulation_number=u_ref.regulation_number or "",
                    amendment_number=u_ref.amendment_number or "",
                    section_number=u_ref.section_number or "",
                    pdf_pages=(),
                    trust_level=EvidenceTrustLevel.BLOCKED_UNVERIFIED.value,
                    included=False,
                    reason=f"excluded:explicit_ref_{u_status.value}({u_ref.canonical_label})",
                    paragraph_number=u_ref.paragraph_number or "",
                    item_number=f"{u_ref.item_number or ''}{u_ref.subitem_number or ''}",
                )
                for u_ref, u_status, _ in unresolved_explicit_refs
            )
            diagnostics = RetrievalDiagnostics(
                query=clean_query,
                effective_filters=self._serialize_filters(effective_filters),
                chroma_where_filter=serialized_chroma_where,
                resolved_references=resolved_refs,
                structural_hit_ids=tuple(structural_hit_ids),
                lexical_hits=tuple(lexical_diagnostics),
                vector_hits=tuple(vector_diagnostics),
                rrf_candidates=(),
                evidence_decisions=empty_decisions,
                max_vector_distance=self._max_vector_distance,
                version_ambiguity_detected=bool(ambiguity_warnings),
                reranker_enabled=self._enable_reranker,
                adaptive_cutoff_enabled=self._enable_adaptive_cutoff,
            )
            return HybridRetrievalResult(
                query=clean_query,
                resolved_references=resolved_refs,
                blocked_evidences=tuple(blocked_subunit_empty),
                lexical_hits_count=len(lexical_pairs),
                vector_hits_count=vector_hits_count,
                structural_hits_count=structural_hits,
                vector_search_available=vector_available,
                warnings=tuple(dict.fromkeys(warnings)),
                diagnostics=diagnostics,
            )

        # 4. Fusão e deduplicação por RRF preservando granularidade estrutural
        fused = fuse_candidates_rrf(
            candidates,
            rrf_k=self._rrf_k,
            source_weights=self._source_weights,
        )
        for acc in fused:
            acc.vector_distance = vector_distance_by_chunk.get(acc.chunk_id)

        explicit_subunit_refs = tuple(r for r in resolved_refs if r.has_subunit_target)

        # Proteção fail-closed integral (single-ref e multi-ref, em TODOS os modos — estrutural,
        # híbrido, lexical-only e vetorial-only): se QUALQUER referência estrutural explícita da
        # consulta não puder ser resolvida/verificada no SSOT, bloqueia TODA a solicitação sem
        # devolver transcrição parcial nem permitir fallback lexical/vetorial para a seção inteira.
        if unresolved_explicit_refs:
            approved_version_set = set(approved_version_ids)
            chunks_by_id_blocked = {
                c.chunk_id: c
                for c in self._repository.get_chunks_by_ids(
                    list(dict.fromkeys(item.chunk_id for item in fused))
                )
                if c.version_id in approved_version_set
            }
            blocked_decisions: list[EvidenceDecisionDiagnostic] = []
            for acc in fused:
                ch = chunks_by_id_blocked.get(acc.chunk_id)
                if ch is None:
                    continue
                is_partial_valid = acc.structural_rank is not None or resolved_explicit_count > 0
                decision_reason = (
                    "excluded:partial_multi_ref_blocked"
                    if is_partial_valid
                    else "excluded:unresolved_explicit_structural_ref"
                )
                blocked_decisions.append(
                    EvidenceDecisionDiagnostic(
                        chunk_id=ch.chunk_id,
                        document_id=ch.document_id,
                        version_id=ch.version_id,
                        regulation_number=ch.regulation_number,
                        amendment_number=ch.amendment_number,
                        section_number=ch.section_number,
                        pdf_pages=ch.pdf_pages,
                        trust_level=EvidenceTrustLevel.BLOCKED_UNVERIFIED.value,
                        included=False,
                        reason=decision_reason,
                        paragraph_number=(
                            acc.target_ref.paragraph_number or "" if acc.target_ref else ""
                        ),
                        item_number=(
                            f"{acc.target_ref.item_number or ''}"
                            f"{acc.target_ref.subitem_number or ''}"
                            if acc.target_ref
                            else ""
                        ),
                    )
                )
            for u_ref, u_status, _ in unresolved_explicit_refs:
                blocked_decisions.append(
                    EvidenceDecisionDiagnostic(
                        chunk_id=f"unresolved::{u_ref.canonical_label}",
                        document_id="",
                        version_id="",
                        regulation_number=u_ref.regulation_number or "",
                        amendment_number=u_ref.amendment_number or "",
                        section_number=u_ref.section_number or "",
                        pdf_pages=(),
                        trust_level=EvidenceTrustLevel.BLOCKED_UNVERIFIED.value,
                        included=False,
                        reason=f"excluded:explicit_ref_{u_status.value}({u_ref.canonical_label})",
                        paragraph_number=u_ref.paragraph_number or "",
                        item_number=f"{u_ref.item_number or ''}{u_ref.subitem_number or ''}",
                    )
                )
            rrf_diag_blocked = tuple(
                RrfCandidateDiagnostic(
                    chunk_id=acc.chunk_id,
                    rrf_score=acc.score,
                    structural_rank=acc.structural_rank,
                    lexical_rank=acc.lexical_rank,
                    vector_rank=acc.vector_rank,
                    sources=tuple(acc.sources or ()),
                    rerank_score=acc.rerank_score,
                    target_ref=acc.target_ref.canonical_label if acc.target_ref else "",
                )
                for acc in fused
            )
            blocked_subunit_evidences = self._build_blocked_subunit_evidences(
                unresolved_subunit_refs=unresolved_explicit_refs,
                chunks_by_id=chunks_by_id_blocked,
            )
            diagnostics = RetrievalDiagnostics(
                query=clean_query,
                effective_filters=self._serialize_filters(effective_filters),
                chroma_where_filter=serialized_chroma_where,
                resolved_references=resolved_refs,
                structural_hit_ids=tuple(structural_hit_ids),
                lexical_hits=tuple(lexical_diagnostics),
                vector_hits=tuple(vector_diagnostics),
                rrf_candidates=rrf_diag_blocked,
                evidence_decisions=tuple(blocked_decisions),
                max_vector_distance=self._max_vector_distance,
                version_ambiguity_detected=False,
                reranker_enabled=self._enable_reranker,
                adaptive_cutoff_enabled=self._enable_adaptive_cutoff,
            )
            return HybridRetrievalResult(
                query=clean_query,
                resolved_references=resolved_refs,
                evidences=(),
                blocked_evidences=tuple(blocked_subunit_evidences),
                lexical_hits_count=len(lexical_pairs),
                vector_hits_count=vector_hits_count,
                structural_hits_count=structural_hits,
                vector_search_available=vector_available,
                warnings=tuple(dict.fromkeys(warnings)),
                diagnostics=diagnostics,
            )

        if structural_hits > 0:
            structural_only = [acc for acc in fused if acc.structural_rank is not None]
            if structural_only:
                fused = structural_only

        chunk_ids = list(dict.fromkeys(item.chunk_id for item in fused))
        approved_version_set = set(approved_version_ids)
        chunks_by_id = {
            c.chunk_id: c
            for c in self._repository.get_chunks_by_ids(chunk_ids)
            if c.version_id in approved_version_set
        }

        # Mesmo em modos sem busca estrutural (`lexical_fts5_only` / `vector_only`), se a pergunta
        # citou uma subunidade explícita da seção recuperada, aplica o escopo granular na hidratação
        # para impedir expansão indevida para a seção inteira.
        if not self._enable_structural and explicit_subunit_refs:
            expanded_fused: list[_RrfAccumulator] = []
            for acc in fused:
                ch = chunks_by_id.get(acc.chunk_id)
                if ch is None:
                    continue
                matching_refs = [
                    r
                    for r in explicit_subunit_refs
                    if r.section_number
                    and r.section_number.strip().lower() == ch.section_number.strip().lower()
                    and (
                        not r.regulation_number
                        or r.regulation_number.strip().lower()
                        == ch.regulation_number.strip().lower()
                    )
                ]
                if matching_refs:
                    for mref in matching_refs:
                        expanded_fused.append(
                            _RrfAccumulator(
                                chunk_id=acc.chunk_id,
                                target_ref=mref,
                                target_key=mref.target_key,
                                score=acc.score,
                                sources=list(acc.sources or ()),
                                structural_rank=acc.structural_rank,
                                lexical_rank=acc.lexical_rank,
                                vector_rank=acc.vector_rank,
                                bm25_score=acc.bm25_score,
                                vector_distance=acc.vector_distance,
                                rerank_score=acc.rerank_score,
                            )
                        )
                else:
                    expanded_fused.append(acc)
            fused = expanded_fused

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
                    item.target_key,
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
                target_ref=acc.target_ref.canonical_label if acc.target_ref else "",
            )
            for acc in fused
        )

        usable_evidences: list[FusedEvidence] = []
        blocked_integrity_evidences: list[FusedEvidence] = []
        evidence_decisions: list[EvidenceDecisionDiagnostic] = []
        seen_targets: set[tuple[str, str, str]] = set()
        # Cache efêmero restrito a uma única chamada `retrieve()` (zero estado persistente).
        docs_cache: dict[str, RegulatoryDocument | None] = {}
        versions_cache: dict[str, DocumentVersion | None] = {}
        all_tables_cache: dict[str, list[NormativeTable]] = {}

        for acc in fused:
            hydrated_chunk = chunks_by_id.get(acc.chunk_id)
            if hydrated_chunk is None:
                continue
            target_key = acc.target_key
            dedupe_key = (
                hydrated_chunk.version_id,
                hydrated_chunk.section_number.lower(),
                target_key,
            )
            if (self._reconstruct_full_section or target_key) and dedupe_key in seen_targets:
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
                        reason="excluded:duplicate_target_reconstructed",
                    )
                )
                continue

            evidence = self._hydrate_evidence_from_ssot(
                chunk=hydrated_chunk,
                acc=acc,
                target_ref=acc.target_ref,
                docs_cache=docs_cache,
                versions_cache=versions_cache,
                all_pages_cache=all_pages_cache,
                all_tables_cache=all_tables_cache,
                units_cache=units_cache,
                doc_assessments=doc_assessments,
            )
            if evidence is None:
                if acc.target_ref is not None and acc.target_ref.has_subunit_target:
                    evidence_decisions.append(
                        EvidenceDecisionDiagnostic(
                            chunk_id=hydrated_chunk.chunk_id,
                            document_id=hydrated_chunk.document_id,
                            version_id=hydrated_chunk.version_id,
                            regulation_number=hydrated_chunk.regulation_number,
                            amendment_number=hydrated_chunk.amendment_number,
                            section_number=hydrated_chunk.section_number,
                            pdf_pages=hydrated_chunk.pdf_pages,
                            trust_level=EvidenceTrustLevel.BLOCKED_UNVERIFIED.value,
                            included=False,
                            reason=f"excluded:subunit_not_found({acc.target_ref.canonical_label})",
                        )
                    )
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
                        unit_id=evidence.unit_id,
                        paragraph_number=evidence.paragraph_number,
                        item_number=evidence.item_number,
                    )
                )
                continue

            seen_targets.add(dedupe_key)
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
                        unit_id=evidence.unit_id,
                        paragraph_number=evidence.paragraph_number,
                        item_number=evidence.item_number,
                    )
                )
            else:
                is_missing_root_block = bool(
                    evidence.warning_message
                    and "UNIDADE-RAIZ AUSENTE NO SSOT" in evidence.warning_message
                )
                is_subunit_block = bool(
                    evidence.warning_message
                    and "GRANULARIDADE NORMATIVA BLOQUEADA" in evidence.warning_message
                )
                is_ownership_block = bool(
                    evidence.warning_message
                    and "PROPRIEDADE TABULAR BLOQUEADA" in evidence.warning_message
                )
                is_integrity_block = bool(
                    evidence.warning_message
                    and "INTEGRIDADE TABULAR BLOQUEADA" in evidence.warning_message
                )
                is_fail_closed_block = (
                    is_missing_root_block
                    or is_subunit_block
                    or is_ownership_block
                    or is_integrity_block
                )
                if is_fail_closed_block:
                    blocked_integrity_evidences.append(evidence)
                    if evidence.warning_message:
                        warnings.append(evidence.warning_message)
                if is_missing_root_block:
                    block_reason = "excluded:missing_root_unit"
                elif is_subunit_block:
                    block_reason = "excluded:subunit_granularity_blocked"
                elif is_ownership_block:
                    block_reason = "excluded:table_ownership_blocked"
                elif is_integrity_block:
                    block_reason = "excluded:table_integrity_failed"
                else:
                    block_reason = "excluded:global_document_approval_required"
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
                        reason=block_reason,
                        unit_id=evidence.unit_id,
                        paragraph_number=evidence.paragraph_number,
                        item_number=evidence.item_number,
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

    def _get_section_units_cached(
        self,
        *,
        version_id: str,
        section_number: str,
        units_cache: dict[tuple[str, str], list[ParsedUnitRecord]] | None,
    ) -> list[ParsedUnitRecord]:
        cache_key = (version_id, section_number.strip().lower())
        if units_cache is not None and cache_key in units_cache:
            return units_cache[cache_key]
        units = self._repository.get_units_by_section(
            version_id=version_id,
            section_number=section_number,
        )
        if units_cache is not None:
            units_cache[cache_key] = units
        return units

    def _map_approved_version_regulations(
        self,
        approved_version_ids: Sequence[str],
    ) -> dict[str, str]:
        """Mapeia cada `version_id` aprovado ao respectivo `regulation_number` normalizado."""
        docs_by_id = {d.document_id: d for d in self._repository.search_documents()}
        reg_by_ver: dict[str, str] = {}
        for vid in approved_version_ids:
            ver = self._repository.get_version(vid)
            if ver is None:
                continue
            doc = docs_by_id.get(ver.document_id)
            if doc is not None:
                reg_by_ver[vid] = doc.regulation_number.strip().lower()
        return reg_by_ver

    @staticmethod
    def _group_explicit_references(
        resolved_refs: Sequence[ResolvedStructuralRef],
    ) -> list[list[ResolvedStructuralRef]]:
        """Agrupa aliases gerados da mesma menção de apêndice (`Apêndice A` + `A43.1`)."""
        groups: list[list[ResolvedStructuralRef]] = []
        for ref in resolved_refs:
            if not ref.section_number:
                continue
            if (
                groups
                and not ref.has_subunit_target
                and not groups[-1][0].has_subunit_target
                and groups[-1][0].raw_match.strip().lower() == ref.raw_match.strip().lower()
                and _APPENDIX_ALIAS_SEC_RE.match(groups[-1][0].section_number or "")
                and _APPENDIX_ALIAS_SEC_RE.match(ref.section_number)
            ):
                groups[-1].append(ref)
            else:
                groups.append([ref])
        return groups

    def _evaluate_explicit_references_against_ssot(
        self,
        *,
        resolved_refs: Sequence[ResolvedStructuralRef],
        approved_version_ids: Sequence[str],
        approved_version_regs: Mapping[str, str],
        units_cache: dict[tuple[str, str], list[ParsedUnitRecord]],
    ) -> tuple[int, list[tuple[ResolvedStructuralRef, SubunitSelectionStatus, str]]]:
        """Audita todas as referências estruturais explícitas da consulta diretamente no SSOT.

        Retorna `(quantidade_resolvidas, lista_de_referencias_nao_resolvidas)` sem depender do
        modo de busca (estrutural, híbrido, lexical-only ou vetorial-only).
        """
        if not resolved_refs:
            return 0, []

        approved_reg_set = set(approved_version_regs.values())
        groups = self._group_explicit_references(resolved_refs)
        resolved_count = 0
        unresolved: list[tuple[ResolvedStructuralRef, SubunitSelectionStatus, str]] = []

        for group in groups:
            is_mandatory = any(
                r.has_subunit_target
                or bool(_EXPLICIT_REF_PREFIX_RE.search(r.raw_match))
                or bool(
                    r.regulation_number
                    and r.regulation_number.strip().lower() in approved_reg_set
                )
                for r in group
            )
            if not is_mandatory:
                continue

            group_resolved = False
            best_ref = group[0]
            best_status = SubunitSelectionStatus.NOT_FOUND
            best_reason = (
                f"Dispositivo '{best_ref.canonical_label}' não encontrado em nenhuma versão "
                "documental aprovada no escopo da consulta."
            )

            for ref in group:
                if not ref.section_number:
                    continue
                target_reg = ref.regulation_number.strip().lower() if ref.regulation_number else ""
                candidate_vids = [
                    vid
                    for vid in approved_version_ids
                    if not target_reg or approved_version_regs.get(vid, "") == target_reg
                ]
                for vid in candidate_vids:
                    sec_units = self._get_section_units_cached(
                        version_id=vid,
                        section_number=ref.section_number,
                        units_cache=units_cache,
                    )
                    if not sec_units:
                        continue
                    sel = select_structural_units(sec_units, ref)
                    if sel.found:
                        group_resolved = True
                        break
                    if (
                        sel.status
                        in (
                            SubunitSelectionStatus.AMBIGUOUS,
                            SubunitSelectionStatus.VERIFICATION_FAILED,
                        )
                        or best_status == SubunitSelectionStatus.NOT_FOUND
                    ):
                        best_ref = ref
                        best_status = sel.status
                        best_reason = sel.reason
                if group_resolved:
                    break

            if group_resolved:
                resolved_count += 1
            else:
                unresolved.append((best_ref, best_status, best_reason))

        return resolved_count, unresolved

    @staticmethod
    def _build_blocked_subunit_evidences(
        *,
        unresolved_subunit_refs: Sequence[
            tuple[ResolvedStructuralRef, SubunitSelectionStatus, str]
        ],
        chunks_by_id: Mapping[str, NormativeChunk],
    ) -> list[FusedEvidence]:
        """Registra evidências bloqueadas sem texto literal para TODA referência não resolvida."""
        blocked: list[FusedEvidence] = []
        for ref, status, reason in unresolved_subunit_refs:
            sec_lower = (ref.section_number or "").strip().lower()
            anchor = next(
                (
                    c
                    for c in chunks_by_id.values()
                    if c.section_number.strip().lower() == sec_lower
                ),
                None,
            )
            if status == SubunitSelectionStatus.NOT_FOUND:
                warning_msg = (
                    f"UNIDADE NORMATIVA NÃO ENCONTRADA ({ref.canonical_label}): {reason}"
                )
            elif not ref.has_subunit_target and "Unidade-raiz" in reason:
                warning_msg = f"UNIDADE-RAIZ AUSENTE NO SSOT ({ref.canonical_label}): {reason}"
            else:
                warning_msg = (
                    f"GRANULARIDADE NORMATIVA BLOQUEADA ({ref.canonical_label}): {reason}"
                )
            blocked.append(
                FusedEvidence(
                    chunk_id=anchor.chunk_id if anchor else f"blocked::{ref.canonical_label}",
                    unit_id="",
                    document_id=anchor.document_id if anchor else "",
                    version_id=anchor.version_id if anchor else "",
                    authority=anchor.authority if anchor else "ANAC",
                    document_type=anchor.document_type if anchor else "RBAC",
                    regulation_number=(
                        anchor.regulation_number if anchor else (ref.regulation_number or "")
                    ),
                    document_title=anchor.title if anchor else "",
                    amendment_number=(
                        anchor.amendment_number if anchor else (ref.amendment_number or "")
                    ),
                    version_status=anchor.version_status if anchor else "",
                    official_url="",
                    section_number=ref.section_number or "",
                    subsection_number="",
                    paragraph_number=ref.paragraph_number or "",
                    item_number=(
                        f"{ref.item_number or ''}{ref.subitem_number or ''}{ref.leaf_number or ''}"
                    ),
                    title=anchor.title if anchor else "",
                    original_text="",
                    pdf_pages=(),
                    printed_pages=(),
                    page_statuses=(),
                    trust_level=EvidenceTrustLevel.BLOCKED_UNVERIFIED,
                    literal_transcription_allowed=False,
                    rrf_score=0.0,
                    matched_sources=(RetrievalSource.EXACT_STRUCTURAL,),
                    warning_message=warning_msg,
                    tables=(),
                )
            )
        return blocked

    def _hydrate_evidence_from_ssot(
        self,
        *,
        chunk: NormativeChunk,
        acc: _RrfAccumulator,
        target_ref: ResolvedStructuralRef | None = None,
        docs_cache: dict[str, RegulatoryDocument | None] | None = None,
        versions_cache: dict[str, DocumentVersion | None] | None = None,
        all_pages_cache: dict[str, list[RegulatoryPage]] | None = None,
        all_tables_cache: dict[str, list[NormativeTable]] | None = None,
        units_cache: dict[tuple[str, str], list[ParsedUnitRecord]] | None = None,
        doc_assessments: dict[str, DocumentVersionValidationAssessment] | None = None,
    ) -> FusedEvidence | None:
        """Recupera texto literal na granularidade exata e valida aprovação ao vivo no SQLite."""
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

        units = self._get_section_units_cached(
            version_id=version.version_id,
            section_number=chunk.section_number,
            units_cache=units_cache,
        )
        root_units = [u for u in units if u.parent_unit_id is None]
        primary_unit = root_units[0] if root_units else None

        is_subunit_query = bool(target_ref is not None and target_ref.has_subunit_target)
        if is_subunit_query and target_ref is not None:
            selection = select_structural_units(units, target_ref)
            if not selection.found:
                if selection.status in (
                    SubunitSelectionStatus.AMBIGUOUS,
                    SubunitSelectionStatus.VERIFICATION_FAILED,
                ):
                    return FusedEvidence(
                        chunk_id=chunk.chunk_id,
                        unit_id="",
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
                        paragraph_number=target_ref.paragraph_number or "",
                        item_number=(
                            f"{target_ref.item_number or ''}{target_ref.subitem_number or ''}"
                        ),
                        title=(primary_unit.title if primary_unit else chunk.title),
                        original_text="",
                        pdf_pages=(),
                        printed_pages=(),
                        page_statuses=(),
                        trust_level=EvidenceTrustLevel.BLOCKED_UNVERIFIED,
                        literal_transcription_allowed=False,
                        rrf_score=acc.score,
                        matched_sources=tuple(acc.sources or ()),
                        lexical_rank=acc.lexical_rank,
                        vector_rank=acc.vector_rank,
                        structural_rank=acc.structural_rank,
                        rerank_score=acc.rerank_score,
                        warning_message=(
                            f"GRANULARIDADE NORMATIVA BLOQUEADA ({target_ref.canonical_label}): "
                            f"{selection.reason}"
                        ),
                        tables=(),
                    )
                return None

            original_text = selection.original_text
            page_numbers = list(selection.page_numbers)
            structure_status = selection.structure_status
            title = (
                selection.root_unit.title
                if selection.root_unit is not None and selection.root_unit.title
                else chunk.title
            )
            effective_unit_id = selection.unit_id
            effective_paragraph = selection.paragraph_number
            effective_item = selection.item_number
        elif primary_unit is None:
            # Fail-closed: na ausência de unidade-raiz (`parent_unit_id=None`) no SSOT,
            # jamais utilizar a primeira subunidade ou texto parcial como seção completa.
            return FusedEvidence(
                chunk_id=chunk.chunk_id,
                unit_id="",
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
                paragraph_number="",
                item_number="",
                title=chunk.title,
                original_text="",
                pdf_pages=(),
                printed_pages=(),
                page_statuses=(),
                trust_level=EvidenceTrustLevel.BLOCKED_UNVERIFIED,
                literal_transcription_allowed=False,
                rrf_score=acc.score,
                matched_sources=tuple(acc.sources or ()),
                lexical_rank=acc.lexical_rank,
                vector_rank=acc.vector_rank,
                structural_rank=acc.structural_rank,
                rerank_score=acc.rerank_score,
                warning_message=(
                    f"UNIDADE-RAIZ AUSENTE NO SSOT ({chunk.section_number}): Não existe "
                    "unidade-raiz (`parent_unit_id=None`) no SQLite para esta seção; por "
                    "política fail-closed, nenhuma subunidade parcial é devolvida como "
                    "seção completa."
                ),
                tables=(),
            )
        elif self._reconstruct_full_section:
            original_text = primary_unit.original_text
            page_numbers = sorted({p for u in units for p in u.page_numbers})
            structure_status = (
                StructureStatus.PARSED
                if all(u.structure_status == StructureStatus.PARSED for u in root_units)
                else StructureStatus.PENDING_VALIDATION
            )
            title = primary_unit.title or chunk.title
            effective_unit_id = primary_unit.unit_id
            effective_paragraph = chunk.paragraph_number
            effective_item = chunk.item_number
        else:
            original_text = chunk.original_text
            page_numbers = list(chunk.pdf_pages)
            structure_status = primary_unit.structure_status
            title = primary_unit.title or chunk.title
            effective_unit_id = primary_unit.unit_id
            effective_paragraph = chunk.paragraph_number
            effective_item = chunk.item_number

        pages_by_num = {p.pdf_page_number: p for p in all_version_pages}
        pages = [pages_by_num[p_num] for p_num in page_numbers if p_num in pages_by_num]

        assessment = self._validation_service.evaluate_section(
            section_number=chunk.section_number,
            pages=pages,
            structure_status=structure_status,
            document_assessment=doc_assessment,
        )

        # Hidrata tabelas/formulários validando propriedade e integridade separadamente
        if all_tables_cache is not None and version.version_id in all_tables_cache:
            version_tables = all_tables_cache[version.version_id]
        else:
            version_tables = self._repository.list_tables(version.version_id)
            if all_tables_cache is not None:
                all_tables_cache[version.version_id] = version_tables
        page_num_set = set(page_numbers)
        raw_page_tables = [t for t in version_tables if t.pdf_page_number in page_num_set]

        target_section_unit_id = primary_unit.unit_id if primary_unit is not None else chunk.unit_id
        section_tables, ownership_issues, table_integrity_issues = resolve_section_tables(
            page_tables=raw_page_tables,
            unit_id=target_section_unit_id,
            section_number=chunk.section_number,
        )
        if is_subunit_query:
            effective_tables = tuple(
                t
                for t in section_tables
                if t.unit_id == effective_unit_id
                or t.ownership_status == TableOwnershipStatus.PAGE_GLOBAL.value
            )
        else:
            effective_tables = section_tables

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
        if ownership_issues:
            trust_level = EvidenceTrustLevel.BLOCKED_UNVERIFIED
            effective_allowed = False
            effective_warning = (
                f"PROPRIEDADE TABULAR BLOQUEADA ({chunk.section_number}): "
                "Há tabela na página sem associação proprietária única e confiável; "
                "resposta bloqueada para evitar atribuição indevida entre seções "
                f"({'; '.join(ownership_issues[:3])})."
            )
            hydrated_tables: tuple[NormativeTable, ...] = ()
        elif table_integrity_issues:
            trust_level = EvidenceTrustLevel.BLOCKED_UNVERIFIED
            effective_allowed = False
            effective_warning = (
                f"INTEGRIDADE TABULAR BLOQUEADA ({chunk.section_number}): "
                "A estrutura tabular desta seção apresentou inconsistência e foi bloqueada "
                "para evitar apresentação de valores sem cabeçalhos ou associação confiável "
                f"({'; '.join(table_integrity_issues[:3])})."
            )
            hydrated_tables = ()
        elif assessment.literal_transcription_allowed:
            trust_level = EvidenceTrustLevel.VALIDATED_LITERAL
            hydrated_tables = effective_tables
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
            unit_id=effective_unit_id,
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
            paragraph_number=effective_paragraph,
            item_number=effective_item,
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
