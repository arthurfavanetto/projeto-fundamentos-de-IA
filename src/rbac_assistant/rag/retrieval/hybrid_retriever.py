"""Motor de busca híbrida (Estrutural + SQLite FTS5 + ChromaDB Vetorial + Fusão RRF).

Responsabilidades:
1. Resolver referências estruturais explícitas (ex.: `RBAC 43.3`, `43.9(a)`, `Art. 1º`).
2. Executar busca lexical SQLite FTS5 (BM25, tolerante a acentos, sem alterar `original_text`).
3. Executar busca semântica vetorial no ChromaDB com filtros de metadados.
4. Fundir e deduplicar candidatos via Reciprocal Rank Fusion (RRF) rastreável.
5. Hidratar evidências diretamente do SQLite (SSOT) e validar ao vivo o estado de revisão
   de todas as páginas abrangidas, bloqueando páginas OCR pendentes ou reprovadas.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import structlog

from rbac_assistant.application.services.validation_service import ValidationService
from rbac_assistant.domain.entities.normative_unit import StructureStatus
from rbac_assistant.domain.entities.rag import (
    EvidenceTrustLevel,
    FusedEvidence,
    HybridRetrievalResult,
    NormativeChunk,
    RetrievalCandidate,
    RetrievalSource,
    SearchFilter,
)
from rbac_assistant.domain.entities.regulatory_document import ValidationPolicy, VersionStatus
from rbac_assistant.domain.entities.regulatory_page import PageValidationStatus
from rbac_assistant.domain.exceptions import (
    EmbeddingCompatibilityError,
    LLMServiceError,
    StorageConnectionError,
)
from rbac_assistant.domain.ports.document_repository_port import DocumentRepositoryPort
from rbac_assistant.domain.ports.rag_ports import EmbeddingPort
from rbac_assistant.domain.ports.vector_store_port import MetadataValue, VectorStorePort
from rbac_assistant.rag.retrieval.section_resolver import resolve_structural_references

logger = structlog.get_logger(__name__)

_SOURCE_WEIGHTS: Mapping[RetrievalSource, float] = {
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

    def add(self, candidate: RetrievalCandidate, rrf_k: int) -> None:
        if self.sources is None:
            self.sources = []
        weight = _SOURCE_WEIGHTS[candidate.source]
        self.score += weight / float(rrf_k + candidate.rank)
        if candidate.source not in self.sources:
            self.sources.append(candidate.source)
        if candidate.source == RetrievalSource.EXACT_STRUCTURAL:
            if self.structural_rank is None or candidate.rank < self.structural_rank:
                self.structural_rank = candidate.rank
        elif candidate.source == RetrievalSource.LEXICAL_FTS5:
            if self.lexical_rank is None or candidate.rank < self.lexical_rank:
                self.lexical_rank = candidate.rank
        elif candidate.source == RetrievalSource.SEMANTIC_VECTOR and (
            self.vector_rank is None or candidate.rank < self.vector_rank
        ):
            self.vector_rank = candidate.rank


def fuse_candidates_rrf(
    candidates: Sequence[RetrievalCandidate],
    *,
    rrf_k: int = 60,
) -> list[_RrfAccumulator]:
    """Combina candidatos de múltiplas fontes usando Reciprocal Rank Fusion (RRF)."""
    by_chunk: dict[str, _RrfAccumulator] = {}
    for cand in candidates:
        acc = by_chunk.get(cand.chunk_id)
        if acc is None:
            acc = _RrfAccumulator(chunk_id=cand.chunk_id)
            by_chunk[cand.chunk_id] = acc
        acc.add(cand, rrf_k)

    return sorted(
        by_chunk.values(),
        key=lambda item: (-item.score, item.chunk_id),
    )


class HybridNormativeRetriever:
    """Recuperador híbrido com SSOT SQLite, ChromaDB, RRF e enforcement de revisão por página."""

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
    ) -> None:
        self._repository = repository
        self._vector_store = vector_store
        self._embedding_port = embedding_port
        self._collection_name = collection_name
        self._validation_service = validation_service or ValidationService(
            policy=ValidationPolicy.REVIEW_ON_DEMAND
        )
        self._lexical_top_k = lexical_top_k
        self._semantic_top_k = semantic_top_k
        self._final_top_k = final_top_k
        self._rrf_k = rrf_k
        self._max_vector_distance = max_vector_distance
        self._reconstruct_full_section = reconstruct_full_section_from_ssot
        self._require_vector_search = require_vector_search

    def retrieve(
        self,
        *,
        query: str,
        filters: SearchFilter | None = None,
        top_k: int | None = None,
    ) -> HybridRetrievalResult:
        """Executa busca híbrida completa e hidrata evidências com validação ao vivo do SQLite."""
        clean_query = query.strip()
        if not clean_query:
            return HybridRetrievalResult(query=query, warnings=("Consulta vazia.",))

        total_chunks = self._repository.count_chunks()
        if total_chunks == 0:
            return HybridRetrievalResult(
                query=clean_query,
                index_empty=True,
                warnings=(
                    "Nenhum documento regulatório está indexado na base no momento. "
                    "Execute a ingestão e a indexação antes de consultar o assistente.",
                ),
            )

        resolved_refs = resolve_structural_references(clean_query)
        effective_filters = self._merge_amendment_hint(filters, resolved_refs)
        limit_final = top_k or self._final_top_k

        candidates: list[RetrievalCandidate] = []
        warnings: list[str] = []

        # 1. Busca estrutural determinística exata (ex.: RBAC 43.3)
        structural_hits = 0
        for ref in resolved_refs:
            if not ref.section_number:
                continue
            matched_chunks = self._repository.find_chunks_by_structure(
                section_number=ref.section_number,
                regulation_number=ref.regulation_number,
                paragraph_number=ref.paragraph_number,
                item_number=ref.item_number,
                filters=effective_filters,
                limit=limit_final,
            )
            for rank, chunk in enumerate(matched_chunks, start=1):
                structural_hits += 1
                candidates.append(
                    RetrievalCandidate(
                        chunk_id=chunk.chunk_id,
                        source=RetrievalSource.EXACT_STRUCTURAL,
                        rank=rank,
                        raw_score=1.0 / rank,
                    )
                )

        # 2. Busca lexical SQLite FTS5 (BM25, tolerante a acentos)
        lexical_pairs = self._repository.search_chunks_fts(
            query=clean_query,
            filters=effective_filters,
            limit=max(self._lexical_top_k, limit_final),
        )
        for rank, (chunk_id, bm25_score) in enumerate(lexical_pairs, start=1):
            candidates.append(
                RetrievalCandidate(
                    chunk_id=chunk_id,
                    source=RetrievalSource.LEXICAL_FTS5,
                    rank=rank,
                    raw_score=bm25_score,
                )
            )

        # 3. Busca semântica vetorial no ChromaDB
        vector_hits_count = 0
        vector_available = True
        if self._embedding_port is not None:
            try:
                self._vector_store.ensure_collection_compatible(
                    collection_name=self._collection_name,
                    embedding_model=self._embedding_port.model_name,
                    embedding_dimension=self._embedding_port.dimensions,
                )
                query_vector = self._embedding_port.embed_query(clean_query)
                where_filter = self._build_chroma_where(effective_filters)
                vector_hits = self._vector_store.query_similar(
                    collection_name=self._collection_name,
                    query_embedding=query_vector,
                    top_k=max(self._semantic_top_k, limit_final),
                    where=where_filter,
                )
                relevant_vector_hits = [
                    hit for hit in vector_hits if hit.distance <= self._max_vector_distance
                ]
                for rank, hit in enumerate(relevant_vector_hits, start=1):
                    vector_hits_count += 1
                    candidates.append(
                        RetrievalCandidate(
                            chunk_id=hit.record_id,
                            source=RetrievalSource.SEMANTIC_VECTOR,
                            rank=rank,
                            raw_score=1.0 / (1.0 + max(0.0, hit.distance)),
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

        if not candidates:
            return HybridRetrievalResult(
                query=clean_query,
                resolved_references=resolved_refs,
                lexical_hits_count=len(lexical_pairs),
                vector_hits_count=vector_hits_count,
                structural_hits_count=structural_hits,
                vector_search_available=vector_available,
                warnings=tuple(warnings),
            )

        # 4. Fusão e deduplicação por RRF
        fused = fuse_candidates_rrf(candidates, rrf_k=self._rrf_k)
        # Quando há correspondência estrutural exata (ex.: "RBAC 43.3"), restringe os resultados
        # às unidades estruturalmente correspondentes para evitar ruído de seções vizinhas
        # (preservando os ranks lexicais/vetoriais fundidos via RRF nessas unidades).
        if structural_hits > 0:
            structural_only = [acc for acc in fused if acc.structural_rank is not None]
            if structural_only:
                fused = structural_only

        chunk_ids = [item.chunk_id for item in fused]
        chunks_by_id = {c.chunk_id: c for c in self._repository.get_chunks_by_ids(chunk_ids)}

        usable_evidences: list[FusedEvidence] = []
        blocked_evidences: list[FusedEvidence] = []
        seen_sections: set[tuple[str, str]] = set()

        for acc in fused:
            hydrated_chunk = chunks_by_id.get(acc.chunk_id)
            if hydrated_chunk is None:
                continue
            section_key = (hydrated_chunk.version_id, hydrated_chunk.section_number.lower())
            if self._reconstruct_full_section and section_key in seen_sections:
                continue

            evidence = self._hydrate_evidence_from_ssot(chunk=hydrated_chunk, acc=acc)
            if evidence is None:
                continue
            if not self._matches_post_hydration_filters(evidence, effective_filters):
                continue

            seen_sections.add(section_key)
            if evidence.trust_level == EvidenceTrustLevel.VALIDATED_LITERAL or (
                evidence.trust_level == EvidenceTrustLevel.PRELIMINARY_UNREVIEWED
                and effective_filters.include_preliminary_native
                and not effective_filters.only_approved_literal
            ):
                usable_evidences.append(evidence)
            else:
                blocked_evidences.append(evidence)

            if len(usable_evidences) >= limit_final:
                break

        return HybridRetrievalResult(
            query=clean_query,
            resolved_references=resolved_refs,
            evidences=tuple(usable_evidences),
            blocked_evidences=tuple(blocked_evidences[:limit_final]),
            lexical_hits_count=len(lexical_pairs),
            vector_hits_count=vector_hits_count,
            structural_hits_count=structural_hits,
            vector_search_available=vector_available,
            warnings=tuple(warnings),
        )

    @staticmethod
    def _merge_amendment_hint(
        filters: SearchFilter | None,
        resolved_refs: Sequence[object],
    ) -> SearchFilter:
        base = filters or SearchFilter()
        if base.amendment_number:
            return base
        for ref in resolved_refs:
            amd = getattr(ref, "amendment_number", None)
            if amd:
                return SearchFilter(
                    document_id=base.document_id,
                    authority=base.authority,
                    document_type=base.document_type,
                    regulation_number=base.regulation_number,
                    amendment_number=str(amd),
                    version_id=base.version_id,
                    version_status=base.version_status,
                    only_current_version=False,
                    only_approved_literal=base.only_approved_literal,
                    include_preliminary_native=base.include_preliminary_native,
                )
        return base

    @staticmethod
    def _build_chroma_where(filters: SearchFilter) -> dict[str, MetadataValue] | None:
        where: dict[str, MetadataValue] = {}
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
        if filters.amendment_number:
            where["amendment_number"] = filters.amendment_number.strip()
        elif filters.version_status is not None:
            where["version_status"] = filters.version_status.value
        elif filters.only_current_version and not filters.version_id:
            where["version_status"] = VersionStatus.CURRENT.value
        return where or None

    def _hydrate_evidence_from_ssot(
        self,
        *,
        chunk: NormativeChunk,
        acc: _RrfAccumulator,
    ) -> FusedEvidence | None:
        """Recupera texto literal e status ao vivo das páginas diretamente do SQLite (SSOT)."""
        document = self._repository.get_document(chunk.document_id)
        version = self._repository.get_version(chunk.version_id)
        if document is None or version is None:
            return None

        # Reconstrói texto integral da seção a partir do SSOT SQLite quando configurado
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

        pages = self._repository.get_pages(version.version_id, page_numbers=page_numbers)
        assessment = self._validation_service.evaluate_section(
            section_number=chunk.section_number,
            pages=pages,
            structure_status=structure_status,
        )

        rejected_pages = tuple(
            p.pdf_page_number
            for p in pages
            if p.validation_status == PageValidationStatus.REJECTED
        )
        page_statuses = tuple((p.pdf_page_number, p.validation_status) for p in pages)
        printed_pages = tuple(
            p.printed_page_number or f"PDF {p.pdf_page_number}" for p in pages
        ) or chunk.printed_pages

        if assessment.literal_transcription_allowed:
            trust_level = EvidenceTrustLevel.VALIDATED_LITERAL
        elif rejected_pages:
            trust_level = EvidenceTrustLevel.BLOCKED_REJECTED
        elif assessment.ocr_blocked_pdf_pages:
            trust_level = EvidenceTrustLevel.BLOCKED_OCR_PENDING
        elif (
            assessment.aggregate_status == PageValidationStatus.NATIVE_QUALITY_CHECKED
            and structure_status == StructureStatus.PARSED
        ):
            trust_level = EvidenceTrustLevel.PRELIMINARY_UNREVIEWED
        else:
            trust_level = EvidenceTrustLevel.BLOCKED_UNVERIFIED

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
            literal_transcription_allowed=assessment.literal_transcription_allowed,
            rrf_score=acc.score,
            matched_sources=tuple(acc.sources or ()),
            pending_pdf_pages=assessment.pending_pdf_pages,
            ocr_blocked_pdf_pages=assessment.ocr_blocked_pdf_pages,
            rejected_pdf_pages=rejected_pages,
            lexical_rank=acc.lexical_rank,
            vector_rank=acc.vector_rank,
            structural_rank=acc.structural_rank,
            warning_message=assessment.warning_message,
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
        if (
            filters.amendment_number
            and evidence.amendment_number.lower() != filters.amendment_number.strip().lower()
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
