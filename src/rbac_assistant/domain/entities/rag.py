"""Entidades puras de domínio para indexação, busca híbrida (FTS5 + ChromaDB + RRF) e RAG."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from rbac_assistant.domain.entities.normative_unit import NormativeTable
from rbac_assistant.domain.entities.regulatory_document import DocumentType, VersionStatus
from rbac_assistant.domain.entities.regulatory_page import PageValidationStatus


class EvidenceTrustLevel(StrEnum):
    """Nível de confiabilidade normativa de uma evidência recuperada sob `review_on_demand`."""

    VALIDATED_LITERAL = "validated_literal"
    PRELIMINARY_UNREVIEWED = "preliminary_unreviewed"
    BLOCKED_OCR_PENDING = "blocked_ocr_pending"
    BLOCKED_REJECTED = "blocked_rejected"
    BLOCKED_UNVERIFIED = "blocked_unverified"


class RetrievalSource(StrEnum):
    """Origem individual de um candidato antes da fusão RRF."""

    EXACT_STRUCTURAL = "exact_structural"
    LEXICAL_FTS5 = "lexical_fts5"
    SEMANTIC_VECTOR = "semantic_vector"


@dataclass(frozen=True, slots=True)
class NormativeChunk:
    """Chunk estrutural determinístico vinculado ao SSOT SQLite e indexado no FTS5/ChromaDB."""

    chunk_id: str
    document_id: str
    version_id: str
    unit_id: str
    authority: str
    document_type: str
    regulation_number: str
    amendment_number: str
    version_status: str
    section_number: str
    subsection_number: str
    paragraph_number: str
    item_number: str
    title: str
    subpart: str
    start_page: int
    end_page: int
    pdf_pages: tuple[int, ...]
    printed_pages: tuple[str, ...]
    chunk_index: int
    original_text: str
    search_text: str
    source_hash: str
    content_hash: str
    validation_status: str
    embedding_model: str = ""
    embedding_dimension: int = 0

    def __post_init__(self) -> None:
        if not self.chunk_id.strip():
            raise ValueError("chunk_id não pode ser vazio.")
        if not self.version_id.strip():
            raise ValueError("version_id não pode ser vazio.")
        if self.start_page < 1 or self.end_page < self.start_page:
            raise ValueError("Intervalo de páginas inválido no NormativeChunk.")
        if self.chunk_index < 0:
            raise ValueError("chunk_index deve ser >= 0.")


@dataclass(frozen=True, slots=True)
class SearchFilter:
    """Filtros estruturais e de governança aplicáveis à busca híbrida."""

    document_id: str | None = None
    authority: str | None = None
    document_type: DocumentType | None = None
    regulation_number: str | None = None
    amendment_number: str | None = None
    version_id: str | None = None
    version_status: VersionStatus | None = None
    only_current_version: bool = True
    only_approved_literal: bool = True
    include_preliminary_native: bool = False
    allowed_version_ids: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class ResolvedStructuralRef:
    """Referência estrutural explícita detectada na pergunta (ex.: RBAC 43.3, 43.9(a)(1)(i))."""

    raw_match: str
    regulation_number: str | None = None
    section_number: str | None = None
    paragraph_number: str | None = None
    item_number: str | None = None
    subitem_number: str | None = None
    amendment_number: str | None = None
    leaf_number: str | None = None

    @property
    def has_subunit_target(self) -> bool:
        """Retorna `True` quando a referência especifica parágrafo, inciso, alínea ou subitem."""
        return bool(
            self.paragraph_number or self.item_number or self.subitem_number or self.leaf_number
        )

    @property
    def canonical_label(self) -> str:
        """Formata o identificador estrutural compacto (ex.: `43.10(a)(1)` ou `43.10(c)(5)(i)`)."""
        parts = [
            (self.section_number or "").strip(),
            (self.paragraph_number or "").strip(),
            (self.item_number or "").strip(),
            (self.subitem_number or "").strip(),
            (self.leaf_number or "").strip(),
        ]
        return "".join(p for p in parts if p)

    @property
    def target_key(self) -> str:
        """Chave determinística da granularidade alvo para fusão RRF e deduplicação."""
        if not self.has_subunit_target:
            return ""
        return "|".join(
            (
                (self.section_number or "").strip().lower(),
                (self.paragraph_number or "").strip().lower(),
                (self.item_number or "").strip().lower(),
                (self.subitem_number or "").strip().lower(),
                (self.leaf_number or "").strip(),
            )
        )


@dataclass(frozen=True, slots=True)
class RetrievalCandidate:
    """Candidato retornado por uma fonte individual antes da fusão por RRF."""

    chunk_id: str
    source: RetrievalSource
    rank: int
    raw_score: float
    target_ref: ResolvedStructuralRef | None = None


@dataclass(frozen=True, slots=True)
class LexicalHitDiagnostic:
    """Diagnóstico de candidato recuperado via SQLite FTS5 (BM25)."""

    chunk_id: str
    rank: int
    bm25_score: float


@dataclass(frozen=True, slots=True)
class VectorHitDiagnostic:
    """Diagnóstico de candidato recuperado via ChromaDB (distância cosseno e limiar)."""

    chunk_id: str
    rank: int
    distance: float
    cosine_similarity: float
    accepted: bool
    rejection_reason: str = ""


@dataclass(frozen=True, slots=True)
class RrfCandidateDiagnostic:
    """Diagnóstico de pontuação RRF consolidada por chunk e referência estrutural alvo."""

    chunk_id: str
    rrf_score: float
    structural_rank: int | None
    lexical_rank: int | None
    vector_rank: int | None
    sources: tuple[RetrievalSource, ...]
    rerank_score: float | None = None
    target_ref: str = ""


@dataclass(frozen=True, slots=True)
class EvidenceDecisionDiagnostic:
    """Diagnóstico de inclusão/exclusão de cada evidência hidratada do SQLite."""

    chunk_id: str
    document_id: str
    version_id: str
    regulation_number: str
    amendment_number: str
    section_number: str
    pdf_pages: tuple[int, ...]
    trust_level: str
    included: bool
    reason: str
    unit_id: str = ""
    paragraph_number: str = ""
    item_number: str = ""


@dataclass(frozen=True, slots=True)
class RetrievalDiagnostics:
    """Rastro completo e determinístico da execução de busca híbrida para modo --debug."""

    query: str
    effective_filters: tuple[tuple[str, str], ...] = ()
    chroma_where_filter: tuple[tuple[str, str], ...] = ()
    resolved_references: tuple[ResolvedStructuralRef, ...] = ()
    structural_hit_ids: tuple[str, ...] = ()
    lexical_hits: tuple[LexicalHitDiagnostic, ...] = ()
    vector_hits: tuple[VectorHitDiagnostic, ...] = ()
    rrf_candidates: tuple[RrfCandidateDiagnostic, ...] = ()
    evidence_decisions: tuple[EvidenceDecisionDiagnostic, ...] = ()
    max_vector_distance: float = 0.85
    version_ambiguity_detected: bool = False
    reranker_enabled: bool = False
    adaptive_cutoff_enabled: bool = False


@dataclass(frozen=True, slots=True)
class CitationValidationResult:
    """Resultado da validação server-side das citações produzidas pelo modelo generativo."""

    sanitized_explanation: str
    valid_evidence_ids: tuple[str, ...] = ()
    invalid_evidence_ids: tuple[str, ...] = ()
    valid_section_citations: tuple[str, ...] = ()
    unverified_section_citations: tuple[str, ...] = ()
    valid_page_citations: tuple[str, ...] = ()
    unverified_page_citations: tuple[str, ...] = ()
    citation_precision: float = 1.0
    had_modifications: bool = False


@dataclass(frozen=True, slots=True)
class FusedEvidence:
    """Evidência hidratada a partir do SQLite (SSOT) após fusão RRF e verificação de revisão."""

    chunk_id: str
    unit_id: str
    document_id: str
    version_id: str
    authority: str
    document_type: str
    regulation_number: str
    document_title: str
    amendment_number: str
    version_status: str
    official_url: str
    section_number: str
    subsection_number: str
    paragraph_number: str
    item_number: str
    title: str
    original_text: str
    pdf_pages: tuple[int, ...]
    printed_pages: tuple[str, ...]
    page_statuses: tuple[tuple[int, PageValidationStatus], ...]
    trust_level: EvidenceTrustLevel
    literal_transcription_allowed: bool
    rrf_score: float
    matched_sources: tuple[RetrievalSource, ...]
    pending_pdf_pages: tuple[int, ...] = ()
    ocr_blocked_pdf_pages: tuple[int, ...] = ()
    rejected_pdf_pages: tuple[int, ...] = ()
    lexical_rank: int | None = None
    vector_rank: int | None = None
    structural_rank: int | None = None
    rerank_score: float | None = None
    warning_message: str | None = None
    tables: tuple[NormativeTable, ...] = ()


@dataclass(frozen=True, slots=True)
class HybridRetrievalResult:
    """Resultado consolidado da busca híbrida (estrutural + FTS5 + vetorial + RRF)."""

    query: str
    resolved_references: tuple[ResolvedStructuralRef, ...] = ()
    evidences: tuple[FusedEvidence, ...] = ()
    blocked_evidences: tuple[FusedEvidence, ...] = ()
    lexical_hits_count: int = 0
    vector_hits_count: int = 0
    structural_hits_count: int = 0
    index_empty: bool = False
    vector_search_available: bool = True
    warnings: tuple[str, ...] = ()
    diagnostics: RetrievalDiagnostics | None = None
    blocked_by_document_approval: bool = False
    blocked_version_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class StructuredNormativeAnswer:
    """Resposta final estruturada separando rigorosamente texto normativo (SSOT) e explicação IA."""

    query: str
    found: bool
    normative_text: str | None
    preliminary_excerpt: str | None
    explanation: str
    references: tuple[str, ...]
    observations: tuple[str, ...]
    evidences: tuple[FusedEvidence, ...] = ()
    blocked_evidences: tuple[FusedEvidence, ...] = ()
    is_literal_validated: bool = False
    is_preliminary_only: bool = False
    has_blocked_sources: bool = False
    generation_model: str | None = None
    used_fallback_without_llm: bool = False
    formatted_text: str = ""
    evidences_sent_to_llm: tuple[str, ...] = ()
    used_evidence_ids: tuple[str, ...] = ()
    discarded_citations: tuple[str, ...] = ()
    diagnostics: RetrievalDiagnostics | None = None
    literal_excerpts: tuple[str, ...] = ()
    structured_tables_text: str | None = None

    def __post_init__(self) -> None:
        if self.normative_text is not None and not self.is_literal_validated:
            raise ValueError(
                "normative_text só pode ser preenchido quando is_literal_validated=True."
            )
        if self.preliminary_excerpt is not None or self.is_preliminary_only:
            raise ValueError(
                "Excertos preliminares não revisados são proibidos pela política global "
                "de aprovação documental."
            )


@dataclass(frozen=True, slots=True)
class IndexRegistryEntry:
    """Registro persistido de estado de indexação vetorial por coleção e versão documental."""

    collection_name: str
    version_id: str
    document_id: str
    source_hash: str
    embedding_model: str
    embedding_dimension: int
    chunk_count: int
    indexed_at: datetime
    instruction_strategy_version: str = ""


@dataclass(frozen=True, slots=True)
class IndexVersionSummary:
    """Diagnóstico de indexação de uma versão documental cadastrada no SQLite."""

    document_id: str
    version_id: str
    regulation_number: str
    amendment_number: str
    version_status: str
    sqlite_chunk_count: int
    indexed_chunk_count: int
    is_indexed: bool
    is_stale: bool
    stale_reason: str = ""


@dataclass(frozen=True, slots=True)
class IndexStatusReport:
    """Relatório consolidado do estado dos índices lexical (FTS5) e vetorial (ChromaDB)."""

    collection_name: str
    configured_embedding_model: str
    configured_embedding_dimension: int
    collection_embedding_model: str | None
    collection_embedding_dimension: int | None
    chroma_available: bool
    chroma_vector_count: int
    total_sqlite_chunks: int
    total_documents: int
    total_versions: int
    indexed_versions_count: int
    unindexed_versions: tuple[str, ...] = ()
    stale_versions: tuple[str, ...] = ()
    version_summaries: tuple[IndexVersionSummary, ...] = ()
    compatibility_warning: str | None = None


@dataclass(frozen=True, slots=True)
class IndexingOutcome:
    """Resultado determinístico da indexação ou reindexação controlada de uma versão."""

    document_id: str
    version_id: str
    collection_name: str
    embedding_model: str
    embedding_dimension: int
    chunks_created: int
    stale_chunks_removed: int
    skipped_up_to_date: bool = False
    warnings: tuple[str, ...] = field(default_factory=tuple)
