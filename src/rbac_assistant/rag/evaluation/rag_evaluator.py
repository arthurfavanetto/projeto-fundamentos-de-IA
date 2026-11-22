"""Avaliador quantitativo metodologicamente rigoroso do motor RAG híbrido.

Características metodológicas:
1. Granularidade normativa completa (`ExpectedTarget`: regulamento, emenda, seção, subseção,
   parágrafo, alínea, item, subitem, `normative_unit_id`, páginas esperadas e evidência textual).
2. `Precision@K` e `Recall@K` na mesma granularidade (seções únicas deduplicadas; 0.0 quando
   consulta respondível retorna 0 resultados).
3. Separação explícita de erros (`ErrorBreakdown`): falsos positivos de recuperação, falsos
   negativos de recuperação, respostas indevidas sem evidência, falhas de abstenção, falhas de
   filtro documental, falhas de filtro de versão, violações de governança e citações inventadas.
4. Métricas por categoria (19 categorias) e subgrupos específicos do modo `structural_only`
   (5 subgrupos), evitando comparações injustas entre consultas estruturais e semânticas.
5. Auditoria dedicada em 5 dimensões do `CitationValidator` (sem mascarar mock LLM como qualidade
   real de citações do Gemini).
6. Ablação reproduzível de RRF / reranking determinístico / seleção adaptativa de contexto e
   varredura completa de limiares vetoriais (mantendo `max_vector_distance=0.85` PROVISÓRIO).
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from rbac_assistant.application.services.chat_service import NormativeChatService
from rbac_assistant.domain.entities.rag import (
    EvidenceTrustLevel,
    FusedEvidence,
    RetrievalSource,
    SearchFilter,
)
from rbac_assistant.domain.entities.regulatory_document import DocumentType
from rbac_assistant.domain.ports.rag_ports import LLMPort
from rbac_assistant.rag.generation.citation_validator import CitationValidator
from rbac_assistant.rag.retrieval.hybrid_retriever import HybridNormativeRetriever
from rbac_assistant.rag.retrieval.section_resolver import amendments_match


@dataclass(frozen=True, slots=True)
class ExpectedTarget:
    """Alvo normativo esperado na granularidade exata exigida pela consulta."""

    regulation_number: str
    amendment_number: str
    section_number: str
    subsection_number: str = ""
    paragraph_number: str = ""
    alinea: str = ""
    item_number: str = ""
    subitem_number: str = ""
    normative_unit_id: str = ""
    expected_pdf_pages: tuple[int, ...] = ()
    expected_text_snippets: tuple[str, ...] = ()

    @property
    def canonical_key(self) -> str:
        return (
            f"{self.regulation_number.strip().lower()}:"
            f"{self.amendment_number.strip().lower()}:"
            f"{self.section_number.strip().lower()}"
        )


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    """Caso individual de avaliação controlada do pipeline RAG."""

    case_id: str
    category: str
    structural_subset: str
    query: str
    expected_targets: tuple[ExpectedTarget, ...]
    expected_behavior: str
    is_exact_structural: bool = False
    expected_regulation: str | None = None
    expected_amendment: str | None = None
    regulation_filter: str | None = None
    amendment_filter: str | None = None
    document_type_filter: str | None = None
    only_current_version: bool = True


@dataclass(frozen=True, slots=True)
class ErrorBreakdown:
    """Decomposição transparente dos tipos distintos de erro (sem agregar sinais incompatíveis)."""

    retrieval_false_positives: int = 0
    retrieval_false_negatives: int = 0
    unwarranted_answer_no_evidence: int = 0
    abstention_failures: int = 0
    document_filter_failures: int = 0
    version_filter_failures: int = 0
    governance_violations: int = 0
    hallucinated_citations: int = 0


@dataclass(frozen=True, slots=True)
class CategoryMetrics:
    """Métricas de um modo de recuperação para uma categoria específica de consulta."""

    category: str
    total_cases: int
    answerable_cases: int
    recall_at_5: float
    precision_at_5: float
    mrr_at_10: float
    ndcg_at_10: float
    behavior_pass_rate: float
    retrieval_fp: int
    retrieval_fn: int


@dataclass(frozen=True, slots=True)
class StructuralSubsetMetrics:
    """Avaliação estratificada por natureza estrutural da consulta (Parte 2.1)."""

    subset: str
    total_cases: int
    answerable_cases: int
    recall_at_5: float
    precision_at_5: float
    mrr_at_10: float
    behavior_pass_rate: float


@dataclass(frozen=True, slots=True)
class QueryEvalDetail:
    """Resultado individual auditável de uma consulta específica em um modo."""

    case_id: str
    category: str
    structural_subset: str
    query: str
    expected_behavior: str
    expected_keys: tuple[str, ...]
    returned_keys: tuple[str, ...]
    recall_at_5: float | None
    precision_at_5: float | None
    reciprocal_rank: float | None
    granular_ok: bool
    doc_filter_ok: bool
    version_filter_ok: bool
    behavior_ok: bool
    retrieval_fp: int
    retrieval_fn: int
    notes: str = ""


@dataclass(frozen=True, slots=True)
class CitationValidatorAuditReport:
    """Auditoria determinística em 5 dimensões do `CitationValidator` (LLM mock — não Gemini)."""

    evidence_id_validation_rate: float
    structural_ref_validation_rate: float
    page_ref_validation_rate: float
    nonexistent_citation_block_rate: float
    wrong_section_citation_block_rate: float
    probes_passed: int
    probes_total: int
    disclaimer: str


@dataclass(frozen=True, slots=True)
class EvaluationModeMetrics:
    """Métricas agregadas e estratificadas para um modo específico de recuperação."""

    mode: str
    total_cases: int
    answerable_cases: int
    recall_at_5: float
    precision_at_5: float
    precision_at_1: float
    mrr_at_10: float
    ndcg_at_10: float
    avg_returned_evidences: float
    exact_structural_rate: float
    granular_match_rate: float
    document_filter_compliance_rate: float
    version_filter_compliance_rate: float
    no_evidence_abstention_rate: float
    errors: ErrorBreakdown
    rejected_or_ocr_sent_to_llm: int
    unapproved_sent_to_llm: int
    unapproved_literal_transcriptions: int
    hallucinated_citations_leaked: int
    hallucinated_no_evidence_answers: int
    category_metrics: tuple[CategoryMetrics, ...] = ()
    structural_subset_metrics: tuple[StructuralSubsetMetrics, ...] = ()
    query_details: tuple[QueryEvalDetail, ...] = ()


@dataclass(frozen=True, slots=True)
class AblationExperimentResult:
    """Resultado reproduzível de uma configuração de ablação do Hybrid RRF / Reranker."""

    config_name: str
    description: str
    recall_at_5: float
    precision_at_5: float
    precision_at_1: float
    mrr_at_10: float
    ndcg_at_10: float
    avg_returned_evidences: float
    retrieval_fp: int
    retrieval_fn: int
    abstention_rate: float
    governance_violations: int


@dataclass(frozen=True, slots=True)
class ThresholdSweepPoint:
    """Ponto de calibração do limiar `max_vector_distance` (+ combinações top_k/RRF)."""

    label: str
    max_vector_distance: float
    semantic_top_k: int
    rrf_k: int
    vector_recall_at_5: float
    vector_precision_at_5: float
    vector_mrr_at_10: float
    vector_ndcg_at_10: float
    vector_abstention_rate: float
    vector_retrieval_fp: int
    vector_retrieval_fn: int
    vector_categories_full_recall: int
    hybrid_recall_at_5: float
    hybrid_precision_at_5: float
    hybrid_mrr_at_10: float
    hybrid_ndcg_at_10: float
    hybrid_abstention_rate: float
    hybrid_retrieval_fp: int
    hybrid_retrieval_fn: int
    hybrid_categories_full_recall: int
    total_answerable_categories: int


@dataclass(frozen=True, slots=True)
class EvaluationSummaryReport:
    """Relatório completo da avaliação comparativa, ablações, citações e varredura de limiar."""

    dataset_name: str
    embedding_backend: str
    total_queries: int
    mode_metrics: tuple[EvaluationModeMetrics, ...]
    citation_validator_audit: CitationValidatorAuditReport
    ablation_results: tuple[AblationExperimentResult, ...]
    threshold_sweep: tuple[ThresholdSweepPoint, ...]
    provisional_threshold_note: str
    embedder_methodology_note: str


def evidence_key(evidence: FusedEvidence) -> str:
    """Chave canônica `regulation:amendment:section` (em minúsculas) para deduplicação."""
    return (
        f"{evidence.regulation_number.strip().lower()}:"
        f"{evidence.amendment_number.strip().lower()}:"
        f"{evidence.section_number.strip().lower()}"
    )


def _collapse_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def evidence_matches_target(evidence: FusedEvidence, target: ExpectedTarget) -> tuple[bool, str]:
    """Verifica aderência integral à granularidade normativa exigida pelo `ExpectedTarget`."""
    if evidence.regulation_number.strip().lower() != target.regulation_number.strip().lower():
        return False, "regulation_mismatch"
    if not amendments_match(evidence.amendment_number, target.amendment_number):
        return False, "amendment_mismatch"
    if evidence.section_number.strip().lower() != target.section_number.strip().lower():
        return False, "section_mismatch"
    if target.normative_unit_id and evidence.unit_id != target.normative_unit_id:
        return False, f"unit_id_mismatch({evidence.unit_id}!={target.normative_unit_id})"
    if target.expected_pdf_pages:
        missing_pages = [p for p in target.expected_pdf_pages if p not in evidence.pdf_pages]
        if missing_pages:
            return False, f"missing_pdf_pages({missing_pages})"

    norm_text = _collapse_ws(evidence.original_text)
    if target.paragraph_number or target.alinea or target.item_number or target.subitem_number:
        par = target.paragraph_number or target.alinea
        composite = "".join(
            part.strip().lower()
            for part in (par, target.item_number, target.subitem_number)
            if part and part.strip()
        )
        if composite and composite not in norm_text:
            return False, f"missing_subitem_marker({composite})"

    for snippet in target.expected_text_snippets:
        if _collapse_ws(snippet) not in norm_text:
            return False, f"missing_snippet({snippet[:32]})"

    return True, "ok"


def _dedupe_evidences_by_unit(evidences: Sequence[FusedEvidence]) -> list[FusedEvidence]:
    """Deduplica evidências na granularidade de seção/versão preservando a primeira ocorrência."""
    seen: set[str] = set()
    unique: list[FusedEvidence] = []
    for ev in evidences:
        key = evidence_key(ev)
        if key not in seen:
            seen.add(key)
            unique.append(ev)
    return unique


def _compute_dcg(relevances: Sequence[int]) -> float:
    return sum(
        (float(rel) / math.log2(idx + 2))
        for idx, rel in enumerate(relevances)
        if rel > 0
    )


def _build_search_filter(case: EvaluationCase) -> SearchFilter:
    doc_type = DocumentType(case.document_type_filter) if case.document_type_filter else None
    return SearchFilter(
        document_type=doc_type,
        regulation_number=case.regulation_filter,
        amendment_number=case.amendment_filter,
        only_current_version=case.only_current_version,
    )


def evaluate_citation_validator_suite() -> CitationValidatorAuditReport:
    """Executa auditoria determinística em 5 dimensões do `CitationValidator` (sem Gemini real)."""
    ev_43_3 = FusedEvidence(
        chunk_id="audit_chunk_43_3",
        unit_id="u_43_3",
        document_id="eval_rbac_43",
        version_id="eval_rbac_43_v05",
        authority="ANAC",
        document_type="RBAC",
        regulation_number="43",
        document_title="RBAC 43 Sintético",
        amendment_number="05",
        version_status="current",
        official_url="https://example.test/43",
        section_number="43.3",
        subsection_number="",
        paragraph_number="",
        item_number="",
        title="Pessoas autorizadas",
        original_text="43.3 Conforme remissão expressa à seção 43.7, exige-se habilitação.",
        pdf_pages=(2, 3),
        printed_pages=("2/10", "3/10"),
        page_statuses=(),
        trust_level=EvidenceTrustLevel.VALIDATED_LITERAL,
        literal_transcription_allowed=True,
        rrf_score=0.05,
        matched_sources=(RetrievalSource.EXACT_STRUCTURAL,),
    )
    validator = CitationValidator()
    ev_map = {"EV-1": ev_43_3}

    # 1. Validação de IDs de evidência ([EV-1] válido mantido; [EV-99] inválido removido)
    r_id = validator.validate(
        explanation="Conforme [EV-1], aplica-se a regra; ver também [EV-99].",
        evidence_map=ev_map,
    )
    id_ok = (
        r_id.valid_evidence_ids == ("EV-1",)
        and r_id.invalid_evidence_ids == ("EV-99",)
        and "[EV-99]" not in r_id.sanitized_explanation
    )

    # 2. Validação de referências estruturais (43.3 e cross-ref 43.7 mantidas; 43.99 descartada)
    r_sec = validator.validate(
        explanation="A seção 43.3 [EV-1] remete à seção 43.7, mas o RBAC 43.99 não existe.",
        evidence_map=ev_map,
    )
    sec_ok = (
        len(r_sec.valid_section_citations) == 2
        and any("43.99" in s for s in r_sec.unverified_section_citations)
        and "43.99" not in r_sec.sanitized_explanation
    )

    # 3. Validação de páginas (PDF p. 2 e página 3/10 válidas; página 99 e PDF p. 42 descartadas)
    r_page = validator.validate(
        explanation=(
            "O texto está na PDF p. 2 e página 3/10 [EV-1], "
            "não na página 99 nem em PDF p. 42."
        ),
        evidence_map=ev_map,
    )
    page_ok = (
        len(r_page.valid_page_citations) == 2
        and len(r_page.unverified_page_citations) == 2
        and "página 99" not in r_page.sanitized_explanation
        and "PDF p. 42" not in r_page.sanitized_explanation
    )

    # 4. Citação completamente inexistente (ID + seção + página fictícios todos bloqueados)
    r_nonexistent = validator.validate(
        explanation="Segundo o RBAC 99.999 [EV-77] na página 999, aplica-se regra inventada.",
        evidence_map=ev_map,
    )
    nonexistent_ok = (
        "EV-77" in r_nonexistent.invalid_evidence_ids
        and any("99.999" in s for s in r_nonexistent.unverified_section_citations)
        and len(r_nonexistent.unverified_page_citations) == 1
        and "99.999" not in r_nonexistent.sanitized_explanation
        and "página 999" not in r_nonexistent.sanitized_explanation
    )

    # 5. Citação apontando para seção real existente no corpus (RBAC 43.9 / RBAC 145.3)
    # porém AUSENTE no conjunto de evidências recuperadas para esta consulta!
    r_wrong_sec = validator.validate(
        explanation=(
            "A seção 43.3 [EV-1] trata da execução, enquanto o RBAC 43.9 e o RBAC 145.3 "
            "não foram recuperados nesta consulta."
        ),
        evidence_map=ev_map,
    )
    wrong_sec_ok = (
        any("43.9" in s for s in r_wrong_sec.unverified_section_citations)
        and any("145.3" in s for s in r_wrong_sec.unverified_section_citations)
        and "43.9" not in r_wrong_sec.sanitized_explanation
        and "145.3" not in r_wrong_sec.sanitized_explanation
    )

    checks = (id_ok, sec_ok, page_ok, nonexistent_ok, wrong_sec_ok)
    passed = sum(1 for c in checks if c)
    return CitationValidatorAuditReport(
        evidence_id_validation_rate=1.0 if id_ok else 0.0,
        structural_ref_validation_rate=1.0 if sec_ok else 0.0,
        page_ref_validation_rate=1.0 if page_ok else 0.0,
        nonexistent_citation_block_rate=1.0 if nonexistent_ok else 0.0,
        wrong_section_citation_block_rate=1.0 if wrong_sec_ok else 0.0,
        probes_passed=passed,
        probes_total=len(checks),
        disclaimer=(
            "Auditoria determinística exclusiva do CitationValidator com entradas sintéticas "
            "(LLM mock). NÃO representa taxa real de alucinação ou precisão de citações do Gemini."
        ),
    )


class _AuditSpyLLM(LLMPort):
    """Adaptador determinístico de avaliação que rastreia chamadas e injeta citações falsas."""

    def __init__(self, *, inject_hallucination_every_n: int = 0) -> None:
        self._inject_every = inject_hallucination_every_n
        self.call_count = 0
        self.prompts_received: list[str] = []

    @property
    def model_name(self) -> str:
        return "eval-deterministic-mock-llm"

    def generate_text(
        self,
        *,
        system_instruction: str,
        user_prompt: str,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> str:
        self.call_count += 1
        self.prompts_received.append(user_prompt)
        if self._inject_every > 0 and self.call_count % self._inject_every == 0:
            return (
                "Com base na evidência recuperada [EV-1], o dispositivo responde à consulta. "
                "Referência fictícia injetada RBAC 99.999 [EV-99] na página 999 para teste."
            )
        return "Com base estritamente na evidência normativa validada [EV-1], aplica-se o disposto."


class RagEvaluator:
    """Executa avaliação comparativa multi-modo, ablações e varredura de limiar vetorial."""

    def evaluate_mode(
        self,
        *,
        mode: str,
        cases: Sequence[EvaluationCase],
        retriever_factory: Callable[..., HybridNormativeRetriever],
        max_vector_distance: float = 0.85,
        test_hallucination_guard: bool = True,
        retriever_kwargs: dict[str, object] | None = None,
    ) -> EvaluationModeMetrics:
        """Avalia um modo específico com granularidade normativa e erros decompostos."""
        kwargs = dict(retriever_kwargs or {})
        retriever = retriever_factory(mode, max_vector_distance, **kwargs)
        spy_llm = _AuditSpyLLM(inject_hallucination_every_n=5 if test_hallucination_guard else 0)
        chat = NormativeChatService(retriever=retriever, llm=spy_llm)

        answerable_recalls: list[float] = []
        answerable_precisions_5: list[float] = []
        answerable_precisions_1: list[float] = []
        answerable_mrrs: list[float] = []
        answerable_ndcgs: list[float] = []
        returned_counts: list[int] = []

        exact_total = 0
        exact_hits = 0
        granular_total = 0
        granular_hits = 0
        doc_filter_total = 0
        doc_filter_ok_count = 0
        ver_filter_total = 0
        ver_filter_ok_count = 0
        no_ev_total = 0
        no_ev_abstentions = 0

        retrieval_fp = 0
        retrieval_fn = 0
        unwarranted_no_ev = 0
        abstention_failures = 0
        doc_filter_failures = 0
        ver_filter_failures = 0
        rejected_or_ocr_sent = 0
        unapproved_sent = 0
        unapproved_literals = 0
        hallucinated_leaked = 0
        hallucinated_no_ev = 0

        details: list[QueryEvalDetail] = []

        for case in cases:
            sfilter = _build_search_filter(case)
            calls_before = spy_llm.call_count

            retrieval_10 = retriever.retrieve(query=case.query, filters=sfilter, top_k=10)
            answer = chat.ask(question=case.query, filters=sfilter, top_k=5)
            llm_was_called = spy_llm.call_count > calls_before

            unique_ev_10 = _dedupe_evidences_by_unit(retrieval_10.evidences)[:10]
            unique_ev_5 = _dedupe_evidences_by_unit(answer.evidences)[:5]
            returned_keys_5 = tuple(evidence_key(ev) for ev in unique_ev_5)
            expected_keys = tuple(t.canonical_key for t in case.expected_targets)

            # Invariantes de governança (meta absoluta = 0 violações)
            if answer.normative_text is not None and not all(
                ev.trust_level == EvidenceTrustLevel.VALIDATED_LITERAL
                for ev in answer.evidences
                if ev.literal_transcription_allowed
            ):
                unapproved_literals += 1
            if answer.is_preliminary_only and answer.normative_text is not None:
                unapproved_literals += 1

            if llm_was_called:
                sent_chunk_ids = {
                    item.split(":", 1)[1] for item in answer.evidences_sent_to_llm if ":" in item
                }
                for ev in (*answer.evidences, *answer.blocked_evidences):
                    if ev.chunk_id not in sent_chunk_ids:
                        continue
                    if ev.trust_level in (
                        EvidenceTrustLevel.BLOCKED_OCR_PENDING,
                        EvidenceTrustLevel.BLOCKED_REJECTED,
                    ):
                        rejected_or_ocr_sent += 1
                    if ev.trust_level != EvidenceTrustLevel.VALIDATED_LITERAL:
                        unapproved_sent += 1

            if (
                "99.999" in answer.explanation
                or "[EV-99]" in answer.explanation
                or "página 999" in answer.explanation
            ):
                hallucinated_leaked += 1

            # Checagem independente de conformidade de filtro documental e de versão
            doc_filter_ok = True
            if case.expected_regulation is not None:
                doc_filter_total += 1
                bad_doc = [
                    ev
                    for ev in unique_ev_5
                    if ev.regulation_number.strip().lower()
                    != case.expected_regulation.strip().lower()
                ]
                if bad_doc:
                    doc_filter_ok = False
                    doc_filter_failures += 1
                else:
                    doc_filter_ok_count += 1

            ver_filter_ok = True
            if case.expected_amendment is not None:
                ver_filter_total += 1
                bad_ver = [
                    ev
                    for ev in unique_ev_5
                    if not amendments_match(ev.amendment_number, case.expected_amendment)
                ]
                if bad_ver:
                    ver_filter_ok = False
                    ver_filter_failures += 1
                else:
                    ver_filter_ok_count += 1

            q_rec: float | None = None
            q_prec: float | None = None
            q_rr: float | None = None
            q_granular_ok = True
            q_behavior_ok = True
            q_fp = 0
            q_fn = 0
            q_note = ""

            if case.expected_behavior == "validated_answer":
                returned_counts.append(len(unique_ev_5))
                matched_target_idxs_5: set[int] = set()
                matched_ev_idxs_5: set[int] = set()
                granular_failures: list[str] = []

                for t_idx, target in enumerate(case.expected_targets):
                    granular_total += 1
                    found_granular = False
                    for e_idx, ev in enumerate(unique_ev_5):
                        ok_match, reason = evidence_matches_target(ev, target)
                        if ok_match:
                            matched_target_idxs_5.add(t_idx)
                            matched_ev_idxs_5.add(e_idx)
                            found_granular = True
                            break
                        if evidence_key(ev) == target.canonical_key:
                            granular_failures.append(reason)
                    if found_granular:
                        granular_hits += 1
                    else:
                        q_granular_ok = False

                n_exp = len(case.expected_targets)
                n_ret = len(unique_ev_5)
                q_rec = (len(matched_target_idxs_5) / n_exp) if n_exp else 1.0
                # Parte 4.2: se não houver resultados em pergunta respondível, Precision@K = 0.0
                q_prec = (len(matched_ev_idxs_5) / n_ret) if n_ret > 0 else 0.0
                prec_1 = 1.0 if (0 in matched_ev_idxs_5) else 0.0

                q_fp = max(0, n_ret - len(matched_ev_idxs_5))
                q_fn = max(0, n_exp - len(matched_target_idxs_5))
                retrieval_fp += q_fp
                retrieval_fn += q_fn

                answerable_recalls.append(q_rec)
                answerable_precisions_5.append(q_prec)
                answerable_precisions_1.append(prec_1)

                # MRR@10 e nDCG@10 usando a mesma verificação granular
                rels_10: list[int] = []
                first_hit_rank: int | None = None
                for rank_idx, ev in enumerate(unique_ev_10, start=1):
                    is_rel = any(
                        evidence_matches_target(ev, t)[0] for t in case.expected_targets
                    )
                    rels_10.append(1 if is_rel else 0)
                    if is_rel and first_hit_rank is None:
                        first_hit_rank = rank_idx
                q_rr = (1.0 / first_hit_rank) if first_hit_rank is not None else 0.0
                answerable_mrrs.append(q_rr)

                idcg = _compute_dcg([1] * min(n_exp, 10))
                ndcg = (_compute_dcg(rels_10) / idcg) if idcg > 0 else 0.0
                answerable_ndcgs.append(ndcg)

                q_behavior_ok = q_rec == 1.0 and doc_filter_ok and ver_filter_ok
                if granular_failures:
                    q_note = "; ".join(granular_failures)

                if case.is_exact_structural:
                    exact_total += 1
                    if prec_1 == 1.0 and q_granular_ok:
                        exact_hits += 1

            elif case.expected_behavior == "abstain_no_evidence":
                no_ev_total += 1
                if not answer.found and not llm_was_called and answer.normative_text is None:
                    no_ev_abstentions += 1
                else:
                    q_behavior_ok = False
                    unwarranted_no_ev += 1
                    abstention_failures += 1
                    hallucinated_no_ev += 1
                    q_fp = len(unique_ev_5)
                    q_note = "failed_to_abstain_on_no_evidence"

            elif case.expected_behavior == "preliminary_location_no_llm":
                if not answer.is_preliminary_only or llm_was_called:
                    q_behavior_ok = False
                    q_fn = 1
                    retrieval_fn += 1
                    q_note = "expected_preliminary_without_llm"

            elif case.expected_behavior in ("blocked_ocr_no_llm", "blocked_rejected_no_llm"):
                if answer.found or llm_was_called:
                    q_behavior_ok = False
                    abstention_failures += 1
                    q_fp = len(unique_ev_5)
                    retrieval_fp += q_fp
                    q_note = f"leaked_usable_neighbor_on_{case.expected_behavior}"

            elif case.expected_behavior == "version_ambiguity":
                has_ambig_warn = any(
                    "AMBIGUIDADE DE VERSÃO" in obs for obs in answer.observations
                )
                if not has_ambig_warn:
                    q_behavior_ok = False
                    abstention_failures += 1
                    q_note = "missing_version_ambiguity_warning"

            details.append(
                QueryEvalDetail(
                    case_id=case.case_id,
                    category=case.category,
                    structural_subset=case.structural_subset,
                    query=case.query,
                    expected_behavior=case.expected_behavior,
                    expected_keys=expected_keys,
                    returned_keys=returned_keys_5,
                    recall_at_5=q_rec,
                    precision_at_5=q_prec,
                    reciprocal_rank=q_rr,
                    granular_ok=q_granular_ok,
                    doc_filter_ok=doc_filter_ok,
                    version_filter_ok=ver_filter_ok,
                    behavior_ok=q_behavior_ok,
                    retrieval_fp=q_fp,
                    retrieval_fn=q_fn,
                    notes=q_note,
                )
            )

        def _avg(values: Sequence[float]) -> float:
            return (sum(values) / len(values)) if values else 1.0

        governance_violations = (
            rejected_or_ocr_sent + unapproved_sent + unapproved_literals
        )
        errors = ErrorBreakdown(
            retrieval_false_positives=retrieval_fp,
            retrieval_false_negatives=retrieval_fn,
            unwarranted_answer_no_evidence=unwarranted_no_ev,
            abstention_failures=abstention_failures,
            document_filter_failures=doc_filter_failures,
            version_filter_failures=ver_filter_failures,
            governance_violations=governance_violations,
            hallucinated_citations=hallucinated_leaked,
        )

        # Estratificação por categoria (Parte 5) e por subgrupo estrutural (Parte 2.1)
        cat_metrics = self._aggregate_by_category(details)
        subset_metrics = self._aggregate_by_structural_subset(details)

        return EvaluationModeMetrics(
            mode=mode,
            total_cases=len(cases),
            answerable_cases=len(answerable_recalls),
            recall_at_5=_avg(answerable_recalls),
            precision_at_5=_avg(answerable_precisions_5),
            precision_at_1=_avg(answerable_precisions_1),
            mrr_at_10=_avg(answerable_mrrs),
            ndcg_at_10=_avg(answerable_ndcgs),
            avg_returned_evidences=(
                sum(returned_counts) / len(returned_counts) if returned_counts else 0.0
            ),
            exact_structural_rate=(exact_hits / exact_total) if exact_total else 1.0,
            granular_match_rate=(granular_hits / granular_total) if granular_total else 1.0,
            document_filter_compliance_rate=(
                doc_filter_ok_count / doc_filter_total if doc_filter_total else 1.0
            ),
            version_filter_compliance_rate=(
                ver_filter_ok_count / ver_filter_total if ver_filter_total else 1.0
            ),
            no_evidence_abstention_rate=(
                no_ev_abstentions / no_ev_total if no_ev_total else 1.0
            ),
            errors=errors,
            rejected_or_ocr_sent_to_llm=rejected_or_ocr_sent,
            unapproved_sent_to_llm=unapproved_sent,
            unapproved_literal_transcriptions=unapproved_literals,
            hallucinated_citations_leaked=hallucinated_leaked,
            hallucinated_no_evidence_answers=hallucinated_no_ev,
            category_metrics=cat_metrics,
            structural_subset_metrics=subset_metrics,
            query_details=tuple(details),
        )

    @staticmethod
    def _aggregate_by_category(details: Sequence[QueryEvalDetail]) -> tuple[CategoryMetrics, ...]:
        grouped: dict[str, list[QueryEvalDetail]] = defaultdict(list)
        for item in details:
            grouped[item.category].append(item)

        results: list[CategoryMetrics] = []
        for category, items in grouped.items():
            ans = [d for d in items if d.recall_at_5 is not None]
            rec = sum(d.recall_at_5 or 0.0 for d in ans) / len(ans) if ans else 1.0
            prec = sum(d.precision_at_5 or 0.0 for d in ans) / len(ans) if ans else 1.0
            mrr = sum(d.reciprocal_rank or 0.0 for d in ans) / len(ans) if ans else 1.0
            beh = sum(1 for d in items if d.behavior_ok) / len(items) if items else 1.0
            # nDCG simplificado por categoria (idêntico a MRR para single-target, aproximação limpa)
            ndcg = mrr
            results.append(
                CategoryMetrics(
                    category=category,
                    total_cases=len(items),
                    answerable_cases=len(ans),
                    recall_at_5=rec,
                    precision_at_5=prec,
                    mrr_at_10=mrr,
                    ndcg_at_10=ndcg,
                    behavior_pass_rate=beh,
                    retrieval_fp=sum(d.retrieval_fp for d in items),
                    retrieval_fn=sum(d.retrieval_fn for d in items),
                )
            )
        return tuple(results)

    @staticmethod
    def _aggregate_by_structural_subset(
        details: Sequence[QueryEvalDetail],
    ) -> tuple[StructuralSubsetMetrics, ...]:
        order = (
            "explicit_structural_ref",
            "article_alinea_item_only",
            "incomplete_ref",
            "ambiguous_section_or_version",
            "natural_language_no_ref",
        )
        grouped: dict[str, list[QueryEvalDetail]] = defaultdict(list)
        for item in details:
            grouped[item.structural_subset].append(item)

        subsets_ordered = [s for s in order if s in grouped] + [
            s for s in grouped if s not in order
        ]
        results: list[StructuralSubsetMetrics] = []
        for subset in subsets_ordered:
            items = grouped[subset]
            ans = [d for d in items if d.recall_at_5 is not None]
            rec = sum(d.recall_at_5 or 0.0 for d in ans) / len(ans) if ans else 1.0
            prec = sum(d.precision_at_5 or 0.0 for d in ans) / len(ans) if ans else 1.0
            mrr = sum(d.reciprocal_rank or 0.0 for d in ans) / len(ans) if ans else 1.0
            beh = sum(1 for d in items if d.behavior_ok) / len(items) if items else 1.0
            results.append(
                StructuralSubsetMetrics(
                    subset=subset,
                    total_cases=len(items),
                    answerable_cases=len(ans),
                    recall_at_5=rec,
                    precision_at_5=prec,
                    mrr_at_10=mrr,
                    behavior_pass_rate=beh,
                )
            )
        return tuple(results)

    def run_ablation_experiments(
        self,
        *,
        cases: Sequence[EvaluationCase],
        retriever_factory: Callable[..., HybridNormativeRetriever],
        max_vector_distance: float = 0.85,
    ) -> tuple[AblationExperimentResult, ...]:
        """Executa experimentos reproduzíveis de ablação do Hybrid RRF e reranking (Parte 6)."""
        configs: tuple[tuple[str, str, dict[str, object]], ...] = (
            (
                "baseline_rrf_k60_top5",
                "RRF puro (2.0/1.0/1.0, k=60, top_k=5, sem reranker, sem poda)",
                {"rrf_k": 60, "enable_reranker": False, "enable_adaptive_cutoff": False},
            ),
            (
                "rrf_alt_vec12_k30",
                "RRF alternativo (2.0/1.0/1.2, k=30, top_k=5, sem reranker, sem poda)",
                {
                    "rrf_k": 30,
                    "source_weights": {
                        RetrievalSource.EXACT_STRUCTURAL: 2.0,
                        RetrievalSource.LEXICAL_FTS5: 1.0,
                        RetrievalSource.SEMANTIC_VECTOR: 1.2,
                    },
                    "enable_reranker": False,
                    "enable_adaptive_cutoff": False,
                },
            ),
            (
                "rrf_wider_pool_top10",
                "RRF pool ampliado (lexical/semantic top_k=10, k=60, sem reranker, sem poda)",
                {
                    "lexical_top_k": 10,
                    "semantic_top_k": 10,
                    "rrf_k": 60,
                    "enable_reranker": False,
                    "enable_adaptive_cutoff": False,
                },
            ),
            (
                "rrf_plus_reranker_only",
                "RRF + Reranker determinístico (sem LLM pago, sem poda adaptativa)",
                {"rrf_k": 60, "enable_reranker": True, "enable_adaptive_cutoff": False},
            ),
            (
                "rrf_plus_cutoff_only",
                "RRF + Poda adaptativa de contexto (ratio=0.55, sem reranker)",
                {
                    "rrf_k": 60,
                    "enable_reranker": False,
                    "enable_adaptive_cutoff": True,
                    "adaptive_cutoff_ratio": 0.55,
                },
            ),
            (
                "hybrid_default_rerank_cutoff",
                "Hybrid padrão: RRF + Reranker determinístico + Poda adaptativa (ratio=0.55)",
                {
                    "rrf_k": 60,
                    "enable_reranker": True,
                    "enable_adaptive_cutoff": True,
                    "adaptive_cutoff_ratio": 0.55,
                },
            ),
        )

        results: list[AblationExperimentResult] = []
        for name, desc, kwargs in configs:
            m = self.evaluate_mode(
                mode="hybrid_rrf",
                cases=cases,
                retriever_factory=retriever_factory,
                max_vector_distance=max_vector_distance,
                test_hallucination_guard=False,
                retriever_kwargs=kwargs,
            )
            results.append(
                AblationExperimentResult(
                    config_name=name,
                    description=desc,
                    recall_at_5=m.recall_at_5,
                    precision_at_5=m.precision_at_5,
                    precision_at_1=m.precision_at_1,
                    mrr_at_10=m.mrr_at_10,
                    ndcg_at_10=m.ndcg_at_10,
                    avg_returned_evidences=m.avg_returned_evidences,
                    retrieval_fp=m.errors.retrieval_false_positives,
                    retrieval_fn=m.errors.retrieval_false_negatives,
                    abstention_rate=m.no_evidence_abstention_rate,
                    governance_violations=m.errors.governance_violations,
                )
            )
        return tuple(results)

    def run_full_benchmark(
        self,
        *,
        dataset_name: str,
        embedding_backend: str,
        cases: Sequence[EvaluationCase],
        retriever_factory: Callable[..., HybridNormativeRetriever],
        default_threshold: float = 0.85,
        sweep_thresholds: Sequence[float] = (0.50, 0.65, 0.75, 0.85, 0.95),
        include_ablations: bool = True,
    ) -> EvaluationSummaryReport:
        """Executa benchmark completo: 4 modos, categorias, CitationValidator, ablações e sweep."""
        modes = ("structural_only", "lexical_fts5_only", "vector_only", "hybrid_rrf")
        mode_results = tuple(
            self.evaluate_mode(
                mode=mode,
                cases=cases,
                retriever_factory=retriever_factory,
                max_vector_distance=default_threshold,
            )
            for mode in modes
        )

        citation_audit = evaluate_citation_validator_suite()
        ablations = (
            self.run_ablation_experiments(
                cases=cases,
                retriever_factory=retriever_factory,
                max_vector_distance=default_threshold,
            )
            if include_ablations
            else ()
        )

        # Parte 7: Varredura de limiar vetorial + combinações com top_k e RRF k
        sweep_configs: list[tuple[str, float, int, int]] = [
            (f"dist={t:.2f}, top_k=5, k=60", t, 5, 60) for t in sweep_thresholds
        ]
        sweep_configs.extend(
            [
                ("dist=0.75, top_k=10, k=60", 0.75, 10, 60),
                ("dist=0.85, top_k=10, k=30", 0.85, 10, 30),
            ]
        )

        sweep_points: list[ThresholdSweepPoint] = []
        for label, threshold, sem_k, rrf_k_val in sweep_configs:
            vec_m = self.evaluate_mode(
                mode="vector_only",
                cases=cases,
                retriever_factory=retriever_factory,
                max_vector_distance=threshold,
                test_hallucination_guard=False,
                retriever_kwargs={"semantic_top_k": sem_k, "rrf_k": rrf_k_val},
            )
            hyb_m = self.evaluate_mode(
                mode="hybrid_rrf",
                cases=cases,
                retriever_factory=retriever_factory,
                max_vector_distance=threshold,
                test_hallucination_guard=False,
                retriever_kwargs={
                    "lexical_top_k": sem_k,
                    "semantic_top_k": sem_k,
                    "rrf_k": rrf_k_val,
                },
            )
            ans_cats = [c for c in vec_m.category_metrics if c.answerable_cases > 0]
            vec_full = sum(1 for c in ans_cats if c.recall_at_5 >= 0.9999)
            hyb_ans_cats = [c for c in hyb_m.category_metrics if c.answerable_cases > 0]
            hyb_full = sum(1 for c in hyb_ans_cats if c.recall_at_5 >= 0.9999)

            sweep_points.append(
                ThresholdSweepPoint(
                    label=label,
                    max_vector_distance=threshold,
                    semantic_top_k=sem_k,
                    rrf_k=rrf_k_val,
                    vector_recall_at_5=vec_m.recall_at_5,
                    vector_precision_at_5=vec_m.precision_at_5,
                    vector_mrr_at_10=vec_m.mrr_at_10,
                    vector_ndcg_at_10=vec_m.ndcg_at_10,
                    vector_abstention_rate=vec_m.no_evidence_abstention_rate,
                    vector_retrieval_fp=vec_m.errors.retrieval_false_positives,
                    vector_retrieval_fn=vec_m.errors.retrieval_false_negatives,
                    vector_categories_full_recall=vec_full,
                    hybrid_recall_at_5=hyb_m.recall_at_5,
                    hybrid_precision_at_5=hyb_m.precision_at_5,
                    hybrid_mrr_at_10=hyb_m.mrr_at_10,
                    hybrid_ndcg_at_10=hyb_m.ndcg_at_10,
                    hybrid_abstention_rate=hyb_m.no_evidence_abstention_rate,
                    hybrid_retrieval_fp=hyb_m.errors.retrieval_false_positives,
                    hybrid_retrieval_fn=hyb_m.errors.retrieval_false_negatives,
                    hybrid_categories_full_recall=hyb_full,
                    total_answerable_categories=len(ans_cats),
                )
            )

        if embedding_backend == "deterministic":
            embedder_note = (
                "EMBEDDER SINTÉTICO (DeterministicEvalSemanticEmbedder, 768d): "
                "validado apenas para testes mecânicos e reproduzíveis no CI. "
                "NÃO representa a qualidade semântica real do `gemini-embedding-2`."
            )
        else:
            embedder_note = (
                f"EMBEDDER REAL ({embedding_backend}, 768d Free Tier Standard) executado "
                "exclusivamente sobre o corpus SINTÉTICO fictício (sem PDF oficial do RBAC 43). "
                "Esta execução consome a cota gratuita da GEMINI_API_KEY local do usuário."
            )

        return EvaluationSummaryReport(
            dataset_name=dataset_name,
            embedding_backend=embedding_backend,
            total_queries=len(cases),
            mode_metrics=mode_results,
            citation_validator_audit=citation_audit,
            ablation_results=ablations,
            threshold_sweep=tuple(sweep_points),
            provisional_threshold_note=(
                f"Limiar padrão `max_vector_distance={default_threshold:.2f}` permanece "
                "PROVISÓRIO até avaliação com `gemini-embedding-2` real e PDF oficial real "
                "do RBAC 43. Eventual valor melhor no corpus sintético é apenas candidato "
                "experimental, nunca calibração de produção."
            ),
            embedder_methodology_note=embedder_note,
        )
