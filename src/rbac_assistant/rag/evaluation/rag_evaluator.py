"""Avaliador quantitativo determinístico do motor RAG híbrido (Recall@5, Precision@5, MRR, nDCG).

ATENÇÃO METODOLÓGICA:
- Quando executado sobre o dataset sintético controlado, os resultados validam a
  corretude da lógica de recuperação, filtros, governança de revisão e validação de citações,
  mas NÃO substituem a calibração posterior sobre o PDF oficial real do RBAC 43.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from rbac_assistant.application.services.chat_service import NormativeChatService
from rbac_assistant.domain.entities.rag import (
    EvidenceTrustLevel,
    FusedEvidence,
    SearchFilter,
)
from rbac_assistant.domain.entities.regulatory_document import DocumentType
from rbac_assistant.domain.ports.rag_ports import LLMPort
from rbac_assistant.rag.retrieval.hybrid_retriever import HybridNormativeRetriever


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    """Caso individual de avaliação controlada do pipeline RAG."""

    case_id: str
    category: str
    query: str
    expected_keys: tuple[str, ...]
    expected_behavior: str
    is_exact_structural: bool = False
    has_explicit_reg_or_version: bool = False
    regulation_filter: str | None = None
    amendment_filter: str | None = None
    document_type_filter: str | None = None
    only_current_version: bool = True
    expected_pdf_pages: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class EvaluationModeMetrics:
    """Métricas agregadas para um modo específico de recuperação."""

    mode: str
    total_cases: int
    answerable_cases: int
    recall_at_5: float
    precision_at_5: float
    mrr_at_10: float
    ndcg_at_10: float
    exact_structural_rate: float
    reg_version_filter_rate: float
    no_evidence_abstention_rate: float
    citation_precision: float
    false_positives: int
    false_negatives: int
    rejected_or_ocr_sent_to_llm: int
    unapproved_sent_to_llm: int
    unapproved_literal_transcriptions: int
    hallucinated_citations_leaked: int
    hallucinated_no_evidence_answers: int


@dataclass(frozen=True, slots=True)
class ThresholdSweepPoint:
    """Ponto de calibração do limiar `max_vector_distance` sobre o modo híbrido e vetorial."""

    max_vector_distance: float
    vector_recall_at_5: float
    vector_precision_at_5: float
    vector_abstention_rate: float
    hybrid_recall_at_5: float
    hybrid_precision_at_5: float
    hybrid_abstention_rate: float
    hybrid_false_positives: int


@dataclass(frozen=True, slots=True)
class EvaluationSummaryReport:
    """Relatório completo da avaliação comparativa e calibração provisória de limiar."""

    dataset_name: str
    total_queries: int
    mode_metrics: tuple[EvaluationModeMetrics, ...]
    threshold_sweep: tuple[ThresholdSweepPoint, ...]
    provisional_threshold_note: str


def evidence_key(evidence: FusedEvidence) -> str:
    """Chave canônica `regulation:amendment:section` (em minúsculas) para comparação exata."""
    return (
        f"{evidence.regulation_number.strip().lower()}:"
        f"{evidence.amendment_number.strip().lower()}:"
        f"{evidence.section_number.strip().lower()}"
    )


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


class _AuditSpyLLM(LLMPort):
    """Adaptador determinístico de avaliação que rastreia cada chamada e testa citações."""

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
                "Referência fictícia injetada RBAC 99.999 [EV-99] para teste do validador."
            )
        return "Com base estritamente na evidência normativa validada [EV-1], aplica-se o disposto."


class RagEvaluator:
    """Executa avaliação comparativa multi-modo e varredura de limiar de distância vetorial."""

    def evaluate_mode(
        self,
        *,
        mode: str,
        cases: Sequence[EvaluationCase],
        retriever_factory: Callable[[str, float], HybridNormativeRetriever],
        max_vector_distance: float = 0.85,
        test_hallucination_guard: bool = True,
    ) -> EvaluationModeMetrics:
        """Avalia um modo específico (`structural_only`, `lexical_fts5_only`, `vector_only`,
        `hybrid_rrf`) sobre todos os casos controlados."""
        retriever = retriever_factory(mode, max_vector_distance)
        spy_llm = _AuditSpyLLM(inject_hallucination_every_n=5 if test_hallucination_guard else 0)
        chat = NormativeChatService(retriever=retriever, llm=spy_llm)

        answerable_recalls: list[float] = []
        answerable_precisions: list[float] = []
        answerable_mrrs: list[float] = []
        answerable_ndcgs: list[float] = []

        exact_total = 0
        exact_hits = 0
        filter_total = 0
        filter_hits = 0
        no_ev_total = 0
        no_ev_abstentions = 0
        citation_precisions: list[float] = []

        false_positives = 0
        false_negatives = 0
        rejected_or_ocr_sent = 0
        unapproved_sent = 0
        unapproved_literals = 0
        hallucinated_leaked = 0
        hallucinated_no_ev = 0

        for case in cases:
            expected_set = {k.strip().lower() for k in case.expected_keys}
            sfilter = _build_search_filter(case)
            calls_before = spy_llm.call_count

            # Avalia recuperação em top-10 para MRR@10 / nDCG@10 e resposta completa em top-5
            retrieval_10 = retriever.retrieve(query=case.query, filters=sfilter, top_k=10)
            answer = chat.ask(question=case.query, filters=sfilter, top_k=5)
            llm_was_called = spy_llm.call_count > calls_before

            ranked_keys_10 = [evidence_key(ev) for ev in retrieval_10.evidences]
            ranked_keys_5 = [evidence_key(ev) for ev in answer.evidences[:5]]

            # Invariantes de governança (devem ser sempre 0!)
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
                all_seen = [*answer.evidences, *answer.blocked_evidences]
                for ev in all_seen:
                    if ev.chunk_id not in sent_chunk_ids:
                        continue
                    if ev.trust_level in (
                        EvidenceTrustLevel.BLOCKED_OCR_PENDING,
                        EvidenceTrustLevel.BLOCKED_REJECTED,
                    ):
                        rejected_or_ocr_sent += 1
                    if ev.trust_level != EvidenceTrustLevel.VALIDATED_LITERAL:
                        unapproved_sent += 1

            # Verifica se alguma citação alucinada injetada vazou na explicação final
            if "99.999" in answer.explanation or "[EV-99]" in answer.explanation:
                hallucinated_leaked += 1

            # Verifica precisão de citações nas respostas geradas pelo LLM
            if llm_was_called and not answer.used_fallback_without_llm:
                valid_cnt = len(answer.used_evidence_ids)
                leaked_cnt = 1 if ("99.999" in answer.explanation) else 0
                total_final = valid_cnt + leaked_cnt
                citation_precisions.append(
                    (valid_cnt / total_final) if total_final > 0 else 1.0
                )

            # Contabiliza comportamento esperado por categoria
            if case.expected_behavior == "validated_answer":
                top5_set = set(ranked_keys_5)
                matched_5 = len(expected_set & top5_set)
                recall_5 = matched_5 / len(expected_set) if expected_set else 1.0
                precision_5 = matched_5 / len(ranked_keys_5) if ranked_keys_5 else 0.0
                answerable_recalls.append(recall_5)
                answerable_precisions.append(precision_5)

                if matched_5 == 0:
                    false_negatives += 1

                # MRR@10
                rr = 0.0
                for idx, key in enumerate(ranked_keys_10[:10], start=1):
                    if key in expected_set:
                        rr = 1.0 / idx
                        break
                answerable_mrrs.append(rr)

                # nDCG@10
                rels = [1 if key in expected_set else 0 for key in ranked_keys_10[:10]]
                ideal_rels = [1] * min(len(expected_set), 10)
                idcg = _compute_dcg(ideal_rels)
                ndcg = (_compute_dcg(rels) / idcg) if idcg > 0 else 0.0
                answerable_ndcgs.append(ndcg)

                if case.is_exact_structural:
                    exact_total += 1
                    if ranked_keys_5 and ranked_keys_5[0] in expected_set:
                        # Também confere páginas multi-página quando especificadas
                        pages_ok = True
                        if case.expected_pdf_pages:
                            pages_ok = all(
                                p in answer.evidences[0].pdf_pages
                                for p in case.expected_pdf_pages
                            )
                        if pages_ok:
                            exact_hits += 1

                if case.has_explicit_reg_or_version:
                    filter_total += 1
                    exp_reg_amds = {":".join(k.split(":")[:2]) for k in expected_set}
                    if (
                        ranked_keys_5
                        and bool(expected_set & set(ranked_keys_5))
                        and all(
                            ":".join(k.split(":")[:2]) in exp_reg_amds for k in ranked_keys_5
                        )
                    ):
                        filter_hits += 1

            elif case.expected_behavior == "abstain_no_evidence":
                no_ev_total += 1
                if not answer.found and not llm_was_called and answer.normative_text is None:
                    no_ev_abstentions += 1
                else:
                    false_positives += 1
                    hallucinated_no_ev += 1

            elif case.expected_behavior == "preliminary_location_no_llm":
                if llm_was_called:
                    unapproved_sent += 1
                if answer.normative_text is not None:
                    unapproved_literals += 1
                if not answer.is_preliminary_only:
                    false_negatives += 1

            elif case.expected_behavior in ("blocked_ocr_no_llm", "blocked_rejected_no_llm"):
                if llm_was_called:
                    rejected_or_ocr_sent += 1
                if answer.normative_text is not None:
                    unapproved_literals += 1
                if answer.found:
                    false_positives += 1

            elif case.expected_behavior == "version_ambiguity":
                has_ambig_warn = any(
                    "AMBIGUIDADE DE VERSÃO" in obs for obs in answer.observations
                )
                if not has_ambig_warn:
                    false_negatives += 1

        def _avg(values: Sequence[float]) -> float:
            return (sum(values) / len(values)) if values else 1.0

        return EvaluationModeMetrics(
            mode=mode,
            total_cases=len(cases),
            answerable_cases=len(answerable_recalls),
            recall_at_5=_avg(answerable_recalls),
            precision_at_5=_avg(answerable_precisions),
            mrr_at_10=_avg(answerable_mrrs),
            ndcg_at_10=_avg(answerable_ndcgs),
            exact_structural_rate=(exact_hits / exact_total) if exact_total else 1.0,
            reg_version_filter_rate=(filter_hits / filter_total) if filter_total else 1.0,
            no_evidence_abstention_rate=(
                no_ev_abstentions / no_ev_total if no_ev_total else 1.0
            ),
            citation_precision=_avg(citation_precisions),
            false_positives=false_positives,
            false_negatives=false_negatives,
            rejected_or_ocr_sent_to_llm=rejected_or_ocr_sent,
            unapproved_sent_to_llm=unapproved_sent,
            unapproved_literal_transcriptions=unapproved_literals,
            hallucinated_citations_leaked=hallucinated_leaked,
            hallucinated_no_evidence_answers=hallucinated_no_ev,
        )

    def run_full_benchmark(
        self,
        *,
        dataset_name: str,
        cases: Sequence[EvaluationCase],
        retriever_factory: Callable[[str, float], HybridNormativeRetriever],
        default_threshold: float = 0.85,
        sweep_thresholds: Sequence[float] = (0.50, 0.65, 0.75, 0.85, 0.95),
    ) -> EvaluationSummaryReport:
        """Executa comparação entre 4 modos e varredura provisória de limiar vetorial."""
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

        sweep_points: list[ThresholdSweepPoint] = []
        for threshold in sweep_thresholds:
            vec_m = self.evaluate_mode(
                mode="vector_only",
                cases=cases,
                retriever_factory=retriever_factory,
                max_vector_distance=threshold,
                test_hallucination_guard=False,
            )
            hyb_m = self.evaluate_mode(
                mode="hybrid_rrf",
                cases=cases,
                retriever_factory=retriever_factory,
                max_vector_distance=threshold,
                test_hallucination_guard=False,
            )
            sweep_points.append(
                ThresholdSweepPoint(
                    max_vector_distance=threshold,
                    vector_recall_at_5=vec_m.recall_at_5,
                    vector_precision_at_5=vec_m.precision_at_5,
                    vector_abstention_rate=vec_m.no_evidence_abstention_rate,
                    hybrid_recall_at_5=hyb_m.recall_at_5,
                    hybrid_precision_at_5=hyb_m.precision_at_5,
                    hybrid_abstention_rate=hyb_m.no_evidence_abstention_rate,
                    hybrid_false_positives=hyb_m.false_positives,
                )
            )

        return EvaluationSummaryReport(
            dataset_name=dataset_name,
            total_queries=len(cases),
            mode_metrics=mode_results,
            threshold_sweep=tuple(sweep_points),
            provisional_threshold_note=(
                f"Limiar atual `max_vector_distance={default_threshold:.2f}` é PROVISÓRIO "
                "(calibrado apenas em dataset sintético com embedder determinístico local; "
                "exige recalibração após ingestão real do RBAC 43 e embeddings reais "
                "gemini-embedding-2)."
            ),
        )
