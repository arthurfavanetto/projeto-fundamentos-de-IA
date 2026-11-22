"""Gate automatizado de avaliação quantitativa controlada do RAG (corpus 100% sintético).

Verifica critérios metodológicos, de recuperação, granularidade normativa e governança:
- >= 40 consultas sintéticas controladas (todas as 36 originais + 4 casos adicionais);
- Cobertura das 19 categorias exigidas e dos 5 subgrupos estruturais;
- Granularidade normativa validada (regulamento, emenda, seção, parágrafo/item/subitem,
  `normative_unit_id`, páginas físicas e evidências textuais);
- Separação transparente de erros (`ErrorBreakdown`) e 0 violações de governança em todos os modos;
- Auditoria dedicada em 5 dimensões do `CitationValidator` (sem mascarar mock como Gemini real);
- Ablação reproduzível demonstrando ganho real do reranker determinístico e poda adaptativa;
- Varredura de limiar vetorial mantendo `max_vector_distance=0.85` marcado como PROVISÓRIO;
- CLI `--embedding-backend gemini-embedding-2` falha com erro seguro sem GEMINI_API_KEY no CI.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from scripts import evaluate_rag as eval_cli

from rbac_assistant.rag.evaluation.rag_evaluator import (
    RagEvaluator,
    evaluate_citation_validator_suite,
)
from tests.evals.synthetic_rag_eval_fixture import (
    build_synthetic_eval_environment,
    load_eval_dataset,
)


@pytest.mark.integration
def test_synthetic_rag_evaluation_passes_all_quality_granularity_and_governance_gates(
    tmp_path: Path,
) -> None:
    cases = load_eval_dataset()
    assert len(cases) >= 40

    started_at = time.perf_counter()
    repo, _, _, factory = build_synthetic_eval_environment(tmp_path)
    try:
        evaluator = RagEvaluator()
        report = evaluator.run_full_benchmark(
            dataset_name="rbac_synthetic_controlled_eval_v2",
            embedding_backend="deterministic",
            cases=cases,
            retriever_factory=factory,
            default_threshold=0.85,
            include_ablations=True,
        )
    finally:
        repo.close()
    elapsed_seconds = time.perf_counter() - started_at

    # Regressão de performance: benchmark completo offline não deve travar nem levar minutos
    # (limite conservador de 30s para tolerar overhead de pytest-cov em CI/Windows)
    assert elapsed_seconds < 30.0, (
        f"Avaliação sintética completa demorou {elapsed_seconds:.2f}s (limite: 30.0s)."
    )

    by_mode = {m.mode: m for m in report.mode_metrics}
    assert set(by_mode.keys()) == {
        "structural_only",
        "lexical_fts5_only",
        "vector_only",
        "hybrid_rrf",
    }

    hybrid = by_mode["hybrid_rrf"]
    structural = by_mode["structural_only"]
    lexical = by_mode["lexical_fts5_only"]
    vector = by_mode["vector_only"]

    # 1. Governança estrita: 0 violações em TODOS os 4 modos!
    for mode_metrics in report.mode_metrics:
        assert mode_metrics.errors.governance_violations == 0
        assert mode_metrics.rejected_or_ocr_sent_to_llm == 0
        assert mode_metrics.unapproved_sent_to_llm == 0
        assert mode_metrics.unapproved_literal_transcriptions == 0
        assert mode_metrics.hallucinated_citations_leaked == 0
        assert mode_metrics.document_filter_compliance_rate == 1.0
        assert mode_metrics.version_filter_compliance_rate == 1.0

    # 2. Qualidade e granularidade do modo híbrido
    assert hybrid.exact_structural_rate == 1.0
    assert hybrid.granular_match_rate == 1.0
    assert hybrid.no_evidence_abstention_rate == 1.0
    assert hybrid.hallucinated_no_evidence_answers == 0
    assert hybrid.errors.retrieval_false_negatives == 0
    assert hybrid.recall_at_5 >= 0.90
    assert hybrid.precision_at_5 >= 0.80
    assert hybrid.mrr_at_10 == 1.0

    # 3. Modo híbrido supera ou iguala modalidades isoladas em Recall@5 e supera em Precision/MRR
    assert hybrid.recall_at_5 > structural.recall_at_5
    assert hybrid.recall_at_5 > vector.recall_at_5
    assert hybrid.recall_at_5 >= lexical.recall_at_5
    assert hybrid.precision_at_5 > lexical.precision_at_5
    assert hybrid.precision_at_5 > vector.precision_at_5
    assert hybrid.mrr_at_10 > lexical.mrr_at_10
    assert hybrid.ndcg_at_10 > lexical.ndcg_at_10

    # 4. Structural_only avaliado justamente nos 5 subgrupos (100% nas refs explícitas)
    struct_by_sub = {s.subset: s for s in structural.structural_subset_metrics}
    assert struct_by_sub["explicit_structural_ref"].recall_at_5 == 1.0
    assert struct_by_sub["explicit_structural_ref"].precision_at_5 == 1.0
    assert struct_by_sub["article_alinea_item_only"].recall_at_5 == 1.0
    assert struct_by_sub["ambiguous_section_or_version"].behavior_pass_rate == 1.0
    assert struct_by_sub["natural_language_no_ref"].recall_at_5 == 0.0

    # 5. Cobertura das 19 categorias
    assert len(hybrid.category_metrics) >= 19

    # 6. Auditoria em 5 dimensões do CitationValidator (claramente rotulada como mock)
    ca = report.citation_validator_audit
    assert ca.probes_passed == 5
    assert ca.probes_total == 5
    assert ca.page_ref_validation_rate == 1.0
    assert ca.wrong_section_citation_block_rate == 1.0
    assert "NÃO representa" in ca.disclaimer

    # 7. Ablação reproduzível confirma ganho real do reranker + poda adaptativa
    ab_by = {a.config_name: a for a in report.ablation_results}
    baseline = ab_by["baseline_rrf_k60_top5"]
    final_cfg = ab_by["hybrid_default_rerank_cutoff"]
    assert final_cfg.recall_at_5 == baseline.recall_at_5 == 1.0
    assert final_cfg.precision_at_5 > baseline.precision_at_5
    assert final_cfg.mrr_at_10 >= baseline.mrr_at_10
    assert final_cfg.retrieval_fp < baseline.retrieval_fp

    # 8. Varredura de limiar vetorial + notas metodológicas obrigatórias
    assert len(report.threshold_sweep) >= 5
    assert "PROVISÓRIO" in report.provisional_threshold_note
    assert "NÃO representa" in report.embedder_methodology_note


@pytest.mark.unit
def test_citation_validator_five_dimension_suite_passes_standalone() -> None:
    audit = evaluate_citation_validator_suite()
    assert audit.probes_passed == 5
    assert audit.evidence_id_validation_rate == 1.0
    assert audit.structural_ref_validation_rate == 1.0
    assert audit.page_ref_validation_rate == 1.0
    assert audit.nonexistent_citation_block_rate == 1.0
    assert audit.wrong_section_citation_block_rate == 1.0


@pytest.mark.integration
def test_evaluate_rag_cli_outputs_valid_json_table_and_blocks_real_backend_without_key(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    rc_table = eval_cli.main(["--per-query"])
    assert rc_table == 0
    out_table = capsys.readouterr().out
    assert "AVALIAÇÃO QUANTITATIVA CONTROLADA DO RAG" in out_table
    assert "Structural Only — Métricas Separadas por Subgrupo" in out_table
    assert "Auditoria Determinística do CitationValidator" in out_table
    assert "Experimento de Ablação" in out_table
    assert "Resultados Individuais por Consulta" in out_table
    assert "PROVISÓRIO" in out_table

    rc_json = eval_cli.main(["--json"])
    assert rc_json == 0
    captured_json = capsys.readouterr()
    payload = json.loads(captured_json.out)
    assert payload["total_queries"] >= 40
    assert payload["embedding_backend"] == "deterministic"
    assert len(payload["mode_metrics"]) == 4
    assert len(payload["ablation_results"]) >= 6
    assert len(payload["threshold_sweep"]) >= 5

    # Mesmo que exista um `.env` local no host contendo GEMINI_API_KEY (ex.: Windows local),
    # o isolamento da suíte impede vazamento de credencial e chamadas reais à API Gemini.
    (tmp_path / ".env").write_text("GEMINI_API_KEY=AIzaFakeShouldNeverBeReadInTests\n", "utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        eval_cli.main(["--embedding-backend", "gemini-embedding-2"])
