"""Gate automatizado de avaliação quantitativa controlada do RAG (corpus 100% sintético).

Verifica todos os critérios de aceitação definidos para o ambiente sintético controlado:
- >= 30 consultas controladas abrangendo múltiplos documentos e categorias;
- 100% de acerto estrutural exato;
- 100% de respeito a filtros de regulamento e versão/emenda;
- 0 páginas reprovadas/OCR/não aprovadas enviadas ao LLM;
- 0 citações alucinadas vazadas após o CitationValidator;
- 0 transcrições literais de páginas não aprovadas;
- 0 respostas inventadas em consultas fora de domínio / sem evidência;
- Recall@5 do modo híbrido RRF >= 90% e superior a cada modalidade isolada.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts import evaluate_rag as eval_cli

from rbac_assistant.rag.evaluation.rag_evaluator import RagEvaluator
from tests.evals.synthetic_rag_eval_fixture import (
    build_synthetic_eval_environment,
    load_eval_dataset,
)


@pytest.mark.integration
def test_synthetic_rag_evaluation_passes_all_quality_and_governance_gates(
    tmp_path: Path,
) -> None:
    cases = load_eval_dataset()
    assert len(cases) >= 30

    _, _, _, factory = build_synthetic_eval_environment(tmp_path)
    evaluator = RagEvaluator()
    report = evaluator.run_full_benchmark(
        dataset_name="rbac_synthetic_controlled_eval_v1",
        cases=cases,
        retriever_factory=factory,
        default_threshold=0.85,
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

    # Critérios obrigatórios de aceitação sintética
    assert hybrid.exact_structural_rate == 1.0
    assert hybrid.reg_version_filter_rate == 1.0
    assert hybrid.rejected_or_ocr_sent_to_llm == 0
    assert hybrid.unapproved_sent_to_llm == 0
    assert hybrid.unapproved_literal_transcriptions == 0
    assert hybrid.hallucinated_citations_leaked == 0
    assert hybrid.hallucinated_no_evidence_answers == 0
    assert hybrid.no_evidence_abstention_rate == 1.0
    assert hybrid.citation_precision == 1.0
    assert hybrid.recall_at_5 >= 0.90

    # Híbrido RRF deve superar ou igualar cada modalidade isolada em Recall@5
    # e superar em precisão e qualidade de ordenação (Precision@5, MRR@10, nDCG@10)
    assert hybrid.recall_at_5 >= structural.recall_at_5
    assert hybrid.recall_at_5 >= lexical.recall_at_5
    assert hybrid.recall_at_5 >= vector.recall_at_5
    assert hybrid.recall_at_5 > structural.recall_at_5
    assert hybrid.recall_at_5 > vector.recall_at_5
    assert hybrid.precision_at_5 > lexical.precision_at_5
    assert hybrid.mrr_at_10 > lexical.mrr_at_10
    assert hybrid.ndcg_at_10 > lexical.ndcg_at_10

    # Varredura de limiar vetorial presente e rotulada como provisória
    assert len(report.threshold_sweep) >= 4
    assert "PROVISÓRIO" in report.provisional_threshold_note


@pytest.mark.integration
def test_evaluate_rag_cli_outputs_valid_json_and_table(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc_table = eval_cli.main([])
    assert rc_table == 0
    out_table = capsys.readouterr().out
    assert "AVALIAÇÃO QUANTITATIVA CONTROLADA DO RAG" in out_table
    assert "hybrid_rrf" in out_table
    assert "PROVISÓRIO" in out_table

    rc_json = eval_cli.main(["--json"])
    assert rc_json == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["total_queries"] >= 30
    assert len(payload["mode_metrics"]) == 4
    assert len(payload["threshold_sweep"]) >= 4
