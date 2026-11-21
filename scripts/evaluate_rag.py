"""CLI operacional para avaliação quantitativa controlada do motor RAG híbrido.

Executa 36 consultas sintéticas controladas (4 documentos fictícios com numeração sobreposta,
paráfrases, sinônimos, ambiguidade de versão e estados de revisão), comparando:
- Busca estrutural isolada (`structural_only`)
- Busca lexical FTS5 isolada (`lexical_fts5_only`)
- Busca vetorial isolada (`vector_only`)
- Busca híbrida RRF (`hybrid_rrf`)

Não realiza chamadas externas pagas à API Gemini e não lê PDFs reais da ANAC.

Exemplos:
    uv run python scripts/evaluate_rag.py
    uv run python scripts/evaluate_rag.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

from rbac_assistant.rag.evaluation.rag_evaluator import (
    EvaluationSummaryReport,
    RagEvaluator,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests.evals.synthetic_rag_eval_fixture import (  # noqa: E402
    build_synthetic_eval_environment,
    load_eval_dataset,
)


def _format_pct(val: float) -> str:
    return f"{val * 100:.1f}%"


def render_report_table(report: EvaluationSummaryReport) -> str:
    """Formata relatório comparativo legível para terminal e auditoria."""
    lines: list[str] = [
        f"=== AVALIAÇÃO QUANTITATIVA CONTROLADA DO RAG ({report.dataset_name}) ===",
        f"Total de consultas controladas: {report.total_queries} (100% corpus sintético)",
        "",
        "1) Comparação por Modo de Recuperação (limiar provisório max_vector_distance=0.85):",
        (
            f"{'Modo':<20} {'Rec@5':>7} {'Prec@5':>7} {'MRR@10':>7} {'nDCG@10':>8} "
            f"{'Estrut.':>8} {'Filtro':>7} {'Abst.':>7} {'CitPr':>7} {'FP':>4} {'FN':>4}"
        ),
        "-" * 96,
    ]
    for m in report.mode_metrics:
        lines.append(
            f"{m.mode:<20} {_format_pct(m.recall_at_5):>7} {_format_pct(m.precision_at_5):>7} "
            f"{m.mrr_at_10:>7.3f} {m.ndcg_at_10:>8.3f} "
            f"{_format_pct(m.exact_structural_rate):>8} "
            f"{_format_pct(m.reg_version_filter_rate):>7} "
            f"{_format_pct(m.no_evidence_abstention_rate):>7} "
            f"{_format_pct(m.citation_precision):>7} "
            f"{m.false_positives:>4} {m.false_negatives:>4}"
        )

    hybrid = next(m for m in report.mode_metrics if m.mode == "hybrid_rrf")
    lines.extend(
        [
            "",
            "2) Invariantes de Governança no Modo Híbrido (meta = 0 violações):",
            f"  - Páginas OCR pendentes/reprovadas enviadas ao LLM: "
            f"{hybrid.rejected_or_ocr_sent_to_llm}",
            f"  - Páginas não aprovadas (incl. preliminares) enviadas ao LLM: "
            f"{hybrid.unapproved_sent_to_llm}",
            f"  - Transcrições literais de páginas não aprovadas: "
            f"{hybrid.unapproved_literal_transcriptions}",
            f"  - Citações alucinadas vazadas após CitationValidator: "
            f"{hybrid.hallucinated_citations_leaked}",
            f"  - Respostas inventadas em consultas sem evidência: "
            f"{hybrid.hallucinated_no_evidence_answers}",
            "",
            "3) Varredura de Limiar Vetorial (`max_vector_distance`) — PROVISÓRIO:",
            (
                f"{'Limiar':>8} {'Vec Rec@5':>10} {'Vec Prec@5':>11} {'Vec Abst':>9} "
                f"{'Hyb Rec@5':>10} {'Hyb Prec@5':>11} {'Hyb Abst':>9} {'Hyb FP':>7}"
            ),
            "-" * 82,
        ]
    )
    for pt in report.threshold_sweep:
        lines.append(
            f"{pt.max_vector_distance:>8.2f} "
            f"{_format_pct(pt.vector_recall_at_5):>10} "
            f"{_format_pct(pt.vector_precision_at_5):>11} "
            f"{_format_pct(pt.vector_abstention_rate):>9} "
            f"{_format_pct(pt.hybrid_recall_at_5):>10} "
            f"{_format_pct(pt.hybrid_precision_at_5):>11} "
            f"{_format_pct(pt.hybrid_abstention_rate):>9} "
            f"{pt.hybrid_false_positives:>7}"
        )

    lines.extend(["", f"NOTA: {report.provisional_threshold_note}"])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Executa avaliação quantitativa sintética controlada do motor RAG híbrido."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Exporta o relatório completo em formato JSON.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="Limiar padrão provisório de distância vetorial (padrão: 0.85).",
    )
    args = parser.parse_args(argv)

    import contextlib

    cases = load_eval_dataset()
    evaluator = RagEvaluator()
    with (
        contextlib.redirect_stdout(sys.stderr),
        tempfile.TemporaryDirectory(prefix="rbac_rag_eval_") as tmp_dir,
    ):
        _, _, _, factory = build_synthetic_eval_environment(Path(tmp_dir))
        report = evaluator.run_full_benchmark(
            dataset_name="rbac_synthetic_controlled_eval_v1",
            cases=cases,
            retriever_factory=factory,
            default_threshold=args.threshold,
        )

    if args.json:
        print(json.dumps(asdict(report), ensure_ascii=False, indent=2))
    else:
        print(render_report_table(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
