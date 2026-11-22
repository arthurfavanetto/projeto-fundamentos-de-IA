"""CLI operacional para avaliação quantitativa metodologicamente rigorosa do motor RAG.

Executa 40 consultas sintéticas controladas (4 documentos fictícios com numeração sobreposta,
subitens de 3 níveis, paráfrases, sinônimos, comparação multidocumento, ambiguidade de versão e
estados de revisão), comparando:
- Busca estrutural isolada (`structural_only`) + tabela estratificada de 5 subgrupos;
- Busca lexical FTS5 isolada (`lexical_fts5_only`);
- Busca vetorial isolada (`vector_only`);
- Busca híbrida RRF (`hybrid_rrf`) + ablações de pesos/reranking/poda adaptativa;
- Auditoria em 5 dimensões do `CitationValidator` (LLM mock — não Gemini real);
- Varredura estendida de limiar vetorial (`max_vector_distance` + `top_k` / `rrf_k`).

MODO MANUAL OPCIONAL (FORA DO CI — Parte 8):
- `--embedding-backend deterministic` (padrão): 100% local, zero chamadas de rede.
- `--embedding-backend gemini-embedding-2` ou `--embedding-backend compare`:
  usa `GEMINI_EMBEDDING_MODEL=gemini-embedding-2` (768d) via API Free Tier Standard
  usando a `GEMINI_API_KEY` local (`.env`), exclusivamente sobre o corpus SINTÉTICO fictício
  (nunca PDF oficial do RBAC 43, sem modelos pagos, sem Batch API, sem chamadas de geração LLM).
  AVISO: consome a cota gratuita da conta do usuário.

Exemplos:
    uv run python scripts/evaluate_rag.py
    uv run python scripts/evaluate_rag.py --per-query
    uv run python scripts/evaluate_rag.py --json
    uv run python scripts/evaluate_rag.py --embedding-backend compare
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from rbac_assistant.config.logging_config import configure_cli_logging
from rbac_assistant.config.settings import load_settings
from rbac_assistant.domain.ports.rag_ports import EmbeddingPort
from rbac_assistant.infrastructure.llm.gemini_client import (
    DEFAULT_EMBEDDING_MAX_INPUT_TOKENS,
    GeminiClientAdapter,
    GeminiEmbeddingAdapter,
)
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


def _pct(val: float) -> str:
    return f"{val * 100:.1f}%"


def render_report_table(
    report: EvaluationSummaryReport,
    *,
    include_per_query: bool = False,
) -> str:
    """Formata relatório completo e auditável para terminal e documentação."""
    by_mode = {m.mode: m for m in report.mode_metrics}
    lines: list[str] = [
        f"=== AVALIAÇÃO QUANTITATIVA CONTROLADA DO RAG ({report.dataset_name}) ===",
        f"Backend de embeddings: {report.embedding_backend} | Consultas: {report.total_queries}",
        f"NOTA EMBEDDER: {report.embedder_methodology_note}",
        "",
        "1) Comparação Agregada por Modo (limiar provisório max_vector_distance=0.85):",
        (
            f"{'Modo':<18} {'Rec@5':>6} {'Prec@5':>7} {'Prec@1':>7} {'MRR@10':>7} "
            f"{'nDCG@10':>8} {'AvgEv':>6} {'Estrut':>7} {'Granul':>7} {'DocFlt':>7} "
            f"{'VerFlt':>7} {'Abst':>6}"
        ),
        "-" * 106,
    ]
    for m in report.mode_metrics:
        lines.append(
            f"{m.mode:<18} {_pct(m.recall_at_5):>6} {_pct(m.precision_at_5):>7} "
            f"{_pct(m.precision_at_1):>7} {m.mrr_at_10:>7.3f} {m.ndcg_at_10:>8.3f} "
            f"{m.avg_returned_evidences:>6.2f} {_pct(m.exact_structural_rate):>7} "
            f"{_pct(m.granular_match_rate):>7} {_pct(m.document_filter_compliance_rate):>7} "
            f"{_pct(m.version_filter_compliance_rate):>7} {_pct(m.no_evidence_abstention_rate):>6}"
        )

    lines.extend(
        [
            "",
            "2) Decomposição Transparente de Erros por Modo (Parte 4.3):",
            (
                f"{'Modo':<18} {'RetFP':>6} {'RetFN':>6} {'NoEvAns':>8} {'AbstF':>6} "
                f"{'DocF':>5} {'VerF':>5} {'GovV':>5} {'HalCit':>7}"
            ),
            "-" * 74,
        ]
    )
    for m in report.mode_metrics:
        e = m.errors
        lines.append(
            f"{m.mode:<18} {e.retrieval_false_positives:>6} {e.retrieval_false_negatives:>6} "
            f"{e.unwarranted_answer_no_evidence:>8} {e.abstention_failures:>6} "
            f"{e.document_filter_failures:>5} {e.version_filter_failures:>5} "
            f"{e.governance_violations:>5} {e.hallucinated_citations:>7}"
        )

    # Parte 2.1: Avaliação estratificada do structural_only (5 subgrupos)
    struct = by_mode["structural_only"]
    lines.extend(
        [
            "",
            "3) Structural Only — Métricas Separadas por Subgrupo Estrutural (Parte 2.1):",
            f"{'Subgrupo':<30} {'Casos':>6} {'Resp':>5} {'Rec@5':>7} {'Prec@5':>7} "
            f"{'MRR@10':>7} {'Comport':>8}",
            "-" * 78,
        ]
    )
    for sub in struct.structural_subset_metrics:
        if sub.answerable_cases > 0:
            rec_s = _pct(sub.recall_at_5)
            prec_s = _pct(sub.precision_at_5)
            mrr_s = f"{sub.mrr_at_10:.3f}"
        else:
            rec_s = "n/a"
            prec_s = "n/a"
            mrr_s = "n/a"
        lines.append(
            f"{sub.subset:<30} {sub.total_cases:>6} {sub.answerable_cases:>5} "
            f"{rec_s:>7} {prec_s:>7} {mrr_s:>7} {_pct(sub.behavior_pass_rate):>8}"
        )

    # Parte 5: Resultados por Categoria comparando os 4 modos
    lines.extend(
        [
            "",
            "4) Resultados por Categoria — Comparação dos 4 Modos (Parte 5):",
            (
                f"{'Categoria':<28} {'N':>3} "
                f"{'Str R/P':>12} {'FTS R/P':>12} {'Vec R/P':>12} {'Hyb R/P':>12} {'HybBeh':>7}"
            ),
            "-" * 92,
        ]
    )
    cat_order = [c.category for c in by_mode["hybrid_rrf"].category_metrics]
    cat_maps = {
        mode: {c.category: c for c in m.category_metrics} for mode, m in by_mode.items()
    }
    for cat in cat_order:
        hc = cat_maps["hybrid_rrf"][cat]
        sc = cat_maps["structural_only"][cat]
        fc = cat_maps["lexical_fts5_only"][cat]
        vc = cat_maps["vector_only"][cat]
        if hc.answerable_cases > 0:
            s_str = f"{sc.recall_at_5 * 100:4.0f}/{sc.precision_at_5 * 100:4.0f}%"
            f_str = f"{fc.recall_at_5 * 100:4.0f}/{fc.precision_at_5 * 100:4.0f}%"
            v_str = f"{vc.recall_at_5 * 100:4.0f}/{vc.precision_at_5 * 100:4.0f}%"
            h_str = f"{hc.recall_at_5 * 100:4.0f}/{hc.precision_at_5 * 100:4.0f}%"
        else:
            s_str = f"beh {sc.behavior_pass_rate * 100:4.0f}%"
            f_str = f"beh {fc.behavior_pass_rate * 100:4.0f}%"
            v_str = f"beh {vc.behavior_pass_rate * 100:4.0f}%"
            h_str = f"beh {hc.behavior_pass_rate * 100:4.0f}%"
        lines.append(
            f"{cat:<28} {hc.total_cases:>3} {s_str:>12} {f_str:>12} {v_str:>12} {h_str:>12} "
            f"{_pct(hc.behavior_pass_rate):>7}"
        )

    # Parte 4.4: Auditoria dedicada em 5 dimensões do CitationValidator
    ca = report.citation_validator_audit
    lines.extend(
        [
            "",
            "5) Auditoria Determinística do CitationValidator — 5 Dimensões (Parte 4.4):",
            f"  - Validação de IDs de evidência ([EV-N]): {_pct(ca.evidence_id_validation_rate)}",
            f"  - Validação de referências estruturais: {_pct(ca.structural_ref_validation_rate)}",
            f"  - Validação de páginas (PDF/impressa): {_pct(ca.page_ref_validation_rate)}",
            f"  - Bloqueio de citação inexistente: {_pct(ca.nonexistent_citation_block_rate)}",
            f"  - Bloqueio de seção errada: {_pct(ca.wrong_section_citation_block_rate)}",
            f"  - Sondas aprovadas: {ca.probes_passed}/{ca.probes_total}",
            f"  - AVISO: {ca.disclaimer}",
        ]
    )

    # Parte 6: Ablação reproduzível do Hybrid RRF / Reranker / Poda Adaptativa
    if report.ablation_results:
        lines.extend(
            [
                "",
                "6) Experimento de Ablação do Hybrid RRF, Reranking e Poda Adaptativa (Parte 6):",
                (
                    f"{'Configuração':<30} {'Rec@5':>6} {'Prec@5':>7} {'Prec@1':>7} "
                    f"{'MRR@10':>7} {'nDCG':>6} {'AvgEv':>6} {'RetFP':>6} {'RetFN':>6} {'GovV':>5}"
                ),
                "-" * 99,
            ]
        )
        for ab in report.ablation_results:
            lines.append(
                f"{ab.config_name:<30} {_pct(ab.recall_at_5):>6} {_pct(ab.precision_at_5):>7} "
                f"{_pct(ab.precision_at_1):>7} {ab.mrr_at_10:>7.3f} {ab.ndcg_at_10:>6.3f} "
                f"{ab.avg_returned_evidences:>6.2f} {ab.retrieval_fp:>6} {ab.retrieval_fn:>6} "
                f"{ab.governance_violations:>5}"
            )

    # Parte 7: Varredura estendida de limiar vetorial
    lines.extend(
        [
            "",
            "7) Varredura de Limiar Vetorial (max_vector_distance + top_k/rrf_k) — PROVISÓRIO:",
            (
                f"{'Config':<25} {'VecR@5':>7} {'VecP@5':>7} {'VecMRR':>7} {'VecFP':>6} "
                f"{'HybR@5':>7} {'HybP@5':>7} {'HybMRR':>7} {'HybFP':>6} {'Abst':>6} {'Cats':>6}"
            ),
            "-" * 104,
        ]
    )
    for pt in report.threshold_sweep:
        cats_str = f"{pt.hybrid_categories_full_recall}/{pt.total_answerable_categories}"
        lines.append(
            f"{pt.label:<25} {_pct(pt.vector_recall_at_5):>7} {_pct(pt.vector_precision_at_5):>7} "
            f"{pt.vector_mrr_at_10:>7.3f} {pt.vector_retrieval_fp:>6} "
            f"{_pct(pt.hybrid_recall_at_5):>7} {_pct(pt.hybrid_precision_at_5):>7} "
            f"{pt.hybrid_mrr_at_10:>7.3f} {pt.hybrid_retrieval_fp:>6} "
            f"{_pct(pt.hybrid_abstention_rate):>6} {cats_str:>6}"
        )

    if include_per_query:
        lines.extend(
            [
                "",
                "8) Resultados Individuais por Consulta — Modo hybrid_rrf vs modos isolados:",
                (
                    f"{'ID':<4} {'Categoria':<26} {'Str':>4} {'FTS':>4} {'Vec':>4} {'Hyb':>4} "
                    f"{'HybP@5':>7} {'Retornados (hybrid_rrf)'}"
                ),
                "-" * 106,
            ]
        )
        mode_q = {
            mode: {d.case_id: d for d in m.query_details} for mode, m in by_mode.items()
        }
        for hd in by_mode["hybrid_rrf"].query_details:
            cid = hd.case_id
            s_ok = "OK" if mode_q["structural_only"][cid].behavior_ok else "MISS"
            f_ok = "OK" if mode_q["lexical_fts5_only"][cid].behavior_ok else "MISS"
            v_ok = "OK" if mode_q["vector_only"][cid].behavior_ok else "MISS"
            h_ok = "OK" if hd.behavior_ok else "MISS"
            p_str = _pct(hd.precision_at_5) if hd.precision_at_5 is not None else "n/a"
            ret_str = ", ".join(hd.returned_keys) if hd.returned_keys else "(vazio/bloqueado)"
            lines.append(
                f"{cid:<4} {hd.category:<26} {s_ok:>4} {f_ok:>4} {v_ok:>4} {h_ok:>4} "
                f"{p_str:>7} {ret_str}"
            )

    lines.extend(["", f"NOTA DE THRESHOLD: {report.provisional_threshold_note}"])
    return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class QueryEmbeddingCacheKey:
    """Chave composta imutável para cache temporário de embeddings de consulta no benchmark."""

    query_text: str
    embedding_model: str
    embedding_dimensions: int
    instruction_strategy_version: str
    extra_parameters: tuple[tuple[str, str], ...] = ()


class InMemoryQueryEmbeddingCache:
    """Cache temporário em memória exclusivo para embeddings de consulta no benchmark RAG.

    Garantias:
    - Vive apenas em memória RAM durante uma única execução do benchmark (zero persistência em
      disco, SQLite, ChromaDB ou volumes Docker).
    - Chave composta por texto exato da consulta, modelo de embedding, dimensionalidade, versão
      da estratégia de instrução/prefixo e parâmetros adicionais que possam alterar o vetor.
    - Encaminha `embed_documents(...)` diretamente ao adaptador delegado sem cache nem alteração.
    - Não altera nenhum serviço de recuperação em produção.
    """

    def __init__(
        self,
        delegate: EmbeddingPort,
        *,
        extra_parameters: Mapping[str, object] | None = None,
    ) -> None:
        self._delegate = delegate
        self._extra_parameters: tuple[tuple[str, str], ...] = tuple(
            sorted((str(k), str(v)) for k, v in (extra_parameters or {}).items())
        )
        self._cache: dict[QueryEmbeddingCacheKey, tuple[float, ...]] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    @property
    def model_name(self) -> str:
        return self._delegate.model_name

    @property
    def dimensions(self) -> int:
        return self._delegate.dimensions

    @property
    def instruction_strategy_version(self) -> str:
        return self._delegate.instruction_strategy_version

    @property
    def extra_parameters(self) -> tuple[tuple[str, str], ...]:
        return self._extra_parameters

    @property
    def cache_hits(self) -> int:
        return self._cache_hits

    @property
    def cache_misses(self) -> int:
        return self._cache_misses

    @property
    def cached_entries(self) -> int:
        return len(self._cache)

    def make_cache_key(self, text: str) -> QueryEmbeddingCacheKey:
        """Constrói chave completa impedindo mistura entre modelos, dimensões ou estratégias."""
        return QueryEmbeddingCacheKey(
            query_text=text,
            embedding_model=self._delegate.model_name,
            embedding_dimensions=self._delegate.dimensions,
            instruction_strategy_version=self._delegate.instruction_strategy_version,
            extra_parameters=self._extra_parameters,
        )

    def clear(self) -> None:
        """Limpa explicitamente todas as entradas em memória ao final da execução."""
        self._cache.clear()

    def embed_documents(
        self,
        texts: Sequence[str],
        *,
        titles: Sequence[str | None] | None = None,
    ) -> list[tuple[float, ...]]:
        """Encaminha vetorização documental sem cache e sem alterar payloads/vetores."""
        return self._delegate.embed_documents(texts, titles=titles)

    def embed_query(self, text: str) -> tuple[float, ...]:
        """Reutiliza vetor em memória para consultas idênticas dentro da mesma execução."""
        key = self.make_cache_key(text)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache_hits += 1
            return cached
        vector = self._delegate.embed_query(text)
        self._cache[key] = vector
        self._cache_misses += 1
        return vector


def _build_real_gemini_embedding_port() -> tuple[GeminiClientAdapter, EmbeddingPort]:
    """Cria adaptador real `gemini-embedding-2` (768d, Free Tier Standard, sem modelos pagos)."""
    settings = load_settings()
    if settings.gemini_allow_paid_models:
        raise ValueError(
            "O benchmark manual do corpus sintético exige GEMINI_ALLOW_PAID_MODELS=false "
            "(somente Free Tier Standard)."
        )
    if settings.gemini_embedding_model != "gemini-embedding-2":
        raise ValueError(
            f"Modelo de embedding configurado '{settings.gemini_embedding_model}' diverge de "
            "'gemini-embedding-2' (768d) exigido para este benchmark."
        )
    if settings.effective_embedding_dimensions != 768:
        raise ValueError(
            f"Dimensionalidade configurada ({settings.effective_embedding_dimensions}) diverge de "
            "768 dimensões obrigatórias para 'gemini-embedding-2'."
        )
    api_key = settings.require_gemini_api_key()
    client = GeminiClientAdapter(
        api_key=api_key,
        generation_model=settings.gemini_generation_model,
        embedding_model="gemini-embedding-2",
        embedding_dimensions=768,
        embedding_instruction_version=settings.gemini.embedding_instruction_version,
        embedding_max_input_tokens=settings.gemini.embedding_max_input_tokens,
        timeout_seconds=settings.gemini.timeout_seconds,
        max_retries=settings.gemini.max_retries,
        retry_min_wait_seconds=settings.gemini.retry_min_wait_seconds,
        retry_max_wait_seconds=settings.gemini.retry_max_wait_seconds,
        embedding_batch_size=settings.gemini.embedding_batch_size,
    )
    return client, GeminiEmbeddingAdapter(client)


def _real_query_cache_extra_parameters(client: GeminiClientAdapter | None) -> dict[str, object]:
    """Extrai parâmetros efetivos de vetorização que influenciam o vetor produzido."""
    max_tokens = getattr(client, "_embedding_max_input_tokens", DEFAULT_EMBEDDING_MAX_INPUT_TOKENS)
    return {
        "embedding_max_input_tokens": max_tokens,
        "query_prefix_template": "task: search result | query: {query}",
        "task_type": "none",
    }


def _run_benchmark_for_backend(
    *,
    backend: str,
    threshold: float,
    include_ablations: bool,
    enable_query_cache: bool | None = None,
) -> EvaluationSummaryReport:
    cases = load_eval_dataset()
    evaluator = RagEvaluator()
    client: GeminiClientAdapter | None = None
    embedder: EmbeddingPort | None = None
    query_cache: InMemoryQueryEmbeddingCache | None = None
    collection = "eval_synthetic_ge2_768_v1"

    try:
        if backend == "gemini-embedding-2":
            client, raw_embedder = _build_real_gemini_embedding_port()
            should_cache = True if enable_query_cache is None else enable_query_cache
            if should_cache:
                query_cache = InMemoryQueryEmbeddingCache(
                    raw_embedder,
                    extra_parameters=_real_query_cache_extra_parameters(client),
                )
                embedder = query_cache
            else:
                embedder = raw_embedder
            collection = "eval_synthetic_real_ge2_768_v1"

        with (
            contextlib.redirect_stdout(sys.stderr),
            tempfile.TemporaryDirectory(prefix="rbac_rag_eval_") as tmp_dir,
        ):
            repo, _, _, factory = build_synthetic_eval_environment(
                Path(tmp_dir),
                embedding_port=embedder,
                collection_name=collection,
            )
            try:
                return evaluator.run_full_benchmark(
                    dataset_name="rbac_synthetic_controlled_eval_v2",
                    embedding_backend=backend,
                    cases=cases,
                    retriever_factory=factory,
                    default_threshold=threshold,
                    include_ablations=include_ablations,
                )
            finally:
                repo.close()
    finally:
        if query_cache is not None:
            query_cache.clear()
        if client is not None:
            client.close()


def render_backend_comparison(
    det_report: EvaluationSummaryReport,
    real_report: EvaluationSummaryReport,
) -> str:
    """Renderiza comparação lado a lado entre embedder determinístico e Gemini Embedding 2 real."""
    det_by = {m.mode: m for m in det_report.mode_metrics}
    real_by = {m.mode: m for m in real_report.mode_metrics}
    lines = [
        "",
        "=== COMPARAÇÃO: DETERMINISTIC (SINTÉTICO) vs GEMINI-EMBEDDING-2 (REAL FREE TIER) ===",
        "Corpus: 100% sintético fictício (mesmas 40 consultas, documentos, filtros e alvos).",
        "AVISO: A execução real consome a cota gratuita (Free Tier) da GEMINI_API_KEY local.",
        "",
        f"{'Modo + Backend':<36} {'Rec@5':>7} {'Prec@5':>7} {'MRR@10':>7} {'nDCG@10':>8} "
        f"{'DocFlt':>7} {'VerFlt':>7} {'RetFP':>6} {'RetFN':>6}",
        "-" * 102,
    ]
    for mode in ("vector_only", "hybrid_rrf"):
        for label, src in (
            (f"{mode} [deterministic]", det_by[mode]),
            (f"{mode} [gemini-embedding-2]", real_by[mode]),
        ):
            lines.append(
                f"{label:<36} {_pct(src.recall_at_5):>7} {_pct(src.precision_at_5):>7} "
                f"{src.mrr_at_10:>7.3f} {src.ndcg_at_10:>8.3f} "
                f"{_pct(src.document_filter_compliance_rate):>7} "
                f"{_pct(src.version_filter_compliance_rate):>7} "
                f"{src.errors.retrieval_false_positives:>6} "
                f"{src.errors.retrieval_false_negatives:>6}"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Executa avaliação quantitativa controlada do motor RAG (corpus sintético)."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Exporta o relatório completo em formato JSON.",
    )
    parser.add_argument(
        "--per-query",
        action="store_true",
        help="Inclui tabela detalhada com o resultado individual de todas as 40 consultas.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="Limiar padrão provisório de distância vetorial (padrão: 0.85).",
    )
    parser.add_argument(
        "--embedding-backend",
        choices=("deterministic", "gemini-embedding-2", "compare"),
        default="deterministic",
        help=(
            "Backend de embeddings: 'deterministic' (padrão offline/CI), 'gemini-embedding-2' "
            "(manual Free Tier real sobre corpus sintético) ou 'compare' (ambos lado a lado)."
        ),
    )
    args = parser.parse_args(argv)
    configure_cli_logging(log_level="WARNING", redact_secrets=True)

    if args.embedding_backend == "compare":
        det_report = _run_benchmark_for_backend(
            backend="deterministic",
            threshold=args.threshold,
            include_ablations=True,
        )
        real_report = _run_benchmark_for_backend(
            backend="gemini-embedding-2",
            threshold=args.threshold,
            include_ablations=False,
        )
        if args.json:
            print(
                json.dumps(
                    {
                        "deterministic": asdict(det_report),
                        "gemini_embedding_2": asdict(real_report),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            print(render_report_table(det_report, include_per_query=args.per_query))
            print(render_backend_comparison(det_report, real_report))
        return 0

    report = _run_benchmark_for_backend(
        backend=args.embedding_backend,
        threshold=args.threshold,
        include_ablations=True,
    )
    if args.json:
        print(json.dumps(asdict(report), ensure_ascii=False, indent=2))
    else:
        print(render_report_table(report, include_per_query=args.per_query))
    return 0


if __name__ == "__main__":
    sys.exit(main())
