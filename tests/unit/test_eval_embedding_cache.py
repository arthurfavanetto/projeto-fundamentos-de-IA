"""Testes do cache temporário em memória para embeddings de consulta no benchmark RAG.

Comprova sem chamadas reais à API Gemini:
- Exatamente 1 chamada por consulta única (40 consultas únicas no corpus de avaliação);
- Exatamente 6 lotes documentais (18 chunks em 6 versões), sem alterar embeddings documentais;
- Total de 46 chamadas HTTP `embed_content` no caminho real simulado de
  `scripts/evaluate_rag.py --embedding-backend compare` (vs. 1.286 sem cache = -96,4%);
- Zero chamadas a `generate_content` (Hydrogen nunca é invocado no benchmark de embeddings);
- Isolamento de chave por texto exato, modelo, dimensionalidade, estratégia e parâmetros extras;
- Existência estritamente temporária em memória (sem reutilização entre execuções e sem
  persistência em disco, SQLite, ChromaDB ou volumes Docker);
- Equivalência integral (bit a bit) das métricas antes e depois da otimização.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pytest
from scripts import evaluate_rag as eval_cli

from rbac_assistant.infrastructure.llm.gemini_client import (
    GeminiClientAdapter,
    GeminiEmbeddingAdapter,
)
from rbac_assistant.rag.evaluation.rag_evaluator import RagEvaluator
from tests.evals.synthetic_rag_eval_fixture import (
    DeterministicEvalSemanticEmbedder,
    build_synthetic_eval_environment,
    load_eval_dataset,
)


@dataclass
class _FakeEmbeddingObj:
    values: list[float]


@dataclass
class _FakeEmbedResponse:
    embeddings: list[_FakeEmbeddingObj]


class _CountingFakeGenAISDKClient:
    """Fake do SDK `google-genai` que delega vetores ao embedder sintético e conta chamadas HTTP."""

    def __init__(self) -> None:
        self._encoder = DeterministicEvalSemanticEmbedder(dimensions=768)
        self.models = self
        self.generate_calls: list[dict[str, Any]] = []
        self.embed_calls: list[dict[str, Any]] = []
        self.document_batch_calls = 0
        self.document_texts_count = 0
        self.query_calls = 0
        self.unique_query_payloads: set[str] = set()
        self.closed = False

    def generate_content(self, **kwargs: Any) -> Any:
        self.generate_calls.append(kwargs)
        raise AssertionError("generate_content (Hydrogen) não deve ser chamado neste benchmark.")

    def embed_content(self, **kwargs: Any) -> _FakeEmbedResponse:
        self.embed_calls.append(kwargs)
        contents = kwargs["contents"]
        dim = kwargs["config"].output_dimensionality
        assert kwargs["model"] == "gemini-embedding-2"
        assert dim == 768
        assert kwargs["config"].task_type is None

        texts = [item.parts[0].text for item in contents]
        if len(texts) == 1 and texts[0].startswith("task: search result | query: "):
            self.query_calls += 1
            self.unique_query_payloads.add(texts[0])
            raw_q = texts[0].removeprefix("task: search result | query: ")
            vec = list(self._encoder.embed_query(raw_q))
            return _FakeEmbedResponse(embeddings=[_FakeEmbeddingObj(values=vec)])

        self.document_batch_calls += 1
        self.document_texts_count += len(texts)
        embeddings = [
            _FakeEmbeddingObj(values=list(self._encoder.embed_query(t))) for t in texts
        ]
        return _FakeEmbedResponse(embeddings=embeddings)

    def close(self) -> None:
        self.closed = True


class _MutableConfigFakeEmbedder:
    """Embedder fake configurável para validar separação de chave no cache."""

    def __init__(
        self,
        *,
        model_name: str = "fake-model-a",
        dimensions: int = 768,
        instruction_strategy_version: str = "strat_v1",
        tag_offset: float = 0.0,
    ) -> None:
        self.model_name = model_name
        self.dimensions = dimensions
        self.instruction_strategy_version = instruction_strategy_version
        self.tag_offset = tag_offset
        self.query_calls: list[str] = []
        self.doc_calls: list[tuple[tuple[str, ...], tuple[str | None, ...] | None]] = []

    def embed_documents(
        self,
        texts: Sequence[str],
        *,
        titles: Sequence[str | None] | None = None,
    ) -> list[tuple[float, ...]]:
        self.doc_calls.append((tuple(texts), tuple(titles) if titles is not None else None))
        return [(1.0 + self.tag_offset,) + (0.0,) * (self.dimensions - 1) for _ in texts]

    def embed_query(self, text: str) -> tuple[float, ...]:
        self.query_calls.append(text)
        base = float(len(self.query_calls)) + self.tag_offset
        return (base,) + (0.0,) * (self.dimensions - 1)


@pytest.mark.unit
def test_in_memory_query_cache_reuses_exact_query_isolates_keys_and_passes_through_documents() -> (
    None
):
    delegate = _MutableConfigFakeEmbedder(
        model_name="fake-model-a",
        dimensions=768,
        instruction_strategy_version="strat_v1",
    )
    cache = eval_cli.InMemoryQueryEmbeddingCache(
        delegate,
        extra_parameters={"task_type": "none", "prefix": "v1"},
    )

    # 1. Consulta repetida reutiliza exatamente o mesmo vetor em memória
    v1 = cache.embed_query("Quem pode executar manutenção no RBAC 43.3?")
    v2 = cache.embed_query("Quem pode executar manutenção no RBAC 43.3?")
    assert v1 == v2
    assert len(delegate.query_calls) == 1
    assert cache.cache_misses == 1
    assert cache.cache_hits == 1
    assert cache.cached_entries == 1

    # 2. Texto distinto (incluindo diferença de caixa/espaço) gera nova entrada
    v_case = cache.embed_query("quem pode executar manutenção no RBAC 43.3?")
    assert v_case != v1
    assert len(delegate.query_calls) == 2

    # 3. Alteração de modelo, dimensão ou estratégia gera chaves isoladas (sem mistura)
    delegate.model_name = "fake-model-b"
    v_model_b = cache.embed_query("Quem pode executar manutenção no RBAC 43.3?")
    assert v_model_b != v1
    assert len(delegate.query_calls) == 3

    delegate.dimensions = 1536
    v_dim_1536 = cache.embed_query("Quem pode executar manutenção no RBAC 43.3?")
    assert len(v_dim_1536) == 1536
    assert len(delegate.query_calls) == 4

    delegate.instruction_strategy_version = "strat_v2"
    v_strat_v2 = cache.embed_query("Quem pode executar manutenção no RBAC 43.3?")
    assert v_strat_v2 != v_dim_1536
    assert len(delegate.query_calls) == 5

    # 4. Parâmetros extras diferentes geram chave diferente
    cache_other_params = eval_cli.InMemoryQueryEmbeddingCache(
        delegate,
        extra_parameters={"task_type": "none", "prefix": "v2"},
    )
    assert cache.make_cache_key("q") != cache_other_params.make_cache_key("q")

    # 5. Embeddings documentais passam diretamente sem cache nem alteração
    d1 = cache.embed_documents(["trecho 1"], titles=["Seção 43.3"])
    d2 = cache.embed_documents(["trecho 1"], titles=["Seção 43.3"])
    assert d1 == d2
    assert len(delegate.doc_calls) == 2

    # 6. Limpeza explícita esvazia completamente o cache
    cache.clear()
    assert cache.cached_entries == 0


@pytest.mark.integration
def test_compare_benchmark_real_path_hits_exactly_46_embed_calls_and_zero_generation(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    created_sdks: list[_CountingFakeGenAISDKClient] = []

    def _fake_build_real_port() -> tuple[GeminiClientAdapter, GeminiEmbeddingAdapter]:
        fake_sdk = _CountingFakeGenAISDKClient()
        created_sdks.append(fake_sdk)
        client = GeminiClientAdapter(
            generation_model="gemini-4-hydrogen",
            embedding_model="gemini-embedding-2",
            embedding_dimensions=768,
            embedding_instruction_version="ge2_asymmetric_v1",
            embedding_max_input_tokens=8192,
            embedding_batch_size=32,
            max_retries=0,
            retry_min_wait_seconds=0.0,
            retry_max_wait_seconds=0.0,
            sdk_client=fake_sdk,
        )
        return client, GeminiEmbeddingAdapter(client)

    monkeypatch.setattr(eval_cli, "_build_real_gemini_embedding_port", _fake_build_real_port)
    monkeypatch.chdir(tmp_path)

    # Execução 1 via CLI `--embedding-backend compare` (sempre sem chamadas reais à API)
    rc = eval_cli.main(["--embedding-backend", "compare"])
    assert rc == 0
    stdout = capsys.readouterr().out
    assert "COMPARAÇÃO: DETERMINISTIC (SINTÉTICO) vs GEMINI-EMBEDDING-2" in stdout

    assert len(created_sdks) == 1
    sdk_run1 = created_sdks[0]
    assert sdk_run1.closed is True
    assert sdk_run1.generate_calls == []
    assert sdk_run1.document_batch_calls == 6
    assert sdk_run1.document_texts_count == 18
    assert sdk_run1.query_calls == 40
    assert len(sdk_run1.unique_query_payloads) == 40
    assert len(sdk_run1.embed_calls) == 46

    # Execução 2 independente não reutiliza cache da execução 1
    eval_cli._run_benchmark_for_backend(
        backend="gemini-embedding-2",
        threshold=0.85,
        include_ablations=False,
    )
    assert len(created_sdks) == 2
    sdk_run2 = created_sdks[1]
    assert sdk_run2.closed is True
    assert sdk_run2.generate_calls == []
    assert sdk_run2.document_batch_calls == 6
    assert sdk_run2.query_calls == 40
    assert len(sdk_run2.embed_calls) == 46

    # Execução sem cache comprova o baseline de 1.286 chamadas (redução real de 96,4%)
    eval_cli._run_benchmark_for_backend(
        backend="gemini-embedding-2",
        threshold=0.85,
        include_ablations=False,
        enable_query_cache=False,
    )
    sdk_uncached = created_sdks[2]
    assert sdk_uncached.document_batch_calls == 6
    assert sdk_uncached.query_calls == 1280
    assert len(sdk_uncached.embed_calls) == 1286
    reduction = 1.0 - (len(sdk_run1.embed_calls) / len(sdk_uncached.embed_calls))
    assert reduction == pytest.approx(1.0 - 46 / 1286, rel=1e-6)
    assert reduction > 0.964

    # Nenhum arquivo persistido no diretório de trabalho (tudo isolado em TemporaryDirectory + RAM)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.integration
def test_query_embedding_cache_preserves_identical_metrics_before_and_after(
    tmp_path: Path,
) -> None:
    cases = load_eval_dataset()
    assert len(cases) == 40

    class _CountingDeterministicEmbedder:
        def __init__(self) -> None:
            self._inner = DeterministicEvalSemanticEmbedder(dimensions=768)
            self.doc_calls = 0
            self.query_calls = 0

        @property
        def model_name(self) -> str:
            return self._inner.model_name

        @property
        def dimensions(self) -> int:
            return self._inner.dimensions

        @property
        def instruction_strategy_version(self) -> str:
            return self._inner.instruction_strategy_version

        def embed_documents(
            self,
            texts: Sequence[str],
            *,
            titles: Sequence[str | None] | None = None,
        ) -> list[tuple[float, ...]]:
            self.doc_calls += 1
            return self._inner.embed_documents(texts, titles=titles)

        def embed_query(self, text: str) -> tuple[float, ...]:
            self.query_calls += 1
            return self._inner.embed_query(text)

    evaluator = RagEvaluator()

    # Baseline sem cache
    base_embedder = _CountingDeterministicEmbedder()
    _, _, _, base_factory = build_synthetic_eval_environment(
        tmp_path / "uncached",
        embedding_port=base_embedder,
        collection_name="eval_compare_uncached",
    )
    report_uncached = evaluator.run_full_benchmark(
        dataset_name="rbac_synthetic_controlled_eval_v2",
        embedding_backend="deterministic",
        cases=cases,
        retriever_factory=base_factory,
        default_threshold=0.85,
        include_ablations=False,
    )

    # Otimizado com InMemoryQueryEmbeddingCache
    cached_delegate = _CountingDeterministicEmbedder()
    cached_embedder = eval_cli.InMemoryQueryEmbeddingCache(
        cached_delegate,
        extra_parameters={"backend": "deterministic"},
    )
    _, _, _, cached_factory = build_synthetic_eval_environment(
        tmp_path / "cached",
        embedding_port=cached_embedder,
        collection_name="eval_compare_cached",
    )
    report_cached = evaluator.run_full_benchmark(
        dataset_name="rbac_synthetic_controlled_eval_v2",
        embedding_backend="deterministic",
        cases=cases,
        retriever_factory=cached_factory,
        default_threshold=0.85,
        include_ablations=False,
    )

    assert base_embedder.doc_calls == cached_delegate.doc_calls == 6
    assert base_embedder.query_calls == 1280
    assert cached_delegate.query_calls == 40
    assert cached_embedder.cache_misses == 40
    assert cached_embedder.cache_hits == 1240
    assert asdict(report_cached) == asdict(report_uncached)
