"""Testes unitários do adaptador oficial Google Gemini (`google-genai==3.6.1`).

Cobre:
- Política única de modelos em runtime no Free Tier Standard:
  - Geração: exclusivamente `gemini-4-hydrogen` (rejeita `gemini-3.5-flash-lite`,
    `gemini-3.8-flash`, `gemini-4-neon`, `gemini-4-carbon`, `gemini-4-argon` e qualquer outro);
  - Embeddings: exclusivamente `gemini-embedding-2` com `768` dimensões obrigatórias
    (rejeita `gemini-embedding-001` e dimensões diferentes de 768);
  - `GEMINI_ALLOW_PAID_MODELS=false` obrigatório (rejeita `true`);
- Ausência de fallback silencioso em geração e embeddings e ausência de ferramentas pagas
  (Google Search grounding/Maps/Batch);
- Gemini Embedding 2 (`gemini-embedding-2`, 768d MRL):
  - ausência de `task_type`;
  - formato assimétrico oficial (`task: search result | query: ...` e `title: ... | text: ...`);
  - encapsulamento individual em `types.Content` para impedir agregação indevida;
  - preservação de ordem e contagem 1:1 de vetores por chunk;
  - validação de texto vazio e limite de 8.192 tokens;
- Retry limitado para 429/5xx/timeout e redação de segredos;
- Variável canônica `GEMINI_GENERATION_MODEL` e aviso de depreciação em `GEMINI_CHAT_MODEL`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from rbac_assistant.config.settings import AppSettings, GeminiConfig, load_settings
from rbac_assistant.domain.exceptions import (
    LLMRateLimitError,
    LLMServiceError,
    LLMTimeoutError,
)
from rbac_assistant.infrastructure.llm.gemini_client import (
    DEFAULT_EMBEDDING_DIMENSIONS,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_GENERATION_MODEL,
    GeminiClientAdapter,
    GeminiEmbeddingAdapter,
    format_ge2_document,
    format_ge2_query,
    is_gemini_embedding_2,
)


@dataclass
class _FakeEmbeddingObj:
    values: list[float]


@dataclass
class _FakeEmbedResponse:
    embeddings: list[_FakeEmbeddingObj]


@dataclass
class _FakeGenerateResponse:
    text: str | None


class _FakeModelsAPI:
    def __init__(self) -> None:
        self.generate_calls: list[dict[str, Any]] = []
        self.embed_calls: list[dict[str, Any]] = []
        self.generate_side_effects: list[Any] = []
        self.embed_side_effects: list[Any] = []

    def generate_content(self, **kwargs: Any) -> Any:
        self.generate_calls.append(kwargs)
        if self.generate_side_effects:
            outcome = self.generate_side_effects.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        return _FakeGenerateResponse(text="Explicação técnica fundamentada na seção.")

    def embed_content(self, **kwargs: Any) -> Any:
        self.embed_calls.append(kwargs)
        if self.embed_side_effects:
            outcome = self.embed_side_effects.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        contents = kwargs["contents"]
        dim = kwargs["config"].output_dimensionality
        # Gera vetor determinístico distinto por posição para validar ordem 1:1
        return _FakeEmbedResponse(
            embeddings=[
                _FakeEmbeddingObj(values=[float(idx + 1) / 100.0] + [0.0] * (dim - 1))
                for idx, _ in enumerate(contents)
            ]
        )


class _FakeGenAISDKClient:
    def __init__(self) -> None:
        self.models = _FakeModelsAPI()
        self.closed = False

    def close(self) -> None:
        self.closed = True


@pytest.mark.unit
def test_default_free_tier_models_and_versioned_collection_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for var in (
        "GEMINI_GENERATION_MODEL",
        "GEMINI_CHAT_MODEL",
        "GEMINI_EMBEDDING_MODEL",
        "GEMINI_EMBEDDING_DIMENSIONS",
        "GEMINI_ALLOW_PAID_MODELS",
    ):
        monkeypatch.delenv(var, raising=False)

    settings = load_settings()
    assert settings.gemini_generation_model == "gemini-4-hydrogen"
    assert settings.gemini_embedding_model == "gemini-embedding-2"
    assert settings.effective_embedding_dimensions == 768
    assert settings.gemini_allow_paid_models is False
    assert settings.gemini.allowed_free_tier_generation_models == ("gemini-4-hydrogen",)
    assert settings.gemini.allowed_free_tier_embedding_models == ("gemini-embedding-2",)
    assert settings.vector_db.collection_name == "regulatory_chunks_ge2_768_v1"
    assert settings.gemini.embedding_instruction_version == "ge2_asymmetric_v1"
    assert settings.gemini.embedding_max_input_tokens == 8192
    assert DEFAULT_GENERATION_MODEL == "gemini-4-hydrogen"
    assert DEFAULT_EMBEDDING_MODEL == "gemini-embedding-2"
    assert DEFAULT_EMBEDDING_DIMENSIONS == 768
    assert is_gemini_embedding_2("gemini-embedding-2") is True
    assert is_gemini_embedding_2("gemini-embedding-001") is False


@pytest.mark.unit
@pytest.mark.parametrize(
    "unauthorized_model",
    [
        "gemini-4-neon",
        "gemini-4-carbon",
        "gemini-4-argon",
        "gemini-3.8-flash",
        "gemini-3.5-flash-lite",
        "gemini-2.5-flash",
    ],
)
def test_single_model_policy_blocks_all_non_hydrogen_generation_models(
    monkeypatch: pytest.MonkeyPatch,
    unauthorized_model: str,
) -> None:
    monkeypatch.setenv("GEMINI_GENERATION_MODEL", unauthorized_model)
    monkeypatch.delenv("GEMINI_ALLOW_PAID_MODELS", raising=False)

    with pytest.raises(ValueError, match="política única Free Tier"):
        AppSettings(_env_file=None)  # type: ignore[call-arg]

    fake_sdk = _FakeGenAISDKClient()
    with pytest.raises(ValueError, match="política única Gemini"):
        GeminiClientAdapter(generation_model=unauthorized_model, sdk_client=fake_sdk)


@pytest.mark.unit
def test_single_model_policy_blocks_legacy_embeddings_bad_dims_paid_flag_and_yaml_allowlist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for var in (
        "GEMINI_GENERATION_MODEL",
        "GEMINI_CHAT_MODEL",
        "GEMINI_EMBEDDING_MODEL",
        "GEMINI_EMBEDDING_DIMENSIONS",
        "GEMINI_ALLOW_PAID_MODELS",
    ):
        monkeypatch.delenv(var, raising=False)

    # 1. Bloqueio de modelos de embedding legados / alternativos em AppSettings e no adapter
    monkeypatch.setenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
    with pytest.raises(ValueError, match="gemini-embedding-001"):
        AppSettings(_env_file=None)  # type: ignore[call-arg]
    monkeypatch.delenv("GEMINI_EMBEDDING_MODEL", raising=False)

    fake_sdk = _FakeGenAISDKClient()
    with pytest.raises(ValueError, match="gemini-embedding-2"):
        GeminiClientAdapter(embedding_model="gemini-embedding-001", sdk_client=fake_sdk)

    # 2. Bloqueio de dimensionalidade diferente de 768 em AppSettings, GeminiConfig e adapter
    for bad_dim in (256, 512, 1536, 3072):
        monkeypatch.setenv("GEMINI_EMBEDDING_DIMENSIONS", str(bad_dim))
        with pytest.raises(ValueError, match="768"):
            AppSettings(_env_file=None)  # type: ignore[call-arg]
        with pytest.raises(ValueError, match="768"):
            GeminiConfig(embedding_dimensions=bad_dim)
        with pytest.raises(ValueError, match="768"):
            GeminiClientAdapter(embedding_dimensions=bad_dim, sdk_client=fake_sdk)
    monkeypatch.delenv("GEMINI_EMBEDDING_DIMENSIONS", raising=False)

    # 3. Bloqueio explícito de GEMINI_ALLOW_PAID_MODELS=true
    monkeypatch.setenv("GEMINI_ALLOW_PAID_MODELS", "true")
    with pytest.raises(ValueError, match="GEMINI_ALLOW_PAID_MODELS=true"):
        AppSettings(_env_file=None)  # type: ignore[call-arg]
    monkeypatch.delenv("GEMINI_ALLOW_PAID_MODELS", raising=False)

    # 4. Allowlist YAML não pode reintroduzir modelos removidos
    with pytest.raises(ValueError, match="allowlist operacional de geração"):
        GeminiConfig(
            allowed_free_tier_generation_models=("gemini-4-hydrogen", "gemini-3.5-flash-lite")
        )
    with pytest.raises(ValueError, match="allowlist operacional de embeddings"):
        GeminiConfig(allowed_free_tier_embedding_models=("gemini-embedding-001",))


@pytest.mark.unit
def test_gemini_client_requires_api_key_without_injected_client() -> None:
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        GeminiClientAdapter(api_key=None)


@pytest.mark.unit
def test_gemini_4_hydrogen_generation_and_ge2_asymmetric_embeddings() -> None:
    fake_sdk = _FakeGenAISDKClient()
    adapter = GeminiClientAdapter(
        generation_model="gemini-4-hydrogen",
        embedding_model="gemini-embedding-2",
        embedding_dimensions=768,
        embedding_batch_size=2,
        retry_min_wait_seconds=0.0,
        retry_max_wait_seconds=0.0,
        sdk_client=fake_sdk,
    )
    emb_adapter = GeminiEmbeddingAdapter(adapter)

    # 1. Geração com gemini-4-hydrogen sem ferramentas pagas (tools=None)
    text = adapter.generate_text(
        system_instruction="Sistema normativo ANAC",
        user_prompt="Explique a seção 43.3",
    )
    assert text == "Explicação técnica fundamentada na seção."
    assert fake_sdk.models.generate_calls[0]["model"] == "gemini-4-hydrogen"
    gen_cfg = fake_sdk.models.generate_calls[0]["config"]
    assert gen_cfg.system_instruction == "Sistema normativo ANAC"
    assert getattr(gen_cfg, "tools", None) is None

    # 2. Embeddings de documentos com gemini-embedding-2 (768d, sem task_type, Content separado)
    raw_texts = ["trecho original 1", "trecho original 2", "trecho original 3"]
    titles = ["RBAC 43 — Seção 43.1", None, "RBAC 43 — Seção 43.9"]
    doc_vecs = emb_adapter.embed_documents(raw_texts, titles=titles)

    assert len(doc_vecs) == 3
    assert all(len(v) == 768 for v in doc_vecs)
    assert doc_vecs[0][0] == pytest.approx(0.01)
    assert doc_vecs[1][0] == pytest.approx(0.02)
    assert doc_vecs[2][0] == pytest.approx(0.01)
    assert raw_texts == ["trecho original 1", "trecho original 2", "trecho original 3"]

    assert len(fake_sdk.models.embed_calls) == 2
    first_call = fake_sdk.models.embed_calls[0]
    assert first_call["model"] == "gemini-embedding-2"
    assert first_call["config"].task_type is None
    assert first_call["config"].output_dimensionality == 768
    assert len(first_call["contents"]) == 2
    assert all(isinstance(item, genai_types.Content) for item in first_call["contents"])
    assert (
        first_call["contents"][0].parts[0].text
        == "title: RBAC 43 — Seção 43.1 | text: trecho original 1"
    )
    assert (
        first_call["contents"][1].parts[0].text
        == "title: none | text: trecho original 2"
    )

    # 3. Embedding de consulta com prefixo oficial assimétrico e sem task_type
    q_vec = emb_adapter.embed_query("quem executa manutenção")
    assert len(q_vec) == 768
    query_call = fake_sdk.models.embed_calls[-1]
    assert query_call["model"] == "gemini-embedding-2"
    assert query_call["config"].task_type is None
    assert query_call["config"].output_dimensionality == 768
    assert isinstance(query_call["contents"][0], genai_types.Content)
    assert (
        query_call["contents"][0].parts[0].text
        == "task: search result | query: quem executa manutenção"
    )

    adapter.close()
    assert fake_sdk.closed is True


@pytest.mark.unit
def test_ge2_rejects_aggregated_single_vector_empty_text_and_token_overflow() -> None:
    fake_sdk = _FakeGenAISDKClient()
    adapter = GeminiClientAdapter(
        embedding_model="gemini-embedding-2",
        embedding_dimensions=768,
        embedding_max_input_tokens=200,
        max_retries=0,
        retry_min_wait_seconds=0.0,
        retry_max_wait_seconds=0.0,
        sdk_client=fake_sdk,
    )

    fake_sdk.models.embed_side_effects = [
        _FakeEmbedResponse(embeddings=[_FakeEmbeddingObj(values=[0.1] * 768)])
    ]
    with pytest.raises(LLMServiceError, match="agregação indevida"):
        adapter.embed_documents(["chunk A", "chunk B"])

    with pytest.raises(ValueError, match="vazio"):
        adapter.embed_documents(["   "])

    with pytest.raises(ValueError, match="vazio"):
        adapter.embed_query("   ")

    with pytest.raises(ValueError, match="excede o limite"):
        adapter.embed_documents(["palavra " * 300])

    assert format_ge2_query("teste") == "task: search result | query: teste"
    assert format_ge2_document("corpo", title=None) == "title: none | text: corpo"
    assert format_ge2_document("corpo", title="Seção 43.3") == "title: Seção 43.3 | text: corpo"


@pytest.mark.unit
def test_gemini_model_unavailable_has_no_silent_fallback() -> None:
    fake_sdk = _FakeGenAISDKClient()
    fake_sdk.models.generate_side_effects = [
        genai_errors.APIError(404, {"error": {"message": "Model gemini-4-hydrogen not found"}})
    ]
    fake_sdk.models.embed_side_effects = [
        genai_errors.APIError(404, {"error": {"message": "Model gemini-embedding-2 not found"}})
    ]
    adapter = GeminiClientAdapter(
        generation_model="gemini-4-hydrogen",
        embedding_model="gemini-embedding-2",
        embedding_dimensions=768,
        max_retries=2,
        retry_min_wait_seconds=0.0,
        retry_max_wait_seconds=0.0,
        sdk_client=fake_sdk,
    )

    with pytest.raises(LLMServiceError, match="Nenhum fallback silencioso foi aplicado"):
        adapter.generate_text(system_instruction="sys", user_prompt="usr")

    assert len(fake_sdk.models.generate_calls) == 1
    assert fake_sdk.models.generate_calls[0]["model"] == "gemini-4-hydrogen"

    with pytest.raises(LLMServiceError, match="Nenhum fallback silencioso foi aplicado"):
        adapter.embed_query("manutenção preventiva")

    assert len(fake_sdk.models.embed_calls) == 1
    assert fake_sdk.models.embed_calls[0]["model"] == "gemini-embedding-2"


@pytest.mark.unit
def test_gemini_retries_transient_429_and_503_and_eventually_succeeds() -> None:
    fake_sdk = _FakeGenAISDKClient()
    fake_sdk.models.generate_side_effects = [
        genai_errors.APIError(429, {"error": {"message": "Quota exceeded"}}),
        genai_errors.APIError(503, {"error": {"message": "Service unavailable"}}),
        _FakeGenerateResponse(text="Explicação após retry."),
    ]
    adapter = GeminiClientAdapter(
        max_retries=2,
        retry_min_wait_seconds=0.0,
        retry_max_wait_seconds=0.0,
        sdk_client=fake_sdk,
    )

    result = adapter.generate_text(system_instruction="sys", user_prompt="usr")
    assert result == "Explicação após retry."
    assert len(fake_sdk.models.generate_calls) == 3


@pytest.mark.unit
def test_gemini_raises_rate_limit_timeout_and_redacts_secrets() -> None:
    fake_sdk = _FakeGenAISDKClient()
    secret_key = "AIzaSySecretKeyLeakTest1234567890"
    fake_sdk.models.generate_side_effects = [
        genai_errors.APIError(429, {"error": {"message": f"Rate limit key={secret_key}"}}),
        genai_errors.APIError(429, {"error": {"message": f"Rate limit key={secret_key}"}}),
    ]
    adapter = GeminiClientAdapter(
        max_retries=1,
        retry_min_wait_seconds=0.0,
        retry_max_wait_seconds=0.0,
        sdk_client=fake_sdk,
    )

    with pytest.raises(LLMRateLimitError) as exc_info:
        adapter.generate_text(system_instruction="sys", user_prompt="usr")
    assert secret_key not in str(exc_info.value)
    assert "[REDACTED]" in str(exc_info.value)

    fake_sdk.models.generate_side_effects = [TimeoutError("socket timed out")]
    adapter_no_retry = GeminiClientAdapter(
        max_retries=0,
        retry_min_wait_seconds=0.0,
        retry_max_wait_seconds=0.0,
        sdk_client=fake_sdk,
    )
    with pytest.raises(LLMTimeoutError, match="Timeout"):
        adapter_no_retry.generate_text(system_instruction="sys", user_prompt="usr")


@pytest.mark.unit
def test_gemini_rejects_empty_response_and_wrong_embedding_dimension() -> None:
    fake_sdk = _FakeGenAISDKClient()
    fake_sdk.models.generate_side_effects = [_FakeGenerateResponse(text="   ")]
    adapter = GeminiClientAdapter(
        embedding_dimensions=768,
        max_retries=0,
        retry_min_wait_seconds=0.0,
        retry_max_wait_seconds=0.0,
        sdk_client=fake_sdk,
    )

    with pytest.raises(LLMServiceError, match="Resposta vazia"):
        adapter.generate_text(system_instruction="sys", user_prompt="usr")

    fake_sdk.models.embed_side_effects = [
        _FakeEmbedResponse(embeddings=[_FakeEmbeddingObj(values=[0.1, 0.2, 0.3])])
    ]
    with pytest.raises(LLMServiceError, match="Dimensionalidade inesperada"):
        adapter.embed_query("teste")


@pytest.mark.unit
def test_settings_canonical_env_var_and_legacy_chat_model_deprecation_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_GENERATION_MODEL", "gemini-4-hydrogen")
    monkeypatch.setenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2")
    monkeypatch.setenv("GEMINI_EMBEDDING_DIMENSIONS", "768")
    monkeypatch.delenv("GEMINI_CHAT_MODEL", raising=False)
    monkeypatch.delenv("GEMINI_ALLOW_PAID_MODELS", raising=False)

    settings = AppSettings(_env_file=None)  # type: ignore[call-arg]
    assert settings.gemini_generation_model == "gemini-4-hydrogen"
    assert settings.gemini_embedding_model == "gemini-embedding-2"
    assert settings.effective_embedding_dimensions == 768

    # Alias legado GEMINI_CHAT_MODEL ainda aceito somente para o modelo autorizado com aviso
    monkeypatch.delenv("GEMINI_GENERATION_MODEL", raising=False)
    monkeypatch.setenv("GEMINI_CHAT_MODEL", "gemini-4-hydrogen")
    with pytest.warns(DeprecationWarning, match="GEMINI_GENERATION_MODEL"):
        legacy_settings = AppSettings(_env_file=None)  # type: ignore[call-arg]
    assert legacy_settings.gemini_generation_model == "gemini-4-hydrogen"

    # Mesmo via alias legado GEMINI_CHAT_MODEL, outros modelos são rejeitados sem fallback
    monkeypatch.setenv("GEMINI_CHAT_MODEL", "gemini-3.5-flash-lite")
    with (
        pytest.warns(DeprecationWarning, match="GEMINI_GENERATION_MODEL"),
        pytest.raises(ValueError, match="política única Free Tier"),
    ):
        AppSettings(_env_file=None)  # type: ignore[call-arg]
