"""Testes unitários do adaptador oficial Google Gemini (`google-genai`) e configurações."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from google.genai import errors as genai_errors

from rbac_assistant.config.settings import AppSettings
from rbac_assistant.domain.exceptions import (
    LLMRateLimitError,
    LLMServiceError,
    LLMTimeoutError,
)
from rbac_assistant.infrastructure.llm.gemini_client import (
    GeminiClientAdapter,
    GeminiEmbeddingAdapter,
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
        return _FakeEmbedResponse(
            embeddings=[_FakeEmbeddingObj(values=[0.1] * dim) for _ in contents]
        )


class _FakeGenAISDKClient:
    def __init__(self) -> None:
        self.models = _FakeModelsAPI()
        self.closed = False

    def close(self) -> None:
        self.closed = True


@pytest.mark.unit
def test_gemini_client_requires_api_key_without_injected_client() -> None:
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        GeminiClientAdapter(api_key=None)


@pytest.mark.unit
def test_gemini_generate_and_embeddings_use_configured_models_and_dimensions() -> None:
    fake_sdk = _FakeGenAISDKClient()
    adapter = GeminiClientAdapter(
        generation_model="gemini-3.5-flash-lite",
        embedding_model="gemini-embedding-001",
        embedding_dimensions=8,
        embedding_batch_size=2,
        retry_min_wait_seconds=0.0,
        retry_max_wait_seconds=0.0,
        sdk_client=fake_sdk,
    )
    emb_adapter = GeminiEmbeddingAdapter(adapter)

    text = adapter.generate_text(
        system_instruction="Sistema normativo ANAC",
        user_prompt="Explique a seção 43.3",
    )
    assert text == "Explicação técnica fundamentada na seção."
    assert fake_sdk.models.generate_calls[0]["model"] == "gemini-3.5-flash-lite"
    gen_cfg = fake_sdk.models.generate_calls[0]["config"]
    assert gen_cfg.system_instruction == "Sistema normativo ANAC"

    doc_vecs = emb_adapter.embed_documents(["trecho 1", "trecho 2", "trecho 3"])
    assert len(doc_vecs) == 3
    assert all(len(v) == 8 for v in doc_vecs)
    # Batch size 2 => 2 chamadas (2 + 1)
    assert len(fake_sdk.models.embed_calls) == 2
    assert fake_sdk.models.embed_calls[0]["model"] == "gemini-embedding-001"
    assert fake_sdk.models.embed_calls[0]["config"].task_type == "RETRIEVAL_DOCUMENT"

    q_vec = emb_adapter.embed_query("quem executa manutenção")
    assert len(q_vec) == 8
    assert fake_sdk.models.embed_calls[-1]["config"].task_type == "RETRIEVAL_QUERY"

    adapter.close()
    assert fake_sdk.closed is True


@pytest.mark.unit
def test_gemini_retries_transient_429_and_eventually_succeeds() -> None:
    fake_sdk = _FakeGenAISDKClient()
    rate_err = genai_errors.APIError(429, {"error": {"message": "Quota exceeded"}})
    fake_sdk.models.generate_side_effects = [
        rate_err,
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
    assert len(fake_sdk.models.generate_calls) == 2


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
        embedding_dimensions=8,
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
def test_settings_supports_gemini_chat_model_alias_and_embedding_dimensions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GEMINI_GENERATION_MODEL", raising=False)
    monkeypatch.setenv("GEMINI_CHAT_MODEL", "gemini-3.8-flash")
    monkeypatch.setenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
    monkeypatch.setenv("GEMINI_EMBEDDING_DIMENSIONS", "1536")

    settings = AppSettings(_env_file=None)  # type: ignore[call-arg]
    assert settings.gemini_generation_model == "gemini-3.8-flash"
    assert settings.gemini_embedding_model == "gemini-embedding-001"
    assert settings.effective_embedding_dimensions == 1536
