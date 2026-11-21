"""Adaptador oficial para a Google Gemini API usando o SDK `google-genai`.

Recursos implementados:
- Geração de explicações fundamentadas (`LLMPort`) e embeddings (`EmbeddingPort`).
- Modelos configuráveis via variáveis de ambiente (`GEMINI_GENERATION_MODEL` /
  `GEMINI_EMBEDDING_MODEL`) e dimensionalidade MRL configurável.
- Timeout explícito via `types.HttpOptions(timeout=ms)`.
- Retry limitado com backoff exponencial via `tenacity` apenas em falhas transitórias
  (HTTP 429 rate limit, 5xx ou timeout).
- Redação rigorosa de segredos em mensagens de erro e logs (nunca expõe `GEMINI_API_KEY`).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import httpx
import structlog
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types
from tenacity import (
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from rbac_assistant.domain.exceptions import (
    LLMRateLimitError,
    LLMServiceError,
    LLMTimeoutError,
)
from rbac_assistant.security.secret_redactor import redact_text

logger = structlog.get_logger(__name__)

_TRANSIENT_STATUS_CODES: frozenset[int] = frozenset({429, 500, 502, 503, 504})


class _TransientGeminiError(Exception):
    """Wrapper interno para acionar retry limitado em falhas transitórias."""

    def __init__(self, domain_error: LLMServiceError) -> None:
        self.domain_error = domain_error
        super().__init__(str(domain_error))


class GeminiClientAdapter:
    """Adaptador unificado `LLMPort` + `EmbeddingPort` sobre o SDK oficial `google-genai`."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        generation_model: str = "gemini-3.5-flash-lite",
        embedding_model: str = "gemini-embedding-001",
        embedding_dimensions: int = 768,
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
        retry_min_wait_seconds: float = 0.5,
        retry_max_wait_seconds: float = 8.0,
        default_temperature: float = 0.1,
        default_max_output_tokens: int = 2048,
        embedding_batch_size: int = 32,
        sdk_client: Any | None = None,
    ) -> None:
        if sdk_client is None and (api_key is None or not api_key.strip()):
            raise ValueError(
                "GEMINI_API_KEY é obrigatória para inicializar o adaptador Google Gemini."
            )
        self._api_key = api_key.strip() if api_key else None
        self._generation_model = generation_model.strip()
        self._embedding_model = embedding_model.strip()
        self._embedding_dimensions = embedding_dimensions
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries
        self._retry_min_wait_seconds = retry_min_wait_seconds
        self._retry_max_wait_seconds = retry_max_wait_seconds
        self._default_temperature = default_temperature
        self._default_max_output_tokens = default_max_output_tokens
        self._embedding_batch_size = max(1, embedding_batch_size)
        self._client: Any | None = sdk_client

    @property
    def model_name(self) -> str:
        """Retorna o modelo de geração configurado (para `LLMPort`)."""
        return self._generation_model

    @property
    def embedding_model_name(self) -> str:
        """Retorna o modelo de embeddings configurado."""
        return self._embedding_model

    @property
    def dimensions(self) -> int:
        """Retorna a dimensionalidade vetorial esperada (para `EmbeddingPort`)."""
        return self._embedding_dimensions

    def _get_client(self) -> Any:
        if self._client is None:
            timeout_ms = max(1000, int(self._timeout_seconds * 1000))
            http_options = genai_types.HttpOptions(timeout=timeout_ms)
            self._client = genai.Client(
                api_key=self._api_key,
                http_options=http_options,
            )
        return self._client

    def close(self) -> None:
        """Fecha recursos HTTP subjacentes do cliente síncrono quando aplicável."""
        if self._client is not None and hasattr(self._client, "close"):
            self._client.close()

    def _run_with_retry[T](self, operation_name: str, fn: Callable[[], T]) -> T:
        attempts = self._max_retries + 1
        retrying = Retrying(
            reraise=True,
            stop=stop_after_attempt(attempts),
            wait=wait_exponential(
                multiplier=max(self._retry_min_wait_seconds, 0.01),
                min=self._retry_min_wait_seconds,
                max=self._retry_max_wait_seconds,
            ),
            retry=retry_if_exception_type(_TransientGeminiError),
        )
        try:
            for attempt in retrying:
                with attempt:
                    try:
                        return fn()
                    except LLMServiceError:
                        raise
                    except Exception as exc:
                        translated, is_transient = self._translate_exception(operation_name, exc)
                        if is_transient:
                            logger.warning(
                                "gemini_transient_error",
                                operation=operation_name,
                                attempt=attempt.retry_state.attempt_number,
                                max_attempts=attempts,
                                error=redact_text(str(translated)),
                            )
                            raise _TransientGeminiError(translated) from exc
                        raise translated from exc
        except _TransientGeminiError as wrapped:
            raise wrapped.domain_error from wrapped
        raise LLMServiceError(f"Falha inesperada na operação Gemini '{operation_name}'.")

    @staticmethod
    def _translate_exception(operation_name: str, exc: Exception) -> tuple[LLMServiceError, bool]:
        safe_detail = redact_text(str(exc))
        if isinstance(exc, TimeoutError | httpx.TimeoutException):
            return (
                LLMTimeoutError(
                    f"Timeout na chamada Gemini ({operation_name}): {safe_detail}"
                ),
                True,
            )
        if isinstance(exc, genai_errors.APIError):
            code = getattr(exc, "code", None)
            if code == 429:
                return (
                    LLMRateLimitError(
                        f"Limite de taxa/cota atingido na Gemini API ({operation_name}, HTTP 429): "
                        f"{safe_detail}"
                    ),
                    True,
                )
            if code in _TRANSIENT_STATUS_CODES:
                return (
                    LLMServiceError(
                        f"Indisponibilidade temporária na Gemini API ({operation_name}, "
                        f"HTTP {code}): {safe_detail}"
                    ),
                    True,
                )
            return (
                LLMServiceError(
                    f"Erro na Gemini API ({operation_name}, HTTP {code}): {safe_detail}"
                ),
                False,
            )
        if isinstance(exc, httpx.HTTPError | ConnectionError):
            return (
                LLMServiceError(
                    f"Erro de conectividade com a Gemini API ({operation_name}): {safe_detail}"
                ),
                True,
            )
        return (
            LLMServiceError(f"Erro ao executar operação Gemini ({operation_name}): {safe_detail}"),
            False,
        )

    def generate_text(
        self,
        *,
        system_instruction: str,
        user_prompt: str,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> str:
        """Gera explicação fundamentada usando `client.models.generate_content`."""
        client = self._get_client()
        config = genai_types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=self._default_temperature if temperature is None else temperature,
            max_output_tokens=(
                self._default_max_output_tokens
                if max_output_tokens is None
                else max_output_tokens
            ),
        )

        def _call() -> str:
            response = client.models.generate_content(
                model=self._generation_model,
                contents=user_prompt,
                config=config,
            )
            text = getattr(response, "text", None)
            if not isinstance(text, str) or not text.strip():
                raise LLMServiceError(
                    "Resposta vazia ou inválida retornada pelo modelo generativo Gemini."
                )
            return text.strip()

        return self._run_with_retry("generate_content", _call)

    def embed_documents(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        """Gera vetores de embeddings em lotes para chunks de documentos regulatórios."""
        if not texts:
            return []
        all_vectors: list[tuple[float, ...]] = []
        for start in range(0, len(texts), self._embedding_batch_size):
            batch = list(texts[start : start + self._embedding_batch_size])
            batch_vectors = self._embed_batch(batch, task_type="RETRIEVAL_DOCUMENT")
            all_vectors.extend(batch_vectors)
        return all_vectors

    def embed_query(self, text: str) -> tuple[float, ...]:
        """Gera vetor de embedding otimizado para consulta do usuário."""
        if not text.strip():
            raise ValueError("Texto de consulta não pode ser vazio para geração de embedding.")
        vectors = self._embed_batch([text], task_type="RETRIEVAL_QUERY")
        return vectors[0]

    def _embed_batch(self, texts: list[str], *, task_type: str) -> list[tuple[float, ...]]:
        client = self._get_client()
        config = genai_types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=self._embedding_dimensions,
        )

        def _call() -> list[tuple[float, ...]]:
            response = client.models.embed_content(
                model=self._embedding_model,
                contents=texts,
                config=config,
            )
            raw_embeddings = getattr(response, "embeddings", None)
            if not raw_embeddings or len(raw_embeddings) != len(texts):
                raise LLMServiceError(
                    "Resposta inválida do endpoint de embeddings Gemini: "
                    f"esperados {len(texts)} vetor(es), recebidos "
                    f"{len(raw_embeddings) if raw_embeddings is not None else 0}."
                )
            vectors: list[tuple[float, ...]] = []
            for emb in raw_embeddings:
                values = getattr(emb, "values", None)
                if not values:
                    raise LLMServiceError("Vetor de embedding vazio retornado pela Gemini API.")
                vec = tuple(float(v) for v in values)
                if len(vec) != self._embedding_dimensions:
                    raise LLMServiceError(
                        f"Dimensionalidade inesperada no embedding retornado: "
                        f"esperado {self._embedding_dimensions}, recebido {len(vec)}."
                    )
                vectors.append(vec)
            return vectors

        return self._run_with_retry("embed_content", _call)


class GeminiEmbeddingAdapter:
    """Visão dedicada de `EmbeddingPort` que expõe `model_name` como o modelo de embeddings."""

    def __init__(self, client_adapter: GeminiClientAdapter) -> None:
        self._adapter = client_adapter

    @property
    def model_name(self) -> str:
        return self._adapter.embedding_model_name

    @property
    def dimensions(self) -> int:
        return self._adapter.dimensions

    def embed_documents(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        return self._adapter.embed_documents(texts)

    def embed_query(self, text: str) -> tuple[float, ...]:
        return self._adapter.embed_query(text)
