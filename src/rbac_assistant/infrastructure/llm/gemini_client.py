"""Adaptador oficial para a Google Gemini Developer API usando `google-genai==3.6.1`.

Recursos implementados:
- Política única de modelos em runtime (Free Tier Standard, sem fallback silencioso):
  - Geração (`LLMPort`): exclusivamente `gemini-4-hydrogen` (sem fallback para Flash-Lite,
    Gemini 3.8 Flash ou modelos pagos, e sem ferramentas pagas como Google Search grounding,
    Google Maps ou Batch API).
  - Embeddings textuais (`EmbeddingPort`): exclusivamente `gemini-embedding-2` com
    dimensionalidade obrigatória de `768` (MRL):
    - Sem parâmetro `task_type` (não suportado pelo `gemini-embedding-2`) e sem suporte
      operacional ao legado `gemini-embedding-001`.
    - Convenção oficial de recuperação assimétrica incorporada apenas à representação de embedding
      (nunca altera `original_text` nem `normalized_text` no SQLite):
        - Consulta:  `task: search result | query: {consulta}`
        - Documento: `title: {título} | text: {conteúdo}` (ou `title: none | text: {conteúdo}`)
    - Cada chunk é encapsulado em seu próprio objeto `types.Content` para impedir agregação
      indevida no `gemini-embedding-2`, garantindo 1 vetor independente por chunk na mesma ordem.
    - Dimensões truncadas para `768` são normalizadas automaticamente pelo `gemini-embedding-2`,
      dispensando normalização L2 manual redundante e mantendo compatibilidade com cosseno.
    - Validação de texto não vazio e limite de entrada de 8.192 tokens.
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
DEFAULT_GENERATION_MODEL = "gemini-4-hydrogen"
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-2"
DEFAULT_EMBEDDING_DIMENSIONS = 768
DEFAULT_EMBEDDING_INSTRUCTION_VERSION = "ge2_asymmetric_v1"
DEFAULT_EMBEDDING_MAX_INPUT_TOKENS = 8192
_CONSERVATIVE_CHARS_PER_TOKEN = 3


def is_gemini_embedding_2(model_name: str) -> bool:
    """Verifica se o identificador corresponde exatamente ao modelo `gemini-embedding-2`."""
    normalized = model_name.strip().lower().removeprefix("models/")
    return normalized == DEFAULT_EMBEDDING_MODEL


def format_ge2_query(query: str) -> str:
    """Formata consulta segundo a convenção oficial assimétrica do Gemini Embedding 2."""
    clean_query = query.strip()
    if not clean_query:
        raise ValueError("Texto de consulta não pode ser vazio para geração de embedding.")
    return f"task: search result | query: {clean_query}"


def format_ge2_document(text: str, *, title: str | None = None) -> str:
    """Formata documento segundo a convenção oficial do Gemini Embedding 2.

    A representação formatada é efêmera e usada apenas na geração do vetor —
    jamais altera `original_text` nem `normalized_text` no SQLite.
    """
    clean_text = text.strip()
    if not clean_text:
        raise ValueError("Texto de documento não pode ser vazio para geração de embedding.")
    clean_title = (title or "").strip() or "none"
    return f"title: {clean_title} | text: {clean_text}"


def estimate_token_count(text: str) -> int:
    """Estimativa conservadora de tokens em português para guarda de limite de entrada."""
    word_est = int(len(text.split()) * 1.35)
    char_est = len(text) // _CONSERVATIVE_CHARS_PER_TOKEN
    return max(1, word_est, char_est)


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
        generation_model: str = DEFAULT_GENERATION_MODEL,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        embedding_dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS,
        embedding_instruction_version: str = DEFAULT_EMBEDDING_INSTRUCTION_VERSION,
        embedding_max_input_tokens: int = DEFAULT_EMBEDDING_MAX_INPUT_TOKENS,
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
        normalized_generation = generation_model.strip().lower().removeprefix("models/")
        if normalized_generation != DEFAULT_GENERATION_MODEL:
            raise ValueError(
                f"Modelo de geração '{generation_model}' não autorizado pela política única "
                f"Gemini. Somente '{DEFAULT_GENERATION_MODEL}' é permitido em runtime "
                "(sem fallback silencioso)."
            )
        normalized_embedding = embedding_model.strip().lower().removeprefix("models/")
        if normalized_embedding != DEFAULT_EMBEDDING_MODEL:
            raise ValueError(
                f"Modelo de embeddings '{embedding_model}' não autorizado pela política única "
                f"Gemini. Somente '{DEFAULT_EMBEDDING_MODEL}' é permitido em runtime "
                "(sem fallback silencioso)."
            )
        if embedding_dimensions != DEFAULT_EMBEDDING_DIMENSIONS:
            raise ValueError(
                f"Dimensionalidade de embeddings inválida ({embedding_dimensions}). "
                f"A integração real com '{DEFAULT_EMBEDDING_MODEL}' exige exatamente "
                f"{DEFAULT_EMBEDDING_DIMENSIONS} dimensões."
            )
        self._api_key = api_key.strip() if api_key else None
        self._generation_model = DEFAULT_GENERATION_MODEL
        self._embedding_model = DEFAULT_EMBEDDING_MODEL
        self._embedding_dimensions = DEFAULT_EMBEDDING_DIMENSIONS
        self._embedding_instruction_version = embedding_instruction_version.strip()
        self._embedding_max_input_tokens = embedding_max_input_tokens
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

    @property
    def instruction_strategy_version(self) -> str:
        """Retorna a versão da estratégia de instrução aplicada aos embeddings."""
        return self._embedding_instruction_version

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

    def _run_with_retry[T](self, operation_name: str, model_id: str, fn: Callable[[], T]) -> T:
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
                        translated, is_transient = self._translate_exception(
                            operation_name, model_id, exc
                        )
                        if is_transient:
                            logger.warning(
                                "gemini_transient_error",
                                operation=operation_name,
                                model=model_id,
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
    def _translate_exception(
        operation_name: str,
        model_id: str,
        exc: Exception,
    ) -> tuple[LLMServiceError, bool]:
        safe_detail = redact_text(str(exc))
        if isinstance(exc, TimeoutError | httpx.TimeoutException):
            return (
                LLMTimeoutError(
                    f"Timeout na chamada Gemini ({operation_name}, modelo '{model_id}'): "
                    f"{safe_detail}"
                ),
                True,
            )
        if isinstance(exc, genai_errors.APIError):
            code = getattr(exc, "code", None)
            if code == 429:
                return (
                    LLMRateLimitError(
                        f"Limite de taxa/cota atingido na Gemini API ({operation_name}, "
                        f"modelo '{model_id}', HTTP 429): {safe_detail}"
                    ),
                    True,
                )
            if code in _TRANSIENT_STATUS_CODES:
                return (
                    LLMServiceError(
                        f"Indisponibilidade temporária na Gemini API ({operation_name}, "
                        f"modelo '{model_id}', HTTP {code}): {safe_detail}"
                    ),
                    True,
                )
            if code in (403, 404):
                return (
                    LLMServiceError(
                        f"Modelo Gemini '{model_id}' indisponível ou sem acesso na conta "
                        f"({operation_name}, HTTP {code}). Nenhum fallback silencioso foi "
                        f"aplicado; verifique a disponibilidade do modelo autorizado na sua "
                        f"conta/chave da Gemini Developer API. Detalhe: {safe_detail}"
                    ),
                    False,
                )
            return (
                LLMServiceError(
                    f"Erro na Gemini API ({operation_name}, modelo '{model_id}', "
                    f"HTTP {code}): {safe_detail}"
                ),
                False,
            )
        if isinstance(exc, httpx.HTTPError | ConnectionError):
            return (
                LLMServiceError(
                    f"Erro de conectividade com a Gemini API ({operation_name}, "
                    f"modelo '{model_id}'): {safe_detail}"
                ),
                True,
            )
        return (
            LLMServiceError(
                f"Erro ao executar operação Gemini ({operation_name}, modelo '{model_id}'): "
                f"{safe_detail}"
            ),
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
        """Gera explicação fundamentada usando `client.models.generate_content`.

        Não executa fallback silencioso para outro modelo em caso de erro.
        """
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
                    f"Resposta vazia ou inválida retornada pelo modelo '{self._generation_model}'."
                )
            return text.strip()

        return self._run_with_retry("generate_content", self._generation_model, _call)

    def embed_documents(
        self,
        texts: Sequence[str],
        *,
        titles: Sequence[str | None] | None = None,
    ) -> list[tuple[float, ...]]:
        """Gera exatamente um vetor independente por chunk (`gemini-embedding-2`, 768d)."""
        if not texts:
            return []
        if titles is not None and len(titles) != len(texts):
            raise ValueError(
                f"Quantidade de títulos ({len(titles)}) difere da quantidade de textos "
                f"({len(texts)})."
            )

        prepared: list[str] = []
        for idx, raw_text in enumerate(texts):
            if not isinstance(raw_text, str) or not raw_text.strip():
                raise ValueError(
                    f"Chunk na posição {idx} está vazio ou inválido para geração de embedding."
                )
            title = titles[idx] if titles is not None else None
            payload = format_ge2_document(raw_text, title=title)
            self._validate_token_budget(payload, label=f"chunk[{idx}]")
            prepared.append(payload)

        all_vectors: list[tuple[float, ...]] = []
        for start in range(0, len(prepared), self._embedding_batch_size):
            batch = prepared[start : start + self._embedding_batch_size]
            batch_vectors = self._embed_batch(batch)
            all_vectors.extend(batch_vectors)
        return all_vectors

    def embed_query(self, text: str) -> tuple[float, ...]:
        """Gera vetor de embedding otimizado para consulta (`gemini-embedding-2`, 768d)."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Texto de consulta não pode ser vazio para geração de embedding.")

        payload = format_ge2_query(text)
        self._validate_token_budget(payload, label="query")
        vectors = self._embed_batch([payload])
        return vectors[0]

    def _validate_token_budget(self, payload: str, *, label: str) -> None:
        estimated = estimate_token_count(payload)
        if estimated > self._embedding_max_input_tokens:
            raise ValueError(
                f"Entrada '{label}' excede o limite de {self._embedding_max_input_tokens} tokens "
                f"do modelo '{self._embedding_model}' (estimado: {estimated} tokens)."
            )

    def _embed_batch(self, payloads: list[str]) -> list[tuple[float, ...]]:
        client = self._get_client()

        # Gemini Embedding 2 (modelo único de runtime, 768d):
        # 1) Não aceita `task_type` no EmbedContentConfig.
        # 2) Para impedir agregação em vetor único ao enviar múltiplos chunks, cada chunk é
        #    encapsulado em seu próprio objeto `types.Content`.
        # 3) Dimensões truncadas para 768 são normalizadas automaticamente pelo modelo.
        config = genai_types.EmbedContentConfig(
            output_dimensionality=self._embedding_dimensions,
        )
        contents: Any = [
            genai_types.Content(parts=[genai_types.Part.from_text(text=item)])
            for item in payloads
        ]

        def _call() -> list[tuple[float, ...]]:
            response = client.models.embed_content(
                model=self._embedding_model,
                contents=contents,
                config=config,
            )
            raw_embeddings = getattr(response, "embeddings", None)
            if not raw_embeddings or len(raw_embeddings) != len(payloads):
                raise LLMServiceError(
                    "Resposta inválida do endpoint de embeddings Gemini (possível agregação "
                    f"indevida ou truncamento): esperados {len(payloads)} vetor(es), recebidos "
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

        return self._run_with_retry("embed_content", self._embedding_model, _call)


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

    @property
    def instruction_strategy_version(self) -> str:
        return self._adapter.instruction_strategy_version

    def embed_documents(
        self,
        texts: Sequence[str],
        *,
        titles: Sequence[str | None] | None = None,
    ) -> list[tuple[float, ...]]:
        return self._adapter.embed_documents(texts, titles=titles)

    def embed_query(self, text: str) -> tuple[float, ...]:
        return self._adapter.embed_query(text)
