"""Configurações da aplicação via Pydantic Settings (.env + config/settings.yaml)."""

from __future__ import annotations

import os
import warnings
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import yaml
from pydantic import AliasChoices, BaseModel, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict


class StorageConfig(BaseModel):
    """Caminhos de volumes persistentes dentro do contêiner ou ambiente local."""

    documents_dir: Path = Path("/app/data/documents")
    sqlite_filename: str = "catalog.sqlite3"
    raw_pdfs_subdir: str = "raw_pdfs"
    extracted_subdir: str = "extracted"
    reviews_subdir: str = "reviews"
    audit_logs_dir: Path = Path("/app/logs")
    sqlite_busy_timeout_ms: int = Field(default=5000, ge=100)

    @property
    def sqlite_path(self) -> Path:
        """Caminho nativo do host/SO para o banco SQLite do catálogo e unidades normativas."""
        return self.documents_dir / self.sqlite_filename

    @property
    def canonical_container_sqlite_path(self) -> PurePosixPath:
        """Caminho POSIX canônico do SQLite dentro do contêiner Docker Linux."""
        return PurePosixPath(self.documents_dir.as_posix()) / self.sqlite_filename

    @property
    def raw_pdfs_dir(self) -> Path:
        """Diretório persistente para preservação dos PDFs originais."""
        return self.documents_dir / self.raw_pdfs_subdir

    @property
    def extracted_dir(self) -> Path:
        """Diretório persistente para artefatos extraídos auxiliares."""
        return self.documents_dir / self.extracted_subdir

    @property
    def reviews_dir(self) -> Path:
        """Diretório persistente para trilhas exportadas de revisão humana."""
        return self.documents_dir / self.reviews_subdir


class ExtractionConfig(BaseModel):
    """Parâmetros de extração PDF, limites de segurança e política de validação híbrida."""

    min_native_chars_per_page: int = Field(default=50, ge=1)
    ocr_language: str = "por"
    ocr_unreviewed_policy: Literal["block_until_reviewed", "warn_banner"] = (
        "block_until_reviewed"
    )
    validation_policy: Literal[
        "strict_all_pages",
        "review_on_demand",
        "native_auto_ocr_manual",
    ] = "review_on_demand"
    require_integrity_check_for_native_pages: bool = True
    max_pdf_size_mb: int = Field(default=50, ge=1, le=500)
    max_pdf_pages: int = Field(default=1000, ge=1, le=10000)
    extraction_version: str = "2.0.0"


class VectorDBConfig(BaseModel):
    """Configurações estruturais do banco vetorial ChromaDB."""

    provider: Literal["chromadb"] = "chromadb"
    pinned_version: str = "1.5.9"
    heartbeat_path: str = "/api/v2/heartbeat"
    request_timeout_seconds: float = Field(default=5.0, gt=0.0)
    collection_name: str = Field(default="regulatory_chunks_ge2_768_v1", min_length=1)
    test_collection_prefix: str = Field(default="test_", min_length=1)

    @field_validator("heartbeat_path")
    @classmethod
    def _validate_heartbeat_path(cls, value: str) -> str:
        if not value.startswith("/"):
            raise ValueError("heartbeat_path deve iniciar com '/'.")
        return value


CANONICAL_GEMINI_GENERATION_MODEL = "gemini-4-hydrogen"
CANONICAL_GEMINI_EMBEDDING_MODEL = "gemini-embedding-2"
CANONICAL_GEMINI_EMBEDDING_DIMENSIONS = 768


def _normalize_gemini_model_id(value: str) -> str:
    """Normaliza identificador de modelo Gemini removendo espaços e prefixo `models/`."""
    return value.strip().lower().removeprefix("models/")


class GeminiConfig(BaseModel):
    """Parâmetros operacionais da integração oficial com Google Gemini (`google-genai`)."""

    timeout_seconds: float = Field(default=30.0, gt=0.0, le=300.0)
    max_retries: int = Field(default=3, ge=0, le=10)
    retry_min_wait_seconds: float = Field(default=0.5, ge=0.0, le=30.0)
    retry_max_wait_seconds: float = Field(default=8.0, ge=0.0, le=120.0)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=2048, ge=128, le=8192)
    embedding_dimensions: int = Field(
        default=CANONICAL_GEMINI_EMBEDDING_DIMENSIONS,
        ge=128,
        le=3072,
    )
    embedding_instruction_version: str = Field(default="ge2_asymmetric_v1", min_length=1)
    embedding_max_input_tokens: int = Field(default=8192, ge=256, le=32768)
    embedding_batch_size: int = Field(default=32, ge=1, le=256)
    prompts_dir: Path = Path("config/prompts")
    allowed_free_tier_generation_models: tuple[str, ...] = (
        CANONICAL_GEMINI_GENERATION_MODEL,
    )
    allowed_free_tier_embedding_models: tuple[str, ...] = (
        CANONICAL_GEMINI_EMBEDDING_MODEL,
    )

    @field_validator("embedding_dimensions")
    @classmethod
    def _validate_canonical_embedding_dimensions(cls, value: int) -> int:
        if value != CANONICAL_GEMINI_EMBEDDING_DIMENSIONS:
            raise ValueError(
                f"Dimensionalidade de embeddings inválida ({value}). "
                f"A política única de integração real exige exatamente "
                f"{CANONICAL_GEMINI_EMBEDDING_DIMENSIONS} dimensões."
            )
        return value

    @field_validator("allowed_free_tier_generation_models")
    @classmethod
    def _validate_single_generation_allowlist(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(_normalize_gemini_model_id(item) for item in value)
        if normalized != (CANONICAL_GEMINI_GENERATION_MODEL,):
            raise ValueError(
                "A allowlist operacional de geração deve conter exclusivamente "
                f"'{CANONICAL_GEMINI_GENERATION_MODEL}' (recebido: {list(value)})."
            )
        return (CANONICAL_GEMINI_GENERATION_MODEL,)

    @field_validator("allowed_free_tier_embedding_models")
    @classmethod
    def _validate_single_embedding_allowlist(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(_normalize_gemini_model_id(item) for item in value)
        if normalized != (CANONICAL_GEMINI_EMBEDDING_MODEL,):
            raise ValueError(
                "A allowlist operacional de embeddings deve conter exclusivamente "
                f"'{CANONICAL_GEMINI_EMBEDDING_MODEL}' (recebido: {list(value)})."
            )
        return (CANONICAL_GEMINI_EMBEDDING_MODEL,)


class ChunkingConfig(BaseModel):
    """Parâmetros de chunking estrutural hierárquico."""

    max_chunk_tokens: int = Field(default=1200, ge=100)
    overlap_tokens: int = Field(default=0, ge=0)
    preserve_hierarchy: bool = True


class RetrievalConfig(BaseModel):
    """Parâmetros de busca híbrida (FTS5 + ChromaDB + RRF) e isolamento de emendas."""

    semantic_top_k: int = Field(default=5, ge=1, le=50)
    lexical_top_k: int = Field(default=5, ge=1, le=50)
    final_top_k: int = Field(default=5, ge=1, le=50)
    rrf_k: int = Field(default=60, ge=1, le=1000)
    # Threshold inicial/provisório para distância de cosseno no ChromaDB (`hnsw:space: cosine`),
    # sujeito a calibração por benchmark real após ingestão do RBAC 43.
    max_vector_distance: float = Field(default=0.85, gt=0.0, le=2.0)
    only_current_amendment_by_default: bool = True
    reconstruct_full_section_from_ssot: bool = True
    allow_preliminary_native_search: bool = True
    telegram_only_approved_literal: bool = True


class TelegramConfig(BaseModel):
    """Parâmetros operacionais da interface Telegram."""

    mode: Literal["long_polling"] = "long_polling"
    max_message_length: int = Field(default=4096, ge=256, le=4096)
    max_input_length: int = Field(default=1000, ge=64, le=4096)
    polling_timeout_seconds: int = Field(default=30, ge=1, le=120)
    max_concurrent_updates_per_user: int = Field(default=1, ge=1, le=5)
    max_global_concurrent_updates: int = Field(default=8, ge=1, le=64)
    command_timeout_seconds: float = Field(default=45.0, gt=1.0, le=300.0)
    duplicate_update_cache_size: int = Field(default=512, ge=32, le=10000)
    startup_retry_max_attempts: int = Field(default=5, ge=1, le=20)
    startup_retry_base_delay_seconds: float = Field(default=2.0, ge=0.1, le=60.0)
    allow_group_chats: bool = False


class RateLimitConfig(BaseModel):
    """Configuração de rate limiting por usuário e limite global para proteção pública."""

    requests_per_minute: int = Field(default=15, ge=1)
    burst_size: int = Field(default=5, ge=1)
    global_requests_per_minute: int = Field(default=60, ge=1)


class AuditConfig(BaseModel):
    """Parâmetros de auditoria persistente em JSONL, redação e rotação em `/app/logs`."""

    pseudonymize_user_ids: bool = False
    redact_secrets: bool = True
    filename: str = Field(default="audit_events.jsonl", min_length=1)
    max_file_size_mb: int = Field(default=10, ge=1, le=500)
    max_rotated_files: int = Field(default=10, ge=1, le=100)
    max_detail_string_length: int = Field(default=300, ge=64, le=2000)

    @property
    def max_file_size_bytes(self) -> int:
        """Retorna o limite de tamanho por arquivo JSONL de auditoria em bytes."""
        return self.max_file_size_mb * 1024 * 1024


class SecurityConfig(BaseModel):
    """Política de segurança operacional (acesso público com admins isolados ou restrito)."""

    public_access: bool = True
    default_role: Literal["USER"] = "USER"
    deny_by_default: bool = False
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)
    audit: AuditConfig = Field(default_factory=AuditConfig)


class AppSettings(BaseSettings):
    """Configuração consolidada carregada de variáveis de ambiente (.env) e YAML."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Omite leitura implícita de `.env` quando `RBAC_DISABLE_DOTENV=1` está ativo."""
        del settings_cls
        disable_dotenv = os.environ.get("RBAC_DISABLE_DOTENV", "").strip().lower() in {
            "1",
            "true",
            "yes",
        }
        if disable_dotenv:
            return (init_settings, env_settings, file_secret_settings)
        return (init_settings, env_settings, dotenv_settings, file_secret_settings)

    environment: Literal["local", "production", "test"] = "local"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # Conexão interna com o contêiner ChromaDB
    chroma_host: str = Field(default="chromadb", min_length=1)
    chroma_port: int = Field(default=8000, ge=1, le=65535)

    # Política única de modelos Gemini em runtime (Free Tier Standard, sem fallback silencioso)
    # Canônica: GEMINI_GENERATION_MODEL (GEMINI_CHAT_MODEL mantida apenas como alias legado)
    gemini_generation_model: str = Field(
        default=CANONICAL_GEMINI_GENERATION_MODEL,
        min_length=1,
        validation_alias=AliasChoices("GEMINI_GENERATION_MODEL", "GEMINI_CHAT_MODEL"),
    )
    gemini_embedding_model: str = Field(default=CANONICAL_GEMINI_EMBEDDING_MODEL, min_length=1)
    gemini_embedding_dimensions: int | None = Field(default=None, ge=128, le=3072)
    gemini_allow_paid_models: bool = False

    # Credenciais opcionais nas Etapas 1 e 2; obrigatórias para chamadas reais Gemini/Telegram
    gemini_api_key: SecretStr | None = None
    telegram_bot_token: SecretStr | None = None
    telegram_polling_enabled: bool = True

    # Caminhos de configuração YAML
    settings_yaml_path: Path = Path("config/settings.yaml")
    access_control_yaml_path: Path = Path("config/access_control.yaml")

    # Blocos não-sensíveis carregados do settings.yaml
    storage: StorageConfig = Field(default_factory=StorageConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    vector_db: VectorDBConfig = Field(default_factory=VectorDBConfig)
    gemini: GeminiConfig = Field(default_factory=GeminiConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    telegram: TelegramConfig = Field(default_factory=TelegramConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)

    @model_validator(mode="after")
    def _validate_gemini_models_and_aliases(self) -> AppSettings:
        if os.environ.get("GEMINI_CHAT_MODEL") and not os.environ.get("GEMINI_GENERATION_MODEL"):
            warnings.warn(
                "A variável GEMINI_CHAT_MODEL está obsoleta; utilize a variável canônica "
                "GEMINI_GENERATION_MODEL.",
                DeprecationWarning,
                stacklevel=2,
            )

        if self.gemini_allow_paid_models:
            raise ValueError(
                "GEMINI_ALLOW_PAID_MODELS=true é bloqueado pela política única Free Tier do "
                f"projeto. Mantenha GEMINI_ALLOW_PAID_MODELS=false com "
                f"'{CANONICAL_GEMINI_GENERATION_MODEL}' e '{CANONICAL_GEMINI_EMBEDDING_MODEL}' "
                f"({CANONICAL_GEMINI_EMBEDDING_DIMENSIONS}d)."
            )

        gen_norm = _normalize_gemini_model_id(self.gemini_generation_model)
        allowed_gen = {
            _normalize_gemini_model_id(m) for m in self.gemini.allowed_free_tier_generation_models
        }
        if gen_norm != CANONICAL_GEMINI_GENERATION_MODEL or gen_norm not in allowed_gen:
            raise ValueError(
                f"Modelo de geração '{self.gemini_generation_model}' bloqueado pela política única "
                f"Free Tier (único modelo permitido em runtime: "
                f"'{CANONICAL_GEMINI_GENERATION_MODEL}'). Não há fallback silencioso para "
                "Flash-Lite, Gemini 3.8 Flash ou modelos pagos."
            )
        self.gemini_generation_model = CANONICAL_GEMINI_GENERATION_MODEL

        emb_norm = _normalize_gemini_model_id(self.gemini_embedding_model)
        allowed_emb = {
            _normalize_gemini_model_id(m) for m in self.gemini.allowed_free_tier_embedding_models
        }
        if emb_norm != CANONICAL_GEMINI_EMBEDDING_MODEL or emb_norm not in allowed_emb:
            raise ValueError(
                f"Modelo de embeddings '{self.gemini_embedding_model}' bloqueado pela política "
                f"única Free Tier (único modelo permitido em runtime: "
                f"'{CANONICAL_GEMINI_EMBEDDING_MODEL}', "
                f"{CANONICAL_GEMINI_EMBEDDING_DIMENSIONS}d). Modelos legados como "
                "'gemini-embedding-001' não são aceitos e não há fallback silencioso."
            )
        self.gemini_embedding_model = CANONICAL_GEMINI_EMBEDDING_MODEL

        effective_dim = self.effective_embedding_dimensions
        if effective_dim != CANONICAL_GEMINI_EMBEDDING_DIMENSIONS:
            raise ValueError(
                f"Dimensionalidade de embeddings inválida ({effective_dim}). "
                f"A integração real com '{CANONICAL_GEMINI_EMBEDDING_MODEL}' exige exatamente "
                f"{CANONICAL_GEMINI_EMBEDDING_DIMENSIONS} dimensões "
                "(GEMINI_EMBEDDING_DIMENSIONS=768)."
            )

        return self

    @property
    def effective_embedding_dimensions(self) -> int:
        """Retorna a dimensão vetorial efetiva (env var sobrepõe YAML)."""
        return (
            self.gemini_embedding_dimensions
            if self.gemini_embedding_dimensions is not None
            else self.gemini.embedding_dimensions
        )

    @property
    def chroma_base_url(self) -> str:
        """Retorna a URL HTTP interna do servidor ChromaDB."""
        return f"http://{self.chroma_host}:{self.chroma_port}"

    @property
    def chroma_heartbeat_url(self) -> str:
        """Retorna a URL completa do endpoint oficial de heartbeat do ChromaDB."""
        return f"{self.chroma_base_url}{self.vector_db.heartbeat_path}"

    def require_gemini_api_key(self) -> str:
        """Retorna a GEMINI_API_KEY ou falha imediatamente se não estiver configurada."""
        if self.gemini_api_key is None or not self.gemini_api_key.get_secret_value().strip():
            raise ValueError("GEMINI_API_KEY não configurada no ambiente.")
        return self.gemini_api_key.get_secret_value()

    def require_telegram_bot_token(self) -> str:
        """Retorna o TELEGRAM_BOT_TOKEN ou falha imediatamente se não estiver configurado."""
        if (
            self.telegram_bot_token is None
            or not self.telegram_bot_token.get_secret_value().strip()
        ):
            raise ValueError("TELEGRAM_BOT_TOKEN não configurado no ambiente.")
        return self.telegram_bot_token.get_secret_value()


def _load_yaml_dict(yaml_path: Path) -> dict[str, Any]:
    if not yaml_path.exists():
        raise FileNotFoundError(f"Arquivo de configuração YAML não encontrado: {yaml_path}")
    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"Conteúdo YAML inválido em {yaml_path}: esperado mapeamento na raiz.")
    return raw


def load_settings(yaml_path: Path | None = None) -> AppSettings:
    """Carrega configurações combinando `config/settings.yaml` com variáveis de ambiente."""
    env_bootstrap = AppSettings()
    resolved_yaml = yaml_path or env_bootstrap.settings_yaml_path

    yaml_data: dict[str, Any] = {}
    if resolved_yaml.exists():
        yaml_data = _load_yaml_dict(resolved_yaml)
    elif yaml_path is not None:
        raise FileNotFoundError(f"Arquivo de configuração YAML não encontrado: {resolved_yaml}")

    section_keys = (
        "storage",
        "extraction",
        "vector_db",
        "gemini",
        "chunking",
        "retrieval",
        "telegram",
        "security",
    )
    overrides: dict[str, Any] = {
        key: yaml_data[key] for key in section_keys if key in yaml_data
    }
    if yaml_path is not None:
        overrides["settings_yaml_path"] = yaml_path

    return AppSettings(**overrides)
