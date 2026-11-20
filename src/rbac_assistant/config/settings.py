"""Configurações da aplicação via Pydantic Settings (.env + config/settings.yaml)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class StorageConfig(BaseModel):
    """Caminhos de volumes persistentes dentro do contêiner ou ambiente local."""

    documents_dir: Path = Path("/app/data/documents")
    raw_pdfs_subdir: str = "raw_pdfs"
    canonical_store_subdir: str = "canonical_store"
    audit_logs_dir: Path = Path("/app/logs")


class ExtractionConfig(BaseModel):
    """Parâmetros de extração PDF e política individual por página para OCR."""

    min_native_chars_per_page: int = Field(default=50, ge=1)
    ocr_language: str = "por"
    ocr_unreviewed_policy: Literal["block_until_reviewed", "warn_banner"] = (
        "block_until_reviewed"
    )
    require_integrity_check_for_native_pages: bool = True


class VectorDBConfig(BaseModel):
    """Configurações estruturais do banco vetorial ChromaDB."""

    provider: Literal["chromadb"] = "chromadb"
    pinned_version: str = "1.5.9"
    heartbeat_path: str = "/api/v2/heartbeat"
    request_timeout_seconds: float = Field(default=5.0, gt=0.0)

    @field_validator("heartbeat_path")
    @classmethod
    def _validate_heartbeat_path(cls, value: str) -> str:
        if not value.startswith("/"):
            raise ValueError("heartbeat_path deve iniciar com '/'.")
        return value


class ChunkingConfig(BaseModel):
    """Parâmetros de chunking estrutural hierárquico."""

    max_chunk_tokens: int = Field(default=1200, ge=100)
    overlap_tokens: int = Field(default=0, ge=0)
    preserve_hierarchy: bool = True


class RetrievalConfig(BaseModel):
    """Parâmetros de busca híbrida e isolamento de emendas."""

    semantic_top_k: int = Field(default=5, ge=1, le=50)
    only_current_amendment_by_default: bool = True
    reconstruct_full_section_from_ssot: bool = True


class TelegramConfig(BaseModel):
    """Parâmetros operacionais da interface Telegram."""

    mode: Literal["long_polling"] = "long_polling"
    max_message_length: int = Field(default=4096, ge=256, le=4096)
    polling_timeout_seconds: int = Field(default=30, ge=1, le=120)


class RateLimitConfig(BaseModel):
    """Configuração de rate limiting por usuário."""

    requests_per_minute: int = Field(default=15, ge=1)
    burst_size: int = Field(default=5, ge=1)


class AuditConfig(BaseModel):
    """Parâmetros de auditoria e redação."""

    pseudonymize_user_ids: bool = False
    redact_secrets: bool = True


class SecurityConfig(BaseModel):
    """Política de segurança operacional."""

    deny_by_default: bool = True
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

    environment: Literal["local", "production", "test"] = "local"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # Conexão interna com o contêiner ChromaDB
    chroma_host: str = Field(default="chromadb", min_length=1)
    chroma_port: int = Field(default=8000, ge=1, le=65535)

    # Identificadores configuráveis dos modelos Gemini (usados a partir da Etapa 3)
    gemini_generation_model: str = Field(default="gemini-2.5-flash", min_length=1)
    gemini_embedding_model: str = Field(default="text-embedding-004", min_length=1)

    # Credenciais opcionais na Etapa 1; obrigatórias nos fluxos das Etapas 3 e 4
    gemini_api_key: SecretStr | None = None
    telegram_bot_token: SecretStr | None = None

    # Caminhos de configuração YAML
    settings_yaml_path: Path = Path("config/settings.yaml")

    # Blocos não-sensíveis carregados do settings.yaml
    storage: StorageConfig = Field(default_factory=StorageConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    vector_db: VectorDBConfig = Field(default_factory=VectorDBConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    telegram: TelegramConfig = Field(default_factory=TelegramConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)

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
