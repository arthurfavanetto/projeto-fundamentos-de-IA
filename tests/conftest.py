"""Fixtures globais compartilhadas pelos testes da Etapa 1."""

from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import structlog
from google import genai as google_genai

# Proteção imediata no tempo de importação (antes da coleta de qualquer módulo de teste,
# inclusive em cada worker `pytest-xdist` criado por `spawn` no Windows e em subprocessos).
# Também habilita o modo moderno `datetime.timedelta` do `python-telegram-bot` (`PTB_TIMEDELTA=1`)
# antes da importação da biblioteca `telegram`.
_SECRET_ENV_VARS: tuple[str, ...] = ("GEMINI_API_KEY", "TELEGRAM_BOT_TOKEN")
os.environ["RBAC_DISABLE_DOTENV"] = "1"
os.environ["RBAC_BLOCK_REAL_GEMINI_API"] = "1"
os.environ["RBAC_BLOCK_REAL_TELEGRAM_API"] = "1"
os.environ["PTB_TIMEDELTA"] = "1"
for _secret_var in _SECRET_ENV_VARS:
    os.environ.pop(_secret_var, None)

from rbac_assistant.config.settings import AppSettings  # noqa: E402
from rbac_assistant.domain.entities.normative_unit import (  # noqa: E402
    ExtractionMethod,
    PageExtractionMetadata,
    ValidationStatus,
)

AppSettings.model_config["env_file"] = None


def _blocked_real_genai_client(*_args: Any, **_kwargs: Any) -> Any:
    raise RuntimeError(
        "Instanciação real de `google.genai.Client` bloqueada durante a suíte de testes "
        "(RBAC_BLOCK_REAL_GEMINI_API=1). Injete um `sdk_client` simulado."
    )


def pytest_configure(config: pytest.Config) -> None:
    """Reforça isolamento de `.env`, modo `PTB_TIMEDELTA=1` e bloqueio Gemini/Telegram."""
    del config
    os.environ["RBAC_DISABLE_DOTENV"] = "1"
    os.environ["RBAC_BLOCK_REAL_GEMINI_API"] = "1"
    os.environ["RBAC_BLOCK_REAL_TELEGRAM_API"] = "1"
    os.environ["PTB_TIMEDELTA"] = "1"
    for secret_var in _SECRET_ENV_VARS:
        os.environ.pop(secret_var, None)
    AppSettings.model_config["env_file"] = None


@pytest.fixture(autouse=True)
def _isolate_tests_from_host_dotenv_and_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[None]:
    """Impede vazamento de `.env`, chamadas reais à Gemini/Telegram e estado de logging.

    Abrange:
    - tempo de importação e `pytest_configure` (antes de qualquer fixture);
    - cada teste individual e cada worker `pytest-xdist` (`spawn` no Windows);
    - subprocessos filhos via `RBAC_DISABLE_DOTENV=1`, `RBAC_BLOCK_REAL_GEMINI_API=1`,
      `RBAC_BLOCK_REAL_TELEGRAM_API=1` e `PTB_TIMEDELTA=1`;
    - tentativa direta de criar `google.genai.Client` sem fake injetado;
    - reset determinístico do `structlog` no teardown.
    Em produção fora do `pytest`, `.env` e os SDKs operam normalmente.
    """
    monkeypatch.setenv("RBAC_DISABLE_DOTENV", "1")
    monkeypatch.setenv("RBAC_BLOCK_REAL_GEMINI_API", "1")
    monkeypatch.setenv("RBAC_BLOCK_REAL_TELEGRAM_API", "1")
    monkeypatch.setenv("PTB_TIMEDELTA", "1")
    monkeypatch.setitem(AppSettings.model_config, "env_file", None)
    monkeypatch.setattr(google_genai, "Client", _blocked_real_genai_client)
    for secret_var in _SECRET_ENV_VARS:
        monkeypatch.delenv(secret_var, raising=False)
    try:
        yield
    finally:
        structlog.reset_defaults()


class InMemoryChromaCollection:
    """Fake em memória compatível com a interface de coleção do ChromaDB 1.5.9."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._store: dict[str, dict[str, Any]] = {}

    def upsert(
        self,
        *,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        for idx, rec_id in enumerate(ids):
            self._store[rec_id] = {
                "document": documents[idx],
                "embedding": list(embeddings[idx]),
                "metadata": dict(metadatas[idx]),
            }

    def get(
        self,
        *,
        ids: list[str],
        include: list[str] | None = None,
    ) -> dict[str, Any]:
        del include
        found_ids: list[str] = []
        found_docs: list[str] = []
        found_embs: list[list[float]] = []
        found_metas: list[dict[str, Any]] = []
        for rec_id in ids:
            if rec_id in self._store:
                found_ids.append(rec_id)
                found_docs.append(self._store[rec_id]["document"])
                found_embs.append(self._store[rec_id]["embedding"])
                found_metas.append(self._store[rec_id]["metadata"])
        return {
            "ids": found_ids,
            "documents": found_docs,
            "embeddings": found_embs,
            "metadatas": found_metas,
        }

    def query(
        self,
        *,
        query_embeddings: list[list[float]],
        n_results: int = 5,
        where: dict[str, Any] | None = None,
        include: list[str] | None = None,
    ) -> dict[str, Any]:
        del query_embeddings, include
        matched: list[tuple[str, dict[str, Any]]] = []
        for rec_id, payload in self._store.items():
            if where and not all(payload["metadata"].get(k) == v for k, v in where.items()):
                continue
            matched.append((rec_id, payload))

        limited = matched[:n_results]
        return {
            "ids": [[item[0] for item in limited]],
            "documents": [[item[1]["document"] for item in limited]],
            "metadatas": [[item[1]["metadata"] for item in limited]],
            "distances": [[0.0 for _ in limited]],
        }


class InMemoryChromaClient:
    """Fake em memória simulando `chromadb.HttpClient` para testes unitários."""

    def __init__(self) -> None:
        self.collections: dict[str, InMemoryChromaCollection] = {}

    def get_or_create_collection(self, *, name: str) -> InMemoryChromaCollection:
        if name not in self.collections:
            self.collections[name] = InMemoryChromaCollection(name)
        return self.collections[name]

    def delete_collection(self, *, name: str) -> None:
        if name not in self.collections:
            raise ValueError(f"Collection '{name}' does not exist")
        del self.collections[name]


@pytest.fixture
def fake_chroma_client() -> InMemoryChromaClient:
    """Fornece um cliente ChromaDB em memória isolado por teste."""
    return InMemoryChromaClient()


@pytest.fixture
def validated_native_page() -> PageExtractionMetadata:
    """Página extraída nativamente do PDF e aprovada nas verificações de integridade."""
    return PageExtractionMetadata(
        pdf_page_number=4,
        printed_page_number="Página 4 de 28",
        extraction_method=ExtractionMethod.NATIVE_PDF,
        validation_status=ValidationStatus.VALIDATED,
        integrity_checks_passed=True,
        text_sha256="a" * 64,
    )


@pytest.fixture
def blocked_ocr_page() -> PageExtractionMetadata:
    """Página dependente de OCR bloqueada sob a política `block_until_reviewed`."""
    return PageExtractionMetadata.create_ocr_page(
        pdf_page_number=5,
        printed_page_number="Página 5 de 28",
        integrity_checks_passed=True,
        text_sha256="b" * 64,
        notes="Página digitalizada aguardando revisão manual.",
    )


@pytest.fixture
def reviewed_ocr_page() -> PageExtractionMetadata:
    """Página OCR já revisada manualmente e aprovada."""
    return PageExtractionMetadata(
        pdf_page_number=6,
        printed_page_number="Página 6 de 28",
        extraction_method=ExtractionMethod.OCR_FALLBACK,
        validation_status=ValidationStatus.VALIDATED,
        integrity_checks_passed=True,
        text_sha256="c" * 64,
        reviewed_by="admin_anac",
        reviewed_at=datetime(2026, 11, 20, 12, 0, tzinfo=UTC),
    )


@pytest.fixture
def repo_root() -> Path:
    """Retorna o diretório raiz do repositório."""
    return Path(__file__).resolve().parents[1]
