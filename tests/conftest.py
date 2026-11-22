"""Fixtures globais compartilhadas pelos testes da Etapa 1."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import structlog

from rbac_assistant.config.settings import AppSettings
from rbac_assistant.domain.entities.normative_unit import (
    ExtractionMethod,
    PageExtractionMetadata,
    ValidationStatus,
)


@pytest.fixture(autouse=True)
def _isolate_tests_from_host_dotenv_and_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[None]:
    """Impede vazamento de `.env` local e estado de logging entre testes/workers xdist.

    Em hosts (como Windows 11 local) onde `.env` já contém `GEMINI_API_KEY`, `AppSettings`
    carregaria credenciais reais durante `pytest`, disparando conexões externas indevidas ao
    ChromaDB/Gemini. Em produção fora do `pytest`, `.env` continua ativo normalmente.
    """
    monkeypatch.setitem(AppSettings.model_config, "env_file", None)
    for secret_var in ("GEMINI_API_KEY", "TELEGRAM_BOT_TOKEN"):
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
