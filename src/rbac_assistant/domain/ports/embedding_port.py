"""Contrato (Port) para geração de embeddings semânticos."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class EmbeddingPort(Protocol):
    """Interface desacoplada para geração de vetores de embeddings (documentos e consultas)."""

    @property
    def model_name(self) -> str:
        """Identificador do modelo de embeddings ativo."""

    @property
    def dimensions(self) -> int:
        """Dimensionalidade vetorial esperada dos embeddings gerados."""

    def embed_documents(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        """Gera vetores de embeddings para uma lista de trechos/chunks de documentos."""

    def embed_query(self, text: str) -> tuple[float, ...]:
        """Gera vetor de embedding otimizado para uma consulta do usuário."""
