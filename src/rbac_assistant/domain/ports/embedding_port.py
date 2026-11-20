"""Contrato (Port) para geração de embeddings semânticos."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class EmbeddingPort(Protocol):
    """Interface abstrata para geração de vetores de embedding (implementado na Etapa 3)."""

    def embed_texts(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        """Gera vetores numéricos para um lote de textos."""

    def embed_query(self, query: str) -> tuple[float, ...]:
        """Gera vetor numérico para uma consulta de busca semântica."""
