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

    @property
    def instruction_strategy_version(self) -> str:
        """Versão da estratégia de instrução/formatação usada para gerar os embeddings."""

    def embed_documents(
        self,
        texts: Sequence[str],
        *,
        titles: Sequence[str | None] | None = None,
    ) -> list[tuple[float, ...]]:
        """Gera exatamente um vetor independente por chunk de documento, preservando a ordem."""

    def embed_query(self, text: str) -> tuple[float, ...]:
        """Gera vetor de embedding otimizado para uma consulta do usuário."""
