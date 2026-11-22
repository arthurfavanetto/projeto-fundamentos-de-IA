"""Contrato (Port) para geração de explicações complementares via LLM."""

from __future__ import annotations

from typing import Protocol


class LLMPort(Protocol):
    """Interface desacoplada para geração de explicações fundamentadas (Google Gemini)."""

    @property
    def model_name(self) -> str:
        """Identificador do modelo generativo ativo."""

    def generate_text(
        self,
        *,
        system_instruction: str,
        user_prompt: str,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> str:
        """Gera explicação textual a partir de instruções de sistema e prompt do usuário."""
