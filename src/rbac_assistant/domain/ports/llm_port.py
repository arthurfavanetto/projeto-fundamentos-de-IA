"""Contrato (Port) para geração de explicações complementares via LLM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class LLMGenerationResult:
    """Resultado de geração de explicação complementar pelo modelo de linguagem."""

    text: str
    model_name: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


class LLMPort(Protocol):
    """Interface abstrata para o provedor de LLM (implementado na Etapa 3 via google-genai)."""

    def generate_explanation(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> LLMGenerationResult:
        """Gera explicação técnica estritamente fundamentada nas evidências fornecidas."""
