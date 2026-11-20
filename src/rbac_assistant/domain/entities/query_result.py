"""Entidade de resultado normativo preservando separação entre texto oficial e explicação IA."""

from __future__ import annotations

from dataclasses import dataclass

from rbac_assistant.domain.entities.citation import Citation


@dataclass(frozen=True, slots=True)
class NormativeResponse:
    """Resposta estruturada que separa rigorosamente transcrição literal e explicação IA.

    Invariantes:
    - `official_text` é sempre oriundo do armazenamento canônico (SSOT), nunca gerado pelo LLM.
    - Quando `literal_transcription_guaranteed` for `False` (ex.: seção abrange página OCR
      bloqueada), `official_text` pode ficar ausente/parcial e `fidelity_warning` deve ser
      preenchido obrigatoriamente.
    """

    query: str
    found: bool
    citations: tuple[Citation, ...] = ()
    official_text: str | None = None
    ai_explanation: str | None = None
    literal_transcription_guaranteed: bool = False
    fidelity_warning: str | None = None

    def __post_init__(self) -> None:
        if self.found and not self.literal_transcription_guaranteed and not self.fidelity_warning:
            raise ValueError(
                "fidelity_warning é obrigatório quando a transcrição literal completa "
                "não puder ser garantida."
            )
