"""Contrato (Port) para obtenção de documentos PDF (URL oficial ou importação manual)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol


@dataclass(frozen=True, slots=True)
class FetchedDocument:
    """Documento binário obtido de fonte oficial ou importação local com integridade verificada."""

    pdf_bytes: bytes
    filename: str
    file_hash: str
    source_uri: str
    fetched_at: datetime
    content_type: str = "application/pdf"


class DocumentSourcePort(Protocol):
    """Interface abstrata para obtenção validada de PDFs regulatórios."""

    def fetch(self, source_uri: str, *, filename: str | None = None) -> FetchedDocument:
        """Obtém os bytes do PDF, valida formato/tamanho e calcula o hash SHA-256."""
