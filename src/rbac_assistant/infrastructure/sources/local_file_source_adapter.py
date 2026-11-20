"""Adaptador para importação manual controlada de arquivos PDF locais."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

from rbac_assistant.domain.exceptions import InvalidPdfError, SourceFetchError
from rbac_assistant.domain.ports.document_source_port import FetchedDocument


class LocalFileSourceAdapter:
    """Importa PDFs locais com validação de existência, tamanho máximo, cabeçalho e SHA-256."""

    def __init__(self, *, max_pdf_size_mb: int = 50) -> None:
        self._max_bytes = max_pdf_size_mb * 1024 * 1024

    def fetch(self, source_uri: str, *, filename: str | None = None) -> FetchedDocument:
        path = Path(source_uri)
        if not path.is_file():
            raise SourceFetchError(f"Arquivo PDF local não encontrado: {path}")

        size = path.stat().st_size
        if size == 0:
            raise InvalidPdfError(f"Arquivo PDF vazio: {path}")
        if size > self._max_bytes:
            raise InvalidPdfError(
                f"Arquivo PDF ({size} bytes) excede o limite de "
                f"{self._max_bytes // (1024 * 1024)} MB."
            )

        pdf_bytes = path.read_bytes()
        if not pdf_bytes.startswith(b"%PDF-"):
            raise InvalidPdfError(
                f"Arquivo {path} não é um PDF válido (assinatura '%PDF-' ausente)."
            )

        file_hash = hashlib.sha256(pdf_bytes).hexdigest()
        return FetchedDocument(
            pdf_bytes=pdf_bytes,
            filename=filename or path.name,
            file_hash=file_hash,
            source_uri=str(path.resolve()),
            fetched_at=datetime.now(UTC),
        )
