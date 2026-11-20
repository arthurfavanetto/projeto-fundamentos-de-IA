"""Adaptador para obtenção segura de PDFs oficiais a partir de URLs HTTPS (ex.: ANAC)."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

import httpx

from rbac_assistant.domain.exceptions import InvalidPdfError, SourceFetchError
from rbac_assistant.domain.ports.document_source_port import FetchedDocument


class AnacHttpSourceAdapter:
    """Baixa PDFs oficiais via HTTPS com validação de esquema, limite de tamanho e SHA-256."""

    def __init__(
        self,
        *,
        max_pdf_size_mb: int = 50,
        timeout_seconds: float = 30.0,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._max_bytes = max_pdf_size_mb * 1024 * 1024
        self._timeout_seconds = timeout_seconds
        self._http_client = http_client

    def fetch(self, source_uri: str, *, filename: str | None = None) -> FetchedDocument:
        parsed = urlparse(source_uri)
        if parsed.scheme not in ("https", "http"):
            raise SourceFetchError(f"Esquema de URL não suportado: {source_uri!r}")

        chunks: list[bytes] = []
        total_size = 0
        content_type = "application/pdf"

        try:
            client = self._http_client or httpx.Client(
                follow_redirects=True,
                timeout=self._timeout_seconds,
            )
            should_close = self._http_client is None
            try:
                with client.stream("GET", source_uri) as response:
                    response.raise_for_status()
                    content_type = response.headers.get("content-type", "application/pdf")
                    for chunk in response.iter_bytes(chunk_size=65536):
                        total_size += len(chunk)
                        if total_size > self._max_bytes:
                            raise InvalidPdfError(
                                f"PDF excede o limite máximo permitido de "
                                f"{self._max_bytes // (1024 * 1024)} MB."
                            )
                        chunks.append(chunk)
            finally:
                if should_close:
                    client.close()
        except InvalidPdfError:
            raise
        except Exception as exc:
            raise SourceFetchError(
                f"Falha ao baixar documento oficial de {source_uri}: {exc}"
            ) from exc

        pdf_bytes = b"".join(chunks)
        if not pdf_bytes.startswith(b"%PDF-"):
            raise InvalidPdfError(
                f"Arquivo obtido de {source_uri} não é um PDF válido (cabeçalho '%PDF-' ausente)."
            )

        derived_name = filename or Path(parsed.path).name or "document.pdf"
        file_hash = hashlib.sha256(pdf_bytes).hexdigest()
        return FetchedDocument(
            pdf_bytes=pdf_bytes,
            filename=derived_name,
            file_hash=file_hash,
            source_uri=source_uri,
            fetched_at=datetime.now(UTC),
            content_type=content_type,
        )
