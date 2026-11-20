"""Contrato (Port) para armazenamento persistente de PDFs originais e artefatos auxiliares."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol


class FileStorePort(Protocol):
    """Interface de persistência em disco no volume `regulatory_documents`."""

    def save_raw_pdf(
        self,
        *,
        authority: str,
        document_type: str,
        regulation_number: str,
        file_hash: str,
        pdf_bytes: bytes,
    ) -> Path:
        """Salva o PDF original de forma imutável e atômica e retorna seu caminho."""

    def get_raw_pdf_path(self, storage_path: str) -> Path:
        """Resolve e valida o caminho do PDF original armazenado."""

    def save_extracted_snapshot(
        self,
        *,
        version_id: str,
        payload: Mapping[str, Any],
    ) -> Path:
        """Persiste artefato JSON da extração para inspeção e auditoria em `extracted/`."""

    def save_review_record(
        self,
        *,
        document_id: str,
        review_id: str,
        payload: Mapping[str, Any],
    ) -> Path:
        """Exporta registro auditável de revisão humana em `reviews/`."""
