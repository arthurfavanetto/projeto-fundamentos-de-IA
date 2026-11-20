"""Armazenamento persistente atômico em disco para PDFs originais, extrações e revisões."""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from rbac_assistant.domain.exceptions import InvalidPdfError

_SAFE_SEGMENT = re.compile(r"[^A-Za-z0-9._-]+")


def _sanitize_segment(segment: str) -> str:
    cleaned = _SAFE_SEGMENT.sub("_", segment.strip()).strip("._")
    if not cleaned:
        raise ValueError(f"Segmento de caminho inválido: {segment!r}")
    return cleaned.lower()


def _atomic_write_bytes(target_path: Path, data: bytes) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=target_path.parent, prefix=".tmp_")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, target_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


class LocalFileDocumentStore:
    """Implementação de `FileStorePort` sobre o volume persistente `regulatory_documents`."""

    def __init__(
        self,
        *,
        documents_dir: Path,
        raw_pdfs_subdir: str = "raw_pdfs",
        extracted_subdir: str = "extracted",
        reviews_subdir: str = "reviews",
    ) -> None:
        self._documents_dir = documents_dir.resolve()
        self._raw_pdfs_dir = self._documents_dir / raw_pdfs_subdir
        self._extracted_dir = self._documents_dir / extracted_subdir
        self._reviews_dir = self._documents_dir / reviews_subdir

    @property
    def raw_pdfs_dir(self) -> Path:
        return self._raw_pdfs_dir

    @property
    def extracted_dir(self) -> Path:
        return self._extracted_dir

    @property
    def reviews_dir(self) -> Path:
        return self._reviews_dir

    def ensure_directories(self) -> None:
        """Garante existência dos subdiretórios persistentes no volume."""
        self._raw_pdfs_dir.mkdir(parents=True, exist_ok=True)
        self._extracted_dir.mkdir(parents=True, exist_ok=True)
        self._reviews_dir.mkdir(parents=True, exist_ok=True)

    def save_raw_pdf(
        self,
        *,
        authority: str,
        document_type: str,
        regulation_number: str,
        file_hash: str,
        pdf_bytes: bytes,
    ) -> Path:
        """Persiste o PDF original de forma imutável e atômica indexado pelo SHA-256."""
        if not pdf_bytes.startswith(b"%PDF-"):
            raise InvalidPdfError("Conteúdo binário não possui assinatura mágica '%PDF-'.")

        safe_auth = _sanitize_segment(authority)
        safe_type = _sanitize_segment(document_type)
        safe_num = _sanitize_segment(regulation_number)
        safe_hash = _sanitize_segment(file_hash)

        target = self._raw_pdfs_dir / safe_auth / safe_type / safe_num / f"{safe_hash}.pdf"
        if not target.exists():
            _atomic_write_bytes(target, pdf_bytes)
        return target

    def get_raw_pdf_path(self, storage_path: str) -> Path:
        """Valida e resolve o caminho do PDF original evitando path traversal."""
        candidate = Path(storage_path).resolve()
        try:
            candidate.relative_to(self._documents_dir)
        except ValueError as exc:
            raise InvalidPdfError(
                f"Caminho fora do diretório permitido de documentos: {storage_path}"
            ) from exc
        if not candidate.is_file():
            raise InvalidPdfError(f"PDF original não encontrado em: {candidate}")
        return candidate

    def save_extracted_snapshot(
        self,
        *,
        version_id: str,
        payload: Mapping[str, Any],
    ) -> Path:
        """Persiste snapshot JSON da extração em `extracted/` para auditoria."""
        safe_id = _sanitize_segment(version_id)
        target = self._extracted_dir / f"{safe_id}.json"
        encoded = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        _atomic_write_bytes(target, encoded)
        return target

    def save_review_record(
        self,
        *,
        document_id: str,
        review_id: str,
        payload: Mapping[str, Any],
    ) -> Path:
        """Exporta registro auditável de revisão humana em `reviews/`."""
        safe_doc = _sanitize_segment(document_id)
        safe_review = _sanitize_segment(review_id)
        target = self._reviews_dir / safe_doc / f"{safe_review}.json"
        encoded = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        _atomic_write_bytes(target, encoded)
        return target
