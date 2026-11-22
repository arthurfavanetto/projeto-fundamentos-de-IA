"""Repositório SQLite (SSOT) com WAL, Foreign Keys, migrações versionadas e transações ACID."""

from __future__ import annotations

import json
import re
import sqlite3
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from rbac_assistant.domain.entities.document_version import DocumentVersion, ExtractionRun
from rbac_assistant.domain.entities.normative_unit import (
    ExtractionMethod,
    NormativeReference,
    NormativeTable,
    NormativeUnitType,
    StructureStatus,
)
from rbac_assistant.domain.entities.rag import (
    IndexRegistryEntry,
    NormativeChunk,
    SearchFilter,
)
from rbac_assistant.domain.entities.regulatory_document import (
    DocumentStatus,
    DocumentType,
    RegulatoryDocument,
    VersionStatus,
)
from rbac_assistant.domain.entities.regulatory_page import (
    ExtractionMetadata,
    PageValidationStatus,
    QualityMetrics,
    RegulatoryPage,
    ReviewDecision,
    ValidationRecord,
)
from rbac_assistant.domain.exceptions import DocumentNotFoundError
from rbac_assistant.domain.ports.validation_port import ParsedUnitRecord

_MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"

_PT_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "ao",
        "aos",
        "as",
        "com",
        "como",
        "da",
        "das",
        "de",
        "do",
        "dos",
        "e",
        "em",
        "na",
        "nas",
        "no",
        "nos",
        "o",
        "os",
        "ou",
        "para",
        "pela",
        "pelas",
        "pelo",
        "pelos",
        "por",
        "qual",
        "quais",
        "que",
        "quem",
        "se",
        "sobre",
        "um",
        "uma",
        "uns",
        "umas",
        "onde",
        "quando",
        "diz",
        "fala",
        "esta",
        "está",
        "sao",
        "são",
        "isso",
        "esse",
        "essa",
        "este",
        "neste",
        "nesta",
        "seu",
        "sua",
        "mais",
        "muito",
        "pode",
        "podem",
        "deve",
        "devem",
        "devo",
        "preciso",
        "saber",
        "depois",
        "antes",
        "após",
        "apos",
        "cada",
        "todo",
        "toda",
        "todos",
        "todas",
        "outro",
        "outra",
        "também",
        "tambem",
        "ainda",
        "apenas",
        "somente",
        "entre",
        "até",
        "ate",
        "ser",
        "estar",
        "ter",
        "foi",
        "seja",
        "tem",
        "têm",
        "novo",
        "nova",
    }
)
# Termos estruturais/boilerplate presentes em cabeçalhos de quase todos os chunks;
# omitidos da busca lexical FTS5 apenas quando existem termos específicos na consulta.
_FTS_GENERIC_HEADER_TERMS: frozenset[str] = frozenset(
    {
        "rbac",
        "anac",
        "seção",
        "secao",
        "emenda",
        "emd",
        "resolução",
        "resolucao",
        "portaria",
        "parágrafo",
        "paragrafo",
        "alínea",
        "alinea",
        "inciso",
        "item",
        "art",
        "artigo",
        "consulte",
        "prevê",
        "preve",
        "dispõe",
        "dispoe",
    }
)
_FTS_TOKEN_RE = re.compile(r"[0-9A-Za-zÀ-ÖØ-öø-ÿ]+(?:[.\-/][0-9A-Za-zÀ-ÖØ-öø-ÿ]+)*")
_FTS_ARTICLE_PHRASE_RE = re.compile(r"\b(?:Art\.?|Artigo)\s*(\d+)[º°]?", re.IGNORECASE)
_FTS_APPENDIX_PHRASE_RE = re.compile(r"\b(Ap[êe]ndice\s+[A-Z0-9]+)\b", re.IGNORECASE)


def _parse_date(value: str | None) -> date | None:
    return date.fromisoformat(value) if value else None


def _parse_datetime(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def _serialize_date(value: date | None) -> str | None:
    return value.isoformat() if value is not None else None


def _serialize_datetime(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _serialize_extraction_metadata(meta: ExtractionMetadata | None) -> str:
    if meta is None:
        return "{}"
    payload = {
        "extraction_method": meta.extraction_method.value,
        "extraction_timestamp": meta.extraction_timestamp.isoformat(),
        "extraction_version": meta.extraction_version,
        "content_hash": meta.content_hash,
        "page_width": meta.page_width,
        "page_height": meta.page_height,
        "table_count": meta.table_count,
        "image_count": meta.image_count,
        "quality_metrics": {
            "char_count": meta.quality_metrics.char_count,
            "line_count": meta.quality_metrics.line_count,
            "word_count": meta.quality_metrics.word_count,
            "has_replacement_chars": meta.quality_metrics.has_replacement_chars,
            "has_mojibake_signals": meta.quality_metrics.has_mojibake_signals,
            "has_portuguese_accents": meta.quality_metrics.has_portuguese_accents,
            "unbalanced_parentheses": meta.quality_metrics.unbalanced_parentheses,
            "is_empty_or_incomplete": meta.quality_metrics.is_empty_or_incomplete,
            "passed_automated_checks": meta.quality_metrics.passed_automated_checks,
            "issues": list(meta.quality_metrics.issues),
        },
    }
    return json.dumps(payload, ensure_ascii=False)


def _deserialize_extraction_metadata(raw_json: str) -> ExtractionMetadata | None:
    if not raw_json or raw_json == "{}":
        return None
    data = json.loads(raw_json)
    qm = data.get("quality_metrics")
    if not isinstance(qm, dict):
        return None
    metrics = QualityMetrics(
        char_count=int(qm.get("char_count", 0)),
        line_count=int(qm.get("line_count", 0)),
        word_count=int(qm.get("word_count", 0)),
        has_replacement_chars=bool(qm.get("has_replacement_chars", False)),
        has_mojibake_signals=bool(qm.get("has_mojibake_signals", False)),
        has_portuguese_accents=bool(qm.get("has_portuguese_accents", False)),
        unbalanced_parentheses=bool(qm.get("unbalanced_parentheses", False)),
        is_empty_or_incomplete=bool(qm.get("is_empty_or_incomplete", False)),
        passed_automated_checks=bool(qm.get("passed_automated_checks", False)),
        issues=tuple(str(item) for item in qm.get("issues", [])),
    )
    return ExtractionMetadata(
        extraction_method=ExtractionMethod(data["extraction_method"]),
        extraction_timestamp=datetime.fromisoformat(data["extraction_timestamp"]),
        extraction_version=str(data.get("extraction_version", "")),
        content_hash=str(data.get("content_hash", "")),
        quality_metrics=metrics,
        page_width=float(data.get("page_width", 0.0)),
        page_height=float(data.get("page_height", 0.0)),
        table_count=int(data.get("table_count", 0)),
        image_count=int(data.get("image_count", 0)),
    )


class SqliteDocumentRepository:
    """Implementação SSOT de `DocumentRepositoryPort` usando SQLite 3 (WAL + ACID)."""

    def __init__(
        self,
        db_path: Path,
        *,
        busy_timeout_ms: int = 5000,
        reuse_connection: bool = False,
    ) -> None:
        self._db_path = db_path
        self._busy_timeout_ms = busy_timeout_ms
        self._reuse_connection = reuse_connection
        self._shared_conn: sqlite3.Connection | None = None
        self._parent_ensured = False

    @property
    def db_path(self) -> Path:
        return self._db_path

    def close(self) -> None:
        """Fecha explicitamente a conexão reutilizada quando `reuse_connection=True`."""
        if self._shared_conn is not None:
            self._shared_conn.close()
            self._shared_conn = None

    def __enter__(self) -> SqliteDocumentRepository:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _open_connection(self) -> sqlite3.Connection:
        if not self._parent_ensured:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._parent_ensured = True
        conn = sqlite3.connect(
            self._db_path,
            timeout=self._busy_timeout_ms / 1000.0,
            isolation_level=None,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA synchronous = FULL;")
        conn.execute(f"PRAGMA busy_timeout = {self._busy_timeout_ms};")
        return conn

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        if self._reuse_connection:
            if self._shared_conn is None:
                self._shared_conn = self._open_connection()
            yield self._shared_conn
            return

        conn = self._open_connection()
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def _transaction(self, conn: sqlite3.Connection) -> Iterator[None]:
        conn.execute("BEGIN IMMEDIATE;")
        try:
            yield
            conn.execute("COMMIT;")
        except Exception:
            conn.execute("ROLLBACK;")
            raise

    def initialize(self) -> None:
        """Aplica WAL persistente e todas as migrações SQL pendentes de forma idempotente."""
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version TEXT PRIMARY KEY,
                    applied_at TEXT NOT NULL
                );
                """
            )
            applied = {
                str(row["version"])
                for row in conn.execute("SELECT version FROM schema_migrations;").fetchall()
            }
            for migration_file in sorted(_MIGRATIONS_DIR.glob("v*.sql")):
                version_name = migration_file.stem
                if version_name in applied:
                    continue
                sql_script = migration_file.read_text(encoding="utf-8")
                with self._transaction(conn):
                    for statement in self._split_sql_statements(sql_script):
                        conn.execute(statement)
                    conn.execute(
                        "INSERT INTO schema_migrations (version, applied_at) VALUES (?, ?);",
                        (version_name, datetime.now(UTC).isoformat()),
                    )

    def list_applied_migrations(self) -> list[str]:
        """Lista as versões de migração aplicadas em ordem crescente."""
        with self._connect() as conn:
            if not self._table_exists(conn, "schema_migrations"):
                return []
            rows = conn.execute(
                "SELECT version FROM schema_migrations ORDER BY version ASC;"
            ).fetchall()
        return [str(row["version"]) for row in rows]

    @staticmethod
    def _split_sql_statements(script: str) -> list[str]:
        statements: list[str] = []
        buffer: list[str] = []
        for line in script.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("--"):
                continue
            buffer.append(line)
            joined = "\n".join(buffer)
            if sqlite3.complete_statement(joined):
                statements.append(joined)
                buffer = []
        if buffer:
            rem = "\n".join(buffer).strip()
            if rem:
                statements.append(rem)
        return statements

    def upsert_document(self, document: RegulatoryDocument) -> RegulatoryDocument:
        """Cadastra ou atualiza um documento regulatório no catálogo."""
        now_iso = datetime.now(UTC).isoformat()
        with self._connect() as conn, self._transaction(conn):
            conn.execute(
                """
                INSERT INTO regulatory_documents (
                    document_id, authority, document_type, regulation_number, document_title,
                    document_status, version_status, amendment_number, publication_date,
                    effective_date, expiration_date, official_url, source_url, pdf_filename,
                    file_hash, imported_at, last_verified_at, subject_categories_json,
                    document_description, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(document_id) DO UPDATE SET
                    authority = excluded.authority,
                    document_type = excluded.document_type,
                    regulation_number = excluded.regulation_number,
                    document_title = excluded.document_title,
                    document_status = excluded.document_status,
                    version_status = excluded.version_status,
                    amendment_number = excluded.amendment_number,
                    publication_date = excluded.publication_date,
                    effective_date = excluded.effective_date,
                    expiration_date = excluded.expiration_date,
                    official_url = excluded.official_url,
                    source_url = excluded.source_url,
                    pdf_filename = excluded.pdf_filename,
                    file_hash = CASE
                        WHEN excluded.file_hash != '' THEN excluded.file_hash
                        ELSE regulatory_documents.file_hash
                    END,
                    imported_at = COALESCE(excluded.imported_at, regulatory_documents.imported_at),
                    last_verified_at = COALESCE(
                        excluded.last_verified_at, regulatory_documents.last_verified_at
                    ),
                    subject_categories_json = excluded.subject_categories_json,
                    document_description = excluded.document_description,
                    updated_at = excluded.updated_at;
                """,
                (
                    document.document_id,
                    document.authority,
                    document.document_type.value,
                    document.regulation_number,
                    document.document_title,
                    document.document_status.value,
                    document.version_status.value,
                    document.amendment_number,
                    _serialize_date(document.publication_date),
                    _serialize_date(document.effective_date),
                    _serialize_date(document.expiration_date),
                    document.official_url,
                    document.source_url,
                    document.pdf_filename,
                    document.file_hash,
                    _serialize_datetime(document.imported_at),
                    _serialize_datetime(document.last_verified_at),
                    json.dumps(list(document.subject_categories), ensure_ascii=False),
                    document.document_description,
                    now_iso,
                    now_iso,
                ),
            )
        stored = self.get_document(document.document_id)
        assert stored is not None
        return stored

    @staticmethod
    def _row_to_document(row: sqlite3.Row) -> RegulatoryDocument:
        categories = tuple(str(c) for c in json.loads(row["subject_categories_json"] or "[]"))
        return RegulatoryDocument(
            document_id=str(row["document_id"]),
            authority=str(row["authority"]),
            document_type=DocumentType(row["document_type"]),
            regulation_number=str(row["regulation_number"]),
            document_title=str(row["document_title"]),
            document_status=DocumentStatus(row["document_status"]),
            version_status=VersionStatus(row["version_status"]),
            amendment_number=str(row["amendment_number"]),
            publication_date=_parse_date(row["publication_date"]),
            effective_date=_parse_date(row["effective_date"]),
            expiration_date=_parse_date(row["expiration_date"]),
            official_url=str(row["official_url"]),
            source_url=str(row["source_url"]),
            pdf_filename=str(row["pdf_filename"]),
            file_hash=str(row["file_hash"]),
            imported_at=_parse_datetime(row["imported_at"]),
            last_verified_at=_parse_datetime(row["last_verified_at"]),
            subject_categories=categories,
            document_description=str(row["document_description"]),
        )

    def get_document(self, document_id: str) -> RegulatoryDocument | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM regulatory_documents WHERE document_id = ?;",
                (document_id,),
            ).fetchone()
        return self._row_to_document(row) if row is not None else None

    def search_documents(
        self,
        *,
        authority: str | None = None,
        document_type: DocumentType | None = None,
        regulation_number: str | None = None,
        subject: str | None = None,
        amendment_number: str | None = None,
        version_status: VersionStatus | None = None,
    ) -> list[RegulatoryDocument]:
        clauses: list[str] = []
        params: list[Any] = []
        if authority:
            clauses.append("LOWER(authority) = LOWER(?)")
            params.append(authority)
        if document_type:
            clauses.append("document_type = ?")
            params.append(document_type.value)
        if regulation_number:
            clauses.append("LOWER(regulation_number) = LOWER(?)")
            params.append(regulation_number)
        if amendment_number:
            clauses.append("LOWER(amendment_number) = LOWER(?)")
            params.append(amendment_number)
        if version_status:
            clauses.append("version_status = ?")
            params.append(version_status.value)
        if subject:
            clauses.append("LOWER(subject_categories_json) LIKE LOWER(?)")
            params.append(f"%{subject}%")

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM regulatory_documents {where_sql} ORDER BY document_id;",
                params,
            ).fetchall()
        return [self._row_to_document(row) for row in rows]

    @staticmethod
    def _row_to_version(row: sqlite3.Row) -> DocumentVersion:
        return DocumentVersion(
            version_id=str(row["version_id"]),
            document_id=str(row["document_id"]),
            amendment_number=str(row["amendment_number"]),
            resolution_number=str(row["resolution_number"]),
            publication_date=_parse_date(row["publication_date"]),
            effective_date=_parse_date(row["effective_date"]),
            expiration_date=_parse_date(row["expiration_date"]),
            version_status=VersionStatus(row["version_status"]),
            official_url=str(row["official_url"]),
            source_url=str(row["source_url"]),
            pdf_filename=str(row["pdf_filename"]),
            pdf_storage_path=str(row["pdf_storage_path"]),
            source_hash=str(row["source_hash"]),
            page_count=int(row["page_count"]),
            imported_at=_parse_datetime(row["imported_at"]),
            last_verified_at=_parse_datetime(row["last_verified_at"]),
        )

    def get_version(self, version_id: str) -> DocumentVersion | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM document_versions WHERE version_id = ?;",
                (version_id,),
            ).fetchone()
        return self._row_to_version(row) if row is not None else None

    def get_version_by_hash(
        self,
        *,
        document_id: str,
        amendment_number: str,
        source_hash: str,
    ) -> DocumentVersion | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM document_versions
                WHERE document_id = ? AND amendment_number = ? AND source_hash = ?;
                """,
                (document_id, amendment_number, source_hash),
            ).fetchone()
        return self._row_to_version(row) if row is not None else None

    def list_versions(self, document_id: str) -> list[DocumentVersion]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM document_versions
                WHERE document_id = ?
                ORDER BY COALESCE(effective_date, '') DESC, amendment_number DESC;
                """,
                (document_id,),
            ).fetchall()
        return [self._row_to_version(row) for row in rows]

    def list_all_versions(self) -> list[DocumentVersion]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM document_versions
                ORDER BY document_id ASC, COALESCE(effective_date, '') DESC, amendment_number DESC;
                """
            ).fetchall()
        return [self._row_to_version(row) for row in rows]

    def get_current_version(self, document_id: str) -> DocumentVersion | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM document_versions
                WHERE document_id = ? AND version_status = ?
                ORDER BY COALESCE(effective_date, '') DESC
                LIMIT 1;
                """,
                (document_id, VersionStatus.CURRENT.value),
            ).fetchone()
        return self._row_to_version(row) if row is not None else None

    def save_version_ingestion(
        self,
        *,
        document: RegulatoryDocument,
        version: DocumentVersion,
        pages: Sequence[RegulatoryPage],
        units: Sequence[ParsedUnitRecord],
        tables: Sequence[NormativeTable],
        extraction_run: ExtractionRun,
    ) -> list[ValidationRecord]:
        """Persiste atomicamente versão, páginas, unidades, tabelas e histórico.

        Invalida aprovações humanas anteriores caso alguma página tenha sofrido
        alteração de `content_hash` durante reprocessamento.
        """
        now = datetime.now(UTC)
        now_iso = now.isoformat()
        invalidation_records: list[ValidationRecord] = []

        with self._connect() as conn, self._transaction(conn):
            conn.execute(
                """
                INSERT INTO regulatory_documents (
                    document_id, authority, document_type, regulation_number, document_title,
                    document_status, version_status, amendment_number, publication_date,
                    effective_date, expiration_date, official_url, source_url, pdf_filename,
                    file_hash, imported_at, last_verified_at, subject_categories_json,
                    document_description, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(document_id) DO UPDATE SET
                    document_title = excluded.document_title,
                    document_status = excluded.document_status,
                    version_status = CASE
                        WHEN excluded.version_status = 'current' THEN excluded.version_status
                        WHEN regulatory_documents.version_status = 'unknown'
                            THEN excluded.version_status
                        ELSE regulatory_documents.version_status
                    END,
                    amendment_number = CASE
                        WHEN excluded.version_status = 'current' THEN excluded.amendment_number
                        WHEN regulatory_documents.amendment_number = ''
                            THEN excluded.amendment_number
                        ELSE regulatory_documents.amendment_number
                    END,
                    publication_date = COALESCE(
                        excluded.publication_date, regulatory_documents.publication_date
                    ),
                    effective_date = COALESCE(
                        excluded.effective_date, regulatory_documents.effective_date
                    ),
                    official_url = CASE
                        WHEN excluded.official_url != '' THEN excluded.official_url
                        ELSE regulatory_documents.official_url
                    END,
                    source_url = CASE
                        WHEN excluded.source_url != '' THEN excluded.source_url
                        ELSE regulatory_documents.source_url
                    END,
                    pdf_filename = CASE
                        WHEN excluded.pdf_filename != '' THEN excluded.pdf_filename
                        ELSE regulatory_documents.pdf_filename
                    END,
                    file_hash = CASE
                        WHEN excluded.file_hash != '' THEN excluded.file_hash
                        ELSE regulatory_documents.file_hash
                    END,
                    imported_at = COALESCE(excluded.imported_at, regulatory_documents.imported_at),
                    last_verified_at = COALESCE(
                        excluded.last_verified_at, regulatory_documents.last_verified_at
                    ),
                    subject_categories_json = excluded.subject_categories_json,
                    document_description = excluded.document_description,
                    updated_at = excluded.updated_at;
                """,
                (
                    document.document_id,
                    document.authority,
                    document.document_type.value,
                    document.regulation_number,
                    document.document_title,
                    document.document_status.value,
                    document.version_status.value,
                    document.amendment_number,
                    _serialize_date(document.publication_date),
                    _serialize_date(document.effective_date),
                    _serialize_date(document.expiration_date),
                    document.official_url,
                    document.source_url,
                    document.pdf_filename,
                    document.file_hash,
                    _serialize_datetime(document.imported_at),
                    _serialize_datetime(document.last_verified_at),
                    json.dumps(list(document.subject_categories), ensure_ascii=False),
                    document.document_description,
                    now_iso,
                    now_iso,
                ),
            )

            # Captura hashes e estados anteriores da mesma versão para regra de invalidação
            existing_pages = {
                int(row["pdf_page_number"]): (
                    str(row["page_id"]),
                    str(row["content_hash"]),
                    PageValidationStatus(row["validation_status"]),
                )
                for row in conn.execute(
                    """
                    SELECT page_id, pdf_page_number, content_hash, validation_status
                    FROM regulatory_pages
                    WHERE version_id = ?;
                    """,
                    (version.version_id,),
                ).fetchall()
            }

            conn.execute(
                """
                INSERT INTO document_versions (
                    version_id, document_id, amendment_number, resolution_number,
                    publication_date, effective_date, expiration_date, version_status,
                    official_url, source_url, pdf_filename, pdf_storage_path, source_hash,
                    page_count, imported_at, last_verified_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(version_id) DO UPDATE SET
                    resolution_number = excluded.resolution_number,
                    publication_date = excluded.publication_date,
                    effective_date = excluded.effective_date,
                    expiration_date = excluded.expiration_date,
                    version_status = excluded.version_status,
                    official_url = excluded.official_url,
                    source_url = excluded.source_url,
                    pdf_filename = excluded.pdf_filename,
                    pdf_storage_path = excluded.pdf_storage_path,
                    source_hash = excluded.source_hash,
                    page_count = excluded.page_count,
                    imported_at = excluded.imported_at,
                    last_verified_at = excluded.last_verified_at;
                """,
                (
                    version.version_id,
                    version.document_id,
                    version.amendment_number,
                    version.resolution_number,
                    _serialize_date(version.publication_date),
                    _serialize_date(version.effective_date),
                    _serialize_date(version.expiration_date),
                    version.version_status.value,
                    version.official_url,
                    version.source_url,
                    version.pdf_filename,
                    version.pdf_storage_path,
                    version.source_hash,
                    version.page_count,
                    _serialize_datetime(version.imported_at),
                    _serialize_datetime(version.last_verified_at),
                ),
            )

            # Remove unidades, tabelas e entradas FTS5 antigas da versão antes de reinserir
            if self._table_exists(conn, "normative_chunks_fts"):
                conn.execute(
                    "DELETE FROM normative_chunks_fts WHERE version_id = ?;",
                    (version.version_id,),
                )
            conn.execute("DELETE FROM normative_units WHERE version_id = ?;", (version.version_id,))
            conn.execute(
                "DELETE FROM normative_tables WHERE version_id = ?;", (version.version_id,)
            )

            page_id_by_number: dict[int, str] = {}
            page_status_by_number: dict[int, PageValidationStatus] = {}

            for page in pages:
                effective_status = page.validation_status
                previous = existing_pages.get(page.pdf_page_number)
                if previous is not None:
                    prev_page_id, prev_hash, prev_status = previous
                    was_human_approved = prev_status in (
                        PageValidationStatus.MANUALLY_APPROVED,
                        PageValidationStatus.APPROVED_FOR_LITERAL_TRANSCRIPTION,
                    )
                    if prev_hash == page.content_hash and was_human_approved:
                        effective_status = prev_status
                    elif prev_hash != page.content_hash and was_human_approved:
                        invalidation = ValidationRecord(
                            review_id=f"inv_{uuid.uuid4().hex[:12]}",
                            page_id=prev_page_id,
                            reviewer="system:reprocessing_guard",
                            reviewed_at=now,
                            decision=ReviewDecision.INVALIDATED_BY_REPROCESSING,
                            source_hash=page.content_hash,
                            comments=(
                                f"Aprovação anterior invalidada por alteração do hash da página "
                                f"{page.pdf_page_number}: {prev_hash[:12]} -> "
                                f"{page.content_hash[:12]}."
                            ),
                        )
                        invalidation_records.append(invalidation)

                conn.execute(
                    """
                    INSERT INTO regulatory_pages (
                        page_id, version_id, pdf_page_number, printed_page_number,
                        original_text, normalized_text, extraction_method, validation_status,
                        content_hash, extraction_quality_json, coordinates_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(version_id, pdf_page_number) DO UPDATE SET
                        printed_page_number = excluded.printed_page_number,
                        original_text = excluded.original_text,
                        normalized_text = excluded.normalized_text,
                        extraction_method = excluded.extraction_method,
                        validation_status = excluded.validation_status,
                        content_hash = excluded.content_hash,
                        extraction_quality_json = excluded.extraction_quality_json,
                        coordinates_json = excluded.coordinates_json;
                    """,
                    (
                        page.page_id,
                        page.version_id,
                        page.pdf_page_number,
                        page.printed_page_number,
                        page.original_text,
                        page.normalized_text,
                        page.extraction_method.value,
                        effective_status.value,
                        page.content_hash,
                        _serialize_extraction_metadata(page.extraction_metadata),
                        json.dumps(list(page.coordinates_summary), ensure_ascii=False),
                    ),
                )
                page_id_by_number[page.pdf_page_number] = (
                    previous[0] if previous is not None else page.page_id
                )
                page_status_by_number[page.pdf_page_number] = effective_status

            for inv in invalidation_records:
                conn.execute(
                    """
                    INSERT INTO validation_records (
                        review_id, page_id, reviewer, reviewed_at, decision, comments, source_hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        inv.review_id,
                        inv.page_id,
                        inv.reviewer,
                        inv.reviewed_at.isoformat(),
                        inv.decision.value,
                        inv.comments,
                        inv.source_hash,
                    ),
                )

            for unit in units:
                unit_page_statuses = [
                    page_status_by_number.get(p_num, PageValidationStatus.PENDING_REVIEW)
                    for p_num in unit.page_numbers
                ]
                unit_validation_status = self._aggregate_unit_status(unit_page_statuses)
                conn.execute(
                    """
                    INSERT INTO normative_units (
                        unit_id, version_id, parent_unit_id, unit_type, section_number,
                        subsection_number, paragraph_number, item_number, title, subpart,
                        original_text, normalized_text, start_page, end_page, order_index,
                        structure_status, validation_status, content_hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        unit.unit_id,
                        unit.version_id,
                        unit.parent_unit_id,
                        unit.unit_type.value,
                        unit.section_number,
                        unit.subsection_number,
                        unit.paragraph_number,
                        unit.item_number,
                        unit.title,
                        unit.subpart,
                        unit.original_text,
                        unit.normalized_text,
                        unit.start_page,
                        unit.end_page,
                        unit.order_index,
                        unit.structure_status.value,
                        unit_validation_status.value,
                        unit.content_hash,
                    ),
                )
                for seq_idx, p_num in enumerate(unit.page_numbers):
                    p_id = page_id_by_number.get(p_num)
                    if p_id is not None:
                        conn.execute(
                            """
                            INSERT OR IGNORE INTO normative_unit_pages (
                                unit_id, page_id, sequence_order
                            ) VALUES (?, ?, ?);
                            """,
                            (unit.unit_id, p_id, seq_idx),
                        )
                for ref in unit.references:
                    conn.execute(
                        """
                        INSERT INTO normative_references (
                            reference_id, unit_id, target_regulation, target_section, raw_citation
                        ) VALUES (?, ?, ?, ?, ?);
                        """,
                        (
                            ref.reference_id,
                            unit.unit_id,
                            ref.target_regulation,
                            ref.target_section,
                            ref.raw_citation,
                        ),
                    )

            for table in tables:
                conn.execute(
                    """
                    INSERT INTO normative_tables (
                        table_id, version_id, unit_id, pdf_page_number, caption, rows_json
                    ) VALUES (?, ?, ?, ?, ?, ?);
                    """,
                    (
                        table.table_id,
                        table.version_id,
                        table.unit_id,
                        table.pdf_page_number,
                        table.caption,
                        json.dumps([list(r) for r in table.rows], ensure_ascii=False),
                    ),
                )

            conn.execute(
                """
                INSERT INTO extraction_runs (
                    run_id, document_id, version_id, started_at, finished_at, status,
                    extraction_version, source_hash, pages_extracted, units_created, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    extraction_run.run_id,
                    extraction_run.document_id,
                    extraction_run.version_id,
                    extraction_run.started_at.isoformat(),
                    _serialize_datetime(extraction_run.finished_at),
                    extraction_run.status,
                    extraction_run.extraction_version,
                    extraction_run.source_hash,
                    extraction_run.pages_extracted,
                    extraction_run.units_created,
                    extraction_run.error_message,
                ),
            )

        return invalidation_records

    @staticmethod
    def _aggregate_unit_status(
        statuses: Sequence[PageValidationStatus],
    ) -> PageValidationStatus:
        if not statuses:
            return PageValidationStatus.PENDING_REVIEW
        if any(s == PageValidationStatus.REJECTED for s in statuses):
            return PageValidationStatus.REJECTED
        if any(
            s in (PageValidationStatus.OCR_EXTRACTED, PageValidationStatus.PENDING_MANUAL_REVIEW)
            for s in statuses
        ):
            return PageValidationStatus.PENDING_MANUAL_REVIEW
        if all(s == PageValidationStatus.APPROVED_FOR_LITERAL_TRANSCRIPTION for s in statuses):
            return PageValidationStatus.APPROVED_FOR_LITERAL_TRANSCRIPTION
        if all(
            s
            in (
                PageValidationStatus.NATIVE_QUALITY_CHECKED,
                PageValidationStatus.MANUALLY_APPROVED,
                PageValidationStatus.APPROVED_FOR_LITERAL_TRANSCRIPTION,
            )
            for s in statuses
        ):
            return PageValidationStatus.NATIVE_QUALITY_CHECKED
        return PageValidationStatus.PENDING_REVIEW

    def record_extraction_run(self, run: ExtractionRun) -> None:
        with self._connect() as conn, self._transaction(conn):
            conn.execute(
                """
                INSERT INTO extraction_runs (
                    run_id, document_id, version_id, started_at, finished_at, status,
                    extraction_version, source_hash, pages_extracted, units_created, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    run.run_id,
                    run.document_id,
                    run.version_id,
                    run.started_at.isoformat(),
                    _serialize_datetime(run.finished_at),
                    run.status,
                    run.extraction_version,
                    run.source_hash,
                    run.pages_extracted,
                    run.units_created,
                    run.error_message,
                ),
            )

    def list_extraction_runs(self, document_id: str) -> list[ExtractionRun]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM extraction_runs
                WHERE document_id = ?
                ORDER BY started_at DESC;
                """,
                (document_id,),
            ).fetchall()
        return [
            ExtractionRun(
                run_id=str(r["run_id"]),
                document_id=str(r["document_id"]),
                version_id=str(r["version_id"]) if r["version_id"] else None,
                started_at=datetime.fromisoformat(r["started_at"]),
                finished_at=_parse_datetime(r["finished_at"]),
                status=str(r["status"]),
                extraction_version=str(r["extraction_version"]),
                source_hash=str(r["source_hash"]),
                pages_extracted=int(r["pages_extracted"]),
                units_created=int(r["units_created"]),
                error_message=str(r["error_message"]),
            )
            for r in rows
        ]

    @staticmethod
    def _row_to_page(row: sqlite3.Row) -> RegulatoryPage:
        coords_raw = json.loads(row["coordinates_json"] or "[]")
        return RegulatoryPage(
            page_id=str(row["page_id"]),
            version_id=str(row["version_id"]),
            pdf_page_number=int(row["pdf_page_number"]),
            printed_page_number=str(row["printed_page_number"]),
            original_text=str(row["original_text"]),
            normalized_text=str(row["normalized_text"]),
            extraction_method=ExtractionMethod(row["extraction_method"]),
            validation_status=PageValidationStatus(row["validation_status"]),
            content_hash=str(row["content_hash"]),
            extraction_metadata=_deserialize_extraction_metadata(
                str(row["extraction_quality_json"])
            ),
            coordinates_summary=tuple(coords_raw),
        )

    def get_pages(
        self,
        version_id: str,
        *,
        page_numbers: Sequence[int] | None = None,
    ) -> list[RegulatoryPage]:
        with self._connect() as conn:
            if page_numbers:
                placeholders = ",".join("?" for _ in page_numbers)
                rows = conn.execute(
                    f"""
                    SELECT * FROM regulatory_pages
                    WHERE version_id = ? AND pdf_page_number IN ({placeholders})
                    ORDER BY pdf_page_number ASC;
                    """,
                    [version_id, *page_numbers],
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM regulatory_pages
                    WHERE version_id = ?
                    ORDER BY pdf_page_number ASC;
                    """,
                    (version_id,),
                ).fetchall()
        return [self._row_to_page(row) for row in rows]

    def get_page(self, page_id: str) -> RegulatoryPage | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM regulatory_pages WHERE page_id = ?;",
                (page_id,),
            ).fetchone()
        return self._row_to_page(row) if row is not None else None

    def update_page_validation(
        self,
        *,
        page_id: str,
        new_status: PageValidationStatus,
        review_record: ValidationRecord,
    ) -> RegulatoryPage:
        with self._connect() as conn, self._transaction(conn):
            existing = conn.execute(
                "SELECT version_id FROM regulatory_pages WHERE page_id = ?;",
                (page_id,),
            ).fetchone()
            if existing is None:
                raise DocumentNotFoundError(f"Página não encontrada: {page_id}")
            version_id = str(existing["version_id"])

            conn.execute(
                "UPDATE regulatory_pages SET validation_status = ? WHERE page_id = ?;",
                (new_status.value, page_id),
            )
            conn.execute(
                """
                INSERT INTO validation_records (
                    review_id, page_id, reviewer, reviewed_at, decision, comments, source_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    review_record.review_id,
                    review_record.page_id,
                    review_record.reviewer,
                    review_record.reviewed_at.isoformat(),
                    review_record.decision.value,
                    review_record.comments,
                    review_record.source_hash,
                ),
            )

            # Recalcula status agregado das unidades normativas vinculadas a esta página
            unit_rows = conn.execute(
                "SELECT unit_id FROM normative_unit_pages WHERE page_id = ?;",
                (page_id,),
            ).fetchall()
            for u_row in unit_rows:
                unit_id = str(u_row["unit_id"])
                p_statuses = [
                    PageValidationStatus(r["validation_status"])
                    for r in conn.execute(
                        """
                        SELECT rp.validation_status
                        FROM normative_unit_pages nup
                        JOIN regulatory_pages rp ON rp.page_id = nup.page_id
                        WHERE nup.unit_id = ? AND rp.version_id = ?;
                        """,
                        (unit_id, version_id),
                    ).fetchall()
                ]
                agg = self._aggregate_unit_status(p_statuses)
                conn.execute(
                    "UPDATE normative_units SET validation_status = ? WHERE unit_id = ?;",
                    (agg.value, unit_id),
                )
                if self._table_exists(conn, "normative_chunks"):
                    conn.execute(
                        "UPDATE normative_chunks SET validation_status = ? WHERE unit_id = ?;",
                        (agg.value, unit_id),
                    )

        updated = self.get_page(page_id)
        assert updated is not None
        return updated

    def list_validation_records(self, page_id: str) -> list[ValidationRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM validation_records
                WHERE page_id = ?
                ORDER BY reviewed_at DESC;
                """,
                (page_id,),
            ).fetchall()
        return [
            ValidationRecord(
                review_id=str(r["review_id"]),
                page_id=str(r["page_id"]),
                reviewer=str(r["reviewer"]),
                reviewed_at=datetime.fromisoformat(r["reviewed_at"]),
                decision=ReviewDecision(r["decision"]),
                comments=str(r["comments"]),
                source_hash=str(r["source_hash"]),
            )
            for r in rows
        ]

    def _hydrate_units(
        self,
        conn: sqlite3.Connection,
        rows: Sequence[sqlite3.Row],
    ) -> list[ParsedUnitRecord]:
        results: list[ParsedUnitRecord] = []
        for r in rows:
            unit_id = str(r["unit_id"])
            page_rows = conn.execute(
                """
                SELECT rp.pdf_page_number
                FROM normative_unit_pages nup
                JOIN regulatory_pages rp ON rp.page_id = nup.page_id
                WHERE nup.unit_id = ?
                ORDER BY nup.sequence_order ASC, rp.pdf_page_number ASC;
                """,
                (unit_id,),
            ).fetchall()
            ref_rows = conn.execute(
                "SELECT * FROM normative_references WHERE unit_id = ?;",
                (unit_id,),
            ).fetchall()
            page_numbers = tuple(int(pr["pdf_page_number"]) for pr in page_rows) or tuple(
                range(int(r["start_page"]), int(r["end_page"]) + 1)
            )
            refs = tuple(
                NormativeReference(
                    reference_id=str(rr["reference_id"]),
                    unit_id=unit_id,
                    target_regulation=str(rr["target_regulation"]),
                    target_section=str(rr["target_section"]),
                    raw_citation=str(rr["raw_citation"]),
                )
                for rr in ref_rows
            )
            results.append(
                ParsedUnitRecord(
                    unit_id=unit_id,
                    version_id=str(r["version_id"]),
                    parent_unit_id=str(r["parent_unit_id"]) if r["parent_unit_id"] else None,
                    unit_type=NormativeUnitType(r["unit_type"]),
                    section_number=str(r["section_number"]),
                    subsection_number=str(r["subsection_number"]),
                    paragraph_number=str(r["paragraph_number"]),
                    item_number=str(r["item_number"]),
                    title=str(r["title"]),
                    subpart=str(r["subpart"]),
                    original_text=str(r["original_text"]),
                    normalized_text=str(r["normalized_text"]),
                    start_page=int(r["start_page"]),
                    end_page=int(r["end_page"]),
                    page_numbers=page_numbers,
                    order_index=int(r["order_index"]),
                    structure_status=StructureStatus(r["structure_status"]),
                    content_hash=str(r["content_hash"]),
                    references=refs,
                )
            )
        return results

    def get_units_by_section(
        self,
        *,
        version_id: str,
        section_number: str,
    ) -> list[ParsedUnitRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM normative_units
                WHERE version_id = ? AND LOWER(section_number) = LOWER(?)
                ORDER BY order_index ASC;
                """,
                (version_id, section_number.strip()),
            ).fetchall()
            return self._hydrate_units(conn, rows)

    def list_units(self, version_id: str) -> list[ParsedUnitRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM normative_units
                WHERE version_id = ?
                ORDER BY order_index ASC;
                """,
                (version_id,),
            ).fetchall()
            return self._hydrate_units(conn, rows)

    def list_tables(
        self,
        version_id: str,
        *,
        page_numbers: Sequence[int] | None = None,
    ) -> list[NormativeTable]:
        with self._connect() as conn:
            if page_numbers:
                placeholders = ",".join("?" for _ in page_numbers)
                rows = conn.execute(
                    f"""
                    SELECT * FROM normative_tables
                    WHERE version_id = ? AND pdf_page_number IN ({placeholders})
                    ORDER BY pdf_page_number ASC, table_id ASC;
                    """,
                    [version_id, *page_numbers],
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM normative_tables
                    WHERE version_id = ?
                    ORDER BY pdf_page_number ASC, table_id ASC;
                    """,
                    (version_id,),
                ).fetchall()
        return [
            NormativeTable(
                table_id=str(r["table_id"]),
                version_id=str(r["version_id"]),
                unit_id=str(r["unit_id"]) if r["unit_id"] else None,
                pdf_page_number=int(r["pdf_page_number"]),
                caption=str(r["caption"]),
                rows=tuple(tuple(str(cell) for cell in row) for row in json.loads(r["rows_json"])),
            )
            for r in rows
        ]

    def list_references(self, unit_id: str) -> list[NormativeReference]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM normative_references WHERE unit_id = ?;",
                (unit_id,),
            ).fetchall()
        return [
            NormativeReference(
                reference_id=str(r["reference_id"]),
                unit_id=str(r["unit_id"]),
                target_regulation=str(r["target_regulation"]),
                target_section=str(r["target_section"]),
                raw_citation=str(r["raw_citation"]),
            )
            for r in rows
        ]

    @staticmethod
    def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type IN ('table', 'view') AND name = ?;",
            (table_name,),
        ).fetchone()
        return row is not None

    @staticmethod
    def _row_to_chunk(row: sqlite3.Row) -> NormativeChunk:
        pdf_pages = tuple(int(p) for p in json.loads(row["pdf_pages_json"] or "[]"))
        printed_pages = tuple(str(p) for p in json.loads(row["printed_pages_json"] or "[]"))
        return NormativeChunk(
            chunk_id=str(row["chunk_id"]),
            document_id=str(row["document_id"]),
            version_id=str(row["version_id"]),
            unit_id=str(row["unit_id"]),
            authority=str(row["authority"]),
            document_type=str(row["document_type"]),
            regulation_number=str(row["regulation_number"]),
            amendment_number=str(row["amendment_number"]),
            version_status=str(row["version_status"]),
            section_number=str(row["section_number"]),
            subsection_number=str(row["subsection_number"]),
            paragraph_number=str(row["paragraph_number"]),
            item_number=str(row["item_number"]),
            title=str(row["title"]),
            subpart=str(row["subpart"]),
            start_page=int(row["start_page"]),
            end_page=int(row["end_page"]),
            pdf_pages=pdf_pages,
            printed_pages=printed_pages,
            chunk_index=int(row["chunk_index"]),
            original_text=str(row["original_text"]),
            search_text=str(row["search_text"]),
            source_hash=str(row["source_hash"]),
            content_hash=str(row["content_hash"]),
            validation_status=str(row["validation_status"]),
            embedding_model=str(row["embedding_model"]),
            embedding_dimension=int(row["embedding_dimension"]),
        )

    @staticmethod
    def _build_chunk_filter_clauses(
        filters: SearchFilter | None,
        *,
        alias: str = "nc",
    ) -> tuple[list[str], list[Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if filters is None:
            return clauses, params

        if filters.document_id:
            clauses.append(f"{alias}.document_id = ?")
            params.append(filters.document_id)
        if filters.authority:
            clauses.append(f"LOWER({alias}.authority) = LOWER(?)")
            params.append(filters.authority)
        if filters.document_type is not None:
            clauses.append(f"{alias}.document_type = ?")
            params.append(filters.document_type.value)
        if filters.regulation_number:
            clauses.append(f"LOWER({alias}.regulation_number) = LOWER(?)")
            params.append(filters.regulation_number.strip())
        if filters.version_id:
            clauses.append(f"{alias}.version_id = ?")
            params.append(filters.version_id)
        if filters.amendment_number:
            amd_clean = filters.amendment_number.strip()
            if amd_clean.isdigit():
                amd_norm = str(int(amd_clean))
                clauses.append(
                    f"(LOWER({alias}.amendment_number) = LOWER(?) "
                    f"OR (LENGTH(LTRIM({alias}.amendment_number, '0')) > 0 "
                    f"AND LTRIM({alias}.amendment_number, '0') = ?) "
                    f"OR (LTRIM({alias}.amendment_number, '0') = '' AND ? = '0'))"
                )
                params.extend([amd_clean, amd_norm, amd_norm])
            else:
                clauses.append(f"LOWER({alias}.amendment_number) = LOWER(?)")
                params.append(amd_clean)
        elif filters.version_status is not None:
            clauses.append(f"{alias}.version_status = ?")
            params.append(filters.version_status.value)
        elif filters.only_current_version and not filters.version_id:
            clauses.append(f"{alias}.version_status = ?")
            params.append(VersionStatus.CURRENT.value)
        if filters.only_approved_literal:
            clauses.append(
                f"EXISTS (SELECT 1 FROM normative_unit_pages nup "
                f"WHERE nup.unit_id = {alias}.unit_id) "
                f"AND NOT EXISTS ("
                f"SELECT 1 FROM normative_unit_pages nup "
                f"JOIN regulatory_pages rp ON rp.page_id = nup.page_id "
                f"WHERE nup.unit_id = {alias}.unit_id "
                f"AND rp.validation_status != ?)"
            )
            params.append(PageValidationStatus.APPROVED_FOR_LITERAL_TRANSCRIPTION.value)
        return clauses, params

    @staticmethod
    def _extract_fts5_terms(
        query: str,
        *,
        filters: SearchFilter | None = None,
    ) -> list[str]:
        """Extrai termos/frases FTS5 preservando expressões estruturais e removendo ruído."""
        grouped_exprs: list[str] = []
        phrases: list[str] = []
        residual = query

        # 1. Preserva frases estruturais ("Art. 1º"/"Art. 1" e "Apêndice A") sem perder 1-char
        for match in _FTS_ARTICLE_PHRASE_RE.finditer(query):
            num = match.group(1)
            grouped_exprs.append(f'("Art {num}º" OR "Art {num}")')
        residual = _FTS_ARTICLE_PHRASE_RE.sub(" ", residual)

        for match in _FTS_APPENDIX_PHRASE_RE.finditer(residual):
            phrases.append(re.sub(r"\s+", " ", match.group(1)).strip())
        residual = _FTS_APPENDIX_PHRASE_RE.sub(" ", residual)

        # 2. Tokens já cobertos por filtros SQL explícitos (regulamento / emenda)
        already_filtered_nums: set[str] = set()
        if filters is not None:
            if filters.regulation_number:
                reg = filters.regulation_number.strip().lower()
                already_filtered_nums.add(reg)
            if filters.amendment_number:
                amd = filters.amendment_number.strip().lower()
                already_filtered_nums.add(amd)
                if amd.isdigit():
                    already_filtered_nums.add(str(int(amd)))
                    already_filtered_nums.add(f"{int(amd):02d}")

        raw_tokens = _FTS_TOKEN_RE.findall(residual)
        meaningful = [
            tok for tok in raw_tokens if len(tok) > 1 and tok.lower() not in _PT_STOPWORDS
        ]
        specific = [
            tok
            for tok in meaningful
            if tok.lower() not in _FTS_GENERIC_HEADER_TERMS
            and tok.lower() not in already_filtered_nums
        ]

        selected_plain: list[str] = (
            [*phrases, *specific]
            if (grouped_exprs or phrases or specific)
            else (meaningful or raw_tokens)
        )

        deduped: list[str] = list(dict.fromkeys(grouped_exprs))
        seen: set[str] = set()
        for term in selected_plain:
            key = term.lower()
            if key and key not in seen:
                seen.add(key)
                escaped = term.replace('"', '""')
                deduped.append(f'"{escaped}"')
        return deduped

    @classmethod
    def _build_fts5_match_query(
        cls,
        query: str,
        *,
        filters: SearchFilter | None = None,
    ) -> str:
        terms = cls._extract_fts5_terms(query, filters=filters)
        return " OR ".join(terms) if terms else ""

    def save_chunks(
        self,
        *,
        version_id: str,
        chunks: Sequence[NormativeChunk],
    ) -> list[str]:
        """Persiste chunks de forma idempotente, sincroniza FTS5 e retorna IDs removidos."""
        now_iso = datetime.now(UTC).isoformat()
        with self._connect() as conn, self._transaction(conn):
            existing_ids = [
                str(r["chunk_id"])
                for r in conn.execute(
                    "SELECT chunk_id FROM normative_chunks WHERE version_id = ?;",
                    (version_id,),
                ).fetchall()
            ]
            new_ids = {c.chunk_id for c in chunks}
            removed_ids = [cid for cid in existing_ids if cid not in new_ids]

            conn.execute("DELETE FROM normative_chunks_fts WHERE version_id = ?;", (version_id,))
            conn.execute("DELETE FROM normative_chunks WHERE version_id = ?;", (version_id,))

            for chunk in chunks:
                conn.execute(
                    """
                    INSERT INTO normative_chunks (
                        chunk_id, document_id, version_id, unit_id, authority, document_type,
                        regulation_number, amendment_number, version_status, section_number,
                        subsection_number, paragraph_number, item_number, title, subpart,
                        start_page, end_page, pdf_pages_json, printed_pages_json, chunk_index,
                        original_text, search_text, source_hash, content_hash, validation_status,
                        embedding_model, embedding_dimension, indexed_at
                    ) VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?
                    );
                    """,
                    (
                        chunk.chunk_id,
                        chunk.document_id,
                        chunk.version_id,
                        chunk.unit_id,
                        chunk.authority,
                        chunk.document_type,
                        chunk.regulation_number,
                        chunk.amendment_number,
                        chunk.version_status,
                        chunk.section_number,
                        chunk.subsection_number,
                        chunk.paragraph_number,
                        chunk.item_number,
                        chunk.title,
                        chunk.subpart,
                        chunk.start_page,
                        chunk.end_page,
                        json.dumps(list(chunk.pdf_pages)),
                        json.dumps(list(chunk.printed_pages), ensure_ascii=False),
                        chunk.chunk_index,
                        chunk.original_text,
                        chunk.search_text,
                        chunk.source_hash,
                        chunk.content_hash,
                        chunk.validation_status,
                        chunk.embedding_model,
                        chunk.embedding_dimension,
                        now_iso,
                    ),
                )
                conn.execute(
                    """
                    INSERT INTO normative_chunks_fts (
                        chunk_id, document_id, version_id, regulation_number, amendment_number,
                        section_number, subsection_number, paragraph_number, item_number, title,
                        search_text
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        chunk.chunk_id,
                        chunk.document_id,
                        chunk.version_id,
                        chunk.regulation_number,
                        chunk.amendment_number,
                        chunk.section_number,
                        chunk.subsection_number,
                        chunk.paragraph_number,
                        chunk.item_number,
                        chunk.title,
                        chunk.search_text,
                    ),
                )

        return removed_ids

    def list_chunks(self, version_id: str) -> list[NormativeChunk]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM normative_chunks
                WHERE version_id = ?
                ORDER BY start_page ASC, section_number ASC, chunk_index ASC, chunk_id ASC;
                """,
                (version_id,),
            ).fetchall()
        return [self._row_to_chunk(row) for row in rows]

    def get_chunks_by_ids(self, chunk_ids: Sequence[str]) -> list[NormativeChunk]:
        if not chunk_ids:
            return []
        placeholders = ",".join("?" for _ in chunk_ids)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM normative_chunks WHERE chunk_id IN ({placeholders});",
                list(chunk_ids),
            ).fetchall()
        by_id = {str(r["chunk_id"]): self._row_to_chunk(r) for r in rows}
        return [by_id[cid] for cid in chunk_ids if cid in by_id]

    def count_chunks(self, version_id: str | None = None) -> int:
        with self._connect() as conn:
            if version_id:
                row = conn.execute(
                    "SELECT COUNT(*) AS cnt FROM normative_chunks WHERE version_id = ?;",
                    (version_id,),
                ).fetchone()
            else:
                row = conn.execute("SELECT COUNT(*) AS cnt FROM normative_chunks;").fetchone()
        return int(row["cnt"]) if row is not None else 0

    def find_chunks_by_structure(
        self,
        *,
        section_number: str,
        regulation_number: str | None = None,
        paragraph_number: str | None = None,
        item_number: str | None = None,
        subitem_number: str | None = None,
        filters: SearchFilter | None = None,
        limit: int = 10,
    ) -> list[NormativeChunk]:
        clauses, params = self._build_chunk_filter_clauses(filters, alias="nc")
        clauses.append("LOWER(nc.section_number) = LOWER(?)")
        params.append(section_number.strip())

        if regulation_number and (not filters or not filters.regulation_number):
            clauses.append("LOWER(nc.regulation_number) = LOWER(?)")
            params.append(regulation_number.strip())
        if paragraph_number:
            clauses.append(
                "(LOWER(nc.paragraph_number) = LOWER(?) OR nc.paragraph_number = '')"
            )
            params.append(paragraph_number.strip())
        if item_number:
            clauses.append("(LOWER(nc.item_number) = LOWER(?) OR nc.item_number = '')")
            params.append(item_number.strip())

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT nc.*
                FROM normative_chunks nc
                {where_sql}
                ORDER BY
                    CASE WHEN LOWER(nc.paragraph_number) = LOWER(?) THEN 0 ELSE 1 END,
                    CASE WHEN nc.paragraph_number = '' THEN 0 ELSE 1 END,
                    nc.chunk_index ASC
                LIMIT ?;
                """,
                [*params, (paragraph_number or "").strip(), max(limit, 10)],
            ).fetchall()

        chunks = [self._row_to_chunk(row) for row in rows]
        if (paragraph_number or item_number or subitem_number) and len(chunks) > 1:
            composite_marker = "".join(
                part.lower() for part in (paragraph_number, item_number, subitem_number) if part
            )
            chunks.sort(
                key=lambda c: (
                    0 if (composite_marker and composite_marker in c.original_text.lower()) else 1,
                    c.chunk_index,
                )
            )
        return chunks[:limit]

    def _run_fts5_query(
        self,
        conn: sqlite3.Connection,
        *,
        match_expr: str,
        filter_clauses: Sequence[str],
        filter_params: Sequence[Any],
        limit: int,
    ) -> list[tuple[str, float]]:
        clauses = ["normative_chunks_fts MATCH ?", *filter_clauses]
        params: list[Any] = [match_expr, *filter_params, limit]
        where_sql = " AND ".join(clauses)
        rows = conn.execute(
            f"""
            SELECT
                nc.chunk_id,
                bm25(
                    normative_chunks_fts,
                    0.0, 0.0, 0.0, 3.0, 1.0, 6.0, 3.0, 2.0, 2.0, 4.0, 1.0
                ) AS rank_score
            FROM normative_chunks_fts
            JOIN normative_chunks nc ON nc.chunk_id = normative_chunks_fts.chunk_id
            WHERE {where_sql}
            ORDER BY rank_score ASC, nc.chunk_id ASC
            LIMIT ?;
            """,
            params,
        ).fetchall()
        # No SQLite FTS5, bm25() retorna valores negativos mais baixos para melhores resultados;
        # convertemos para escore positivo decrescente para maior clareza.
        return [(str(r["chunk_id"]), float(-r["rank_score"])) for r in rows]

    def search_chunks_fts(
        self,
        *,
        query: str,
        filters: SearchFilter | None = None,
        limit: int = 10,
    ) -> list[tuple[str, float]]:
        terms = self._extract_fts5_terms(query, filters=filters)
        if not terms:
            return []

        filter_clauses, filter_params = self._build_chunk_filter_clauses(filters, alias="nc")
        with self._connect() as conn:
            strict_hits: list[tuple[str, float]] = []
            # Estágio 1 (alta precisão): quando há 2..5 termos específicos, tenta conjunção AND
            if 2 <= len(terms) <= 5:
                strict_hits = self._run_fts5_query(
                    conn,
                    match_expr=" AND ".join(terms),
                    filter_clauses=filter_clauses,
                    filter_params=filter_params,
                    limit=limit,
                )

            broad_hits = self._run_fts5_query(
                conn,
                match_expr=" OR ".join(terms),
                filter_clauses=filter_clauses,
                filter_params=filter_params,
                limit=limit,
            )

        # No SQLite FTS5, termos presentes em >= 50% do corpus recebem IDF limitado a 1e-6
        # (gerando escore ~1e-6 sem poder discriminativo). Filtramos abaixo de 1e-3 sempre que
        # existirem candidatos com BM25 discriminativo real.
        min_informative_bm25 = 1e-3
        positive_strict = [(cid, sc) for cid, sc in strict_hits if sc > min_informative_bm25]
        positive_broad = [(cid, sc) for cid, sc in broad_hits if sc > min_informative_bm25]

        if positive_strict:
            merged: list[tuple[str, float]] = list(positive_strict)
            seen = {cid for cid, _ in merged}
            top_strict = merged[0][1]
            # Complementa com candidatos OR apenas se tiverem relevância lexical comparável
            for cid, sc in positive_broad:
                if cid in seen:
                    continue
                if sc >= 0.35 * top_strict:
                    merged.append((cid, sc))
                    seen.add(cid)
                if len(merged) >= limit:
                    break
            return merged[:limit]

        if positive_broad:
            return positive_broad[:limit]

        # Fallback para bancos minúsculos onde o único termo buscado ocorre em 100% dos chunks
        return (strict_hits or broad_hits)[:limit]

    def rebuild_fts_index(self, *, version_id: str | None = None) -> int:
        """Reconstrói atomicamente o índice derivado FTS5 a partir dos chunks do SQLite."""
        with self._connect() as conn, self._transaction(conn):
            if version_id:
                conn.execute(
                    "DELETE FROM normative_chunks_fts WHERE version_id = ?;",
                    (version_id,),
                )
                rows = conn.execute(
                    """
                    SELECT
                        chunk_id, document_id, version_id, regulation_number, amendment_number,
                        section_number, subsection_number, paragraph_number, item_number,
                        title, search_text
                    FROM normative_chunks
                    WHERE version_id = ?
                    ORDER BY chunk_id ASC;
                    """,
                    (version_id,),
                ).fetchall()
            else:
                conn.execute("DELETE FROM normative_chunks_fts;")
                rows = conn.execute(
                    """
                    SELECT
                        chunk_id, document_id, version_id, regulation_number, amendment_number,
                        section_number, subsection_number, paragraph_number, item_number,
                        title, search_text
                    FROM normative_chunks
                    ORDER BY chunk_id ASC;
                    """
                ).fetchall()

            for r in rows:
                conn.execute(
                    """
                    INSERT INTO normative_chunks_fts (
                        chunk_id, document_id, version_id, regulation_number, amendment_number,
                        section_number, subsection_number, paragraph_number, item_number, title,
                        search_text
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        str(r["chunk_id"]),
                        str(r["document_id"]),
                        str(r["version_id"]),
                        str(r["regulation_number"]),
                        str(r["amendment_number"]),
                        str(r["section_number"]),
                        str(r["subsection_number"]),
                        str(r["paragraph_number"]),
                        str(r["item_number"]),
                        str(r["title"]),
                        str(r["search_text"]),
                    ),
                )
        return len(rows)

    def upsert_index_registry(self, entry: IndexRegistryEntry) -> None:
        with self._connect() as conn, self._transaction(conn):
            conn.execute(
                """
                INSERT INTO vector_index_registry (
                    collection_name, version_id, document_id, source_hash,
                    embedding_model, embedding_dimension, chunk_count, indexed_at,
                    instruction_strategy_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(collection_name, version_id) DO UPDATE SET
                    document_id = excluded.document_id,
                    source_hash = excluded.source_hash,
                    embedding_model = excluded.embedding_model,
                    embedding_dimension = excluded.embedding_dimension,
                    chunk_count = excluded.chunk_count,
                    indexed_at = excluded.indexed_at,
                    instruction_strategy_version = excluded.instruction_strategy_version;
                """,
                (
                    entry.collection_name,
                    entry.version_id,
                    entry.document_id,
                    entry.source_hash,
                    entry.embedding_model,
                    entry.embedding_dimension,
                    entry.chunk_count,
                    entry.indexed_at.isoformat(),
                    entry.instruction_strategy_version,
                ),
            )

    def list_index_registry(
        self,
        *,
        collection_name: str | None = None,
    ) -> list[IndexRegistryEntry]:
        with self._connect() as conn:
            if collection_name:
                rows = conn.execute(
                    """
                    SELECT * FROM vector_index_registry
                    WHERE collection_name = ?
                    ORDER BY document_id ASC, version_id ASC;
                    """,
                    (collection_name,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM vector_index_registry
                    ORDER BY collection_name ASC, document_id ASC, version_id ASC;
                    """
                ).fetchall()
        return [
            IndexRegistryEntry(
                collection_name=str(r["collection_name"]),
                version_id=str(r["version_id"]),
                document_id=str(r["document_id"]),
                source_hash=str(r["source_hash"]),
                embedding_model=str(r["embedding_model"]),
                embedding_dimension=int(r["embedding_dimension"]),
                chunk_count=int(r["chunk_count"]),
                indexed_at=datetime.fromisoformat(r["indexed_at"]),
                instruction_strategy_version=str(r["instruction_strategy_version"] or ""),
            )
            for r in rows
        ]

    def delete_index_registry(
        self,
        *,
        collection_name: str,
        version_id: str | None = None,
    ) -> None:
        with self._connect() as conn, self._transaction(conn):
            if version_id:
                conn.execute(
                    """
                    DELETE FROM vector_index_registry
                    WHERE collection_name = ? AND version_id = ?;
                    """,
                    (collection_name, version_id),
                )
            else:
                conn.execute(
                    "DELETE FROM vector_index_registry WHERE collection_name = ?;",
                    (collection_name,),
                )

    def checkpoint_wal(self) -> None:
        """Executa checkpoint WAL (TRUNCATE) antes de operações de backup."""
        with self._connect() as conn:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")

    def backup_to(self, destination_path: Path) -> Path:
        """Executa checkpoint WAL e backup online consistente via API nativa do SQLite."""
        self.checkpoint_wal()
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as source_conn:
            dest_conn = sqlite3.connect(destination_path)
            try:
                source_conn.backup(dest_conn)
            finally:
                dest_conn.close()
        return destination_path
