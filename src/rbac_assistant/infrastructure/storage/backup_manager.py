"""Gerenciador seguro e verificável de backup e restauração dos volumes do RBAC AI Assistant.

Garantias operacionais:
- Identifica cada backup com timestamp UTC ISO-8601 (`rbac_backup_YYYYMMDDTHHMMSSZ`).
- Nunca sobrescreve nem remove backups anteriores automaticamente.
- Executa cópia consistente do SQLite via `sqlite3.Connection.backup()` + `PRAGMA integrity_check`
  antes de empacotar `regulatory_documents`, preservando PDFs originais e revisões humanas.
- Empacota separadamente `regulatory_documents`, `chroma_data` e `audit_logs` em `.tar.gz`
  com manifesto `manifest.json` contendo SHA-256 e tamanho de cada arquivo.
- Valida integralmente SHA-256 e proteção contra path traversal antes de restaurar, exigindo
  `confirm_restore=True` e executando verificação pós-restauração (SQLite + JSONL).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import tarfile
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

BACKUP_SCHEMA_VERSION = "1.0"
EXPECTED_VOLUME_ARCHIVE_NAMES: tuple[str, ...] = (
    "regulatory_documents",
    "chroma_data",
    "audit_logs",
)


@dataclass(frozen=True, slots=True)
class ArchiveRecord:
    """Metadados criptográficos de um arquivo `.tar.gz` do bundle de backup."""

    volume_key: str
    filename: str
    size_bytes: int
    sha256: str
    member_count: int


@dataclass(frozen=True, slots=True)
class BackupManifest:
    """Manifesto auditável do conjunto completo de backup com timestamp UTC e SHA-256."""

    schema_version: str
    bundle_id: str
    created_at_utc: str
    sqlite_integrity_check: str
    archives: tuple[ArchiveRecord, ...]


@dataclass(frozen=True, slots=True)
class BackupVerificationResult:
    """Resultado determinístico da verificação de integridade de um bundle de backup."""

    valid: bool
    bundle_dir: str
    bundle_id: str | None
    created_at_utc: str | None
    verified_archives: tuple[str, ...]
    errors: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BackupRestoreResult:
    """Resultado da restauração segura e verificação pós-restauração."""

    restored: bool
    bundle_id: str
    sqlite_integrity_check: str
    audit_jsonl_files_checked: int
    restored_volumes: tuple[str, ...]


def compute_sha256(file_path: Path) -> str:
    """Calcula SHA-256 em streaming para evitar carga excessiva em memória."""
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        while True:
            chunk = handle.read(65536)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def check_sqlite_integrity(sqlite_path: Path) -> str:
    """Executa `PRAGMA integrity_check` somente leitura em um arquivo SQLite."""
    if not sqlite_path.is_file():
        return "not_present"
    conn = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    try:
        rows = conn.execute("PRAGMA integrity_check;").fetchall()
        results = [str(r[0]) for r in rows]
        return "ok" if results == ["ok"] else "; ".join(results)
    finally:
        conn.close()


def _create_consistent_sqlite_snapshot(source_db: Path, dest_db: Path) -> str:
    """Cria cópia consistente via API oficial `sqlite3.backup()` e valida integridade."""
    dest_db.parent.mkdir(parents=True, exist_ok=True)
    src_conn = sqlite3.connect(str(source_db))
    dst_conn = sqlite3.connect(str(dest_db))
    try:
        src_conn.backup(dst_conn)
        dst_conn.commit()
    finally:
        dst_conn.close()
        src_conn.close()
    integrity = check_sqlite_integrity(dest_db)
    if integrity != "ok":
        raise RuntimeError(f"Falha em PRAGMA integrity_check no snapshot SQLite: {integrity}")
    return integrity


def _pack_directory_to_targz(source_dir: Path, archive_path: Path) -> int:
    """Compacta conteúdo de `source_dir` em `.tar.gz` determinístico sem caminhos absolutos."""
    member_count = 0
    with tarfile.open(archive_path, mode="w:gz") as tar:
        if source_dir.exists():
            for path in sorted(source_dir.rglob("*")):
                rel = path.relative_to(source_dir)
                tar.add(path, arcname=rel.as_posix(), recursive=False)
                member_count += 1
    return member_count


def _stage_documents_snapshot(
    *,
    documents_dir: Path,
    staging_dir: Path,
    sqlite_filename: str,
) -> str:
    """Copia documentos preservando consistência transacional do SQLite (sem WAL solto)."""
    staging_dir.mkdir(parents=True, exist_ok=True)
    sqlite_integrity = "not_present"
    if not documents_dir.exists():
        return sqlite_integrity

    wal_sidecars = {f"{sqlite_filename}-wal", f"{sqlite_filename}-shm"}
    for item in sorted(documents_dir.iterdir()):
        if item.name == sqlite_filename and item.is_file():
            sqlite_integrity = _create_consistent_sqlite_snapshot(
                item,
                staging_dir / sqlite_filename,
            )
        elif item.name in wal_sidecars:
            # Snapshot criado com sqlite3.backup() já consolidou o estado do WAL
            continue
        elif item.is_dir():
            shutil.copytree(item, staging_dir / item.name)
        elif item.is_file():
            shutil.copy2(item, staging_dir / item.name)
    return sqlite_integrity


def create_backup_bundle(
    *,
    documents_dir: Path,
    chroma_dir: Path,
    audit_logs_dir: Path,
    backup_root: Path,
    sqlite_filename: str = "catalog.sqlite3",
    timestamp_utc: datetime | None = None,
) -> tuple[Path, BackupManifest]:
    """Gera bundle de backup imutável com timestamp UTC e manifesto SHA-256.

    Nunca remove backups anteriores nem sobrescreve um bundle já existente.
    """
    now_utc = (timestamp_utc or datetime.now(UTC)).astimezone(UTC)
    stamp = now_utc.strftime("%Y%m%dT%H%M%SZ")
    bundle_id = f"rbac_backup_{stamp}"
    bundle_dir = backup_root / bundle_id

    if bundle_dir.exists():
        raise FileExistsError(
            f"Diretório de backup já existe e não será sobrescrito: '{bundle_dir}'."
        )

    bundle_dir.mkdir(parents=True, exist_ok=False)

    records: list[ArchiveRecord] = []
    with tempfile.TemporaryDirectory(prefix="rbac_backup_stage_") as tmp_stage:
        docs_stage = Path(tmp_stage) / "regulatory_documents"
        sqlite_integrity = _stage_documents_snapshot(
            documents_dir=documents_dir,
            staging_dir=docs_stage,
            sqlite_filename=sqlite_filename,
        )

        sources: tuple[tuple[str, Path], ...] = (
            ("regulatory_documents", docs_stage),
            ("chroma_data", chroma_dir),
            ("audit_logs", audit_logs_dir),
        )
        for volume_key, src_dir in sources:
            filename = f"{volume_key}_{stamp}.tar.gz"
            archive_path = bundle_dir / filename
            member_count = _pack_directory_to_targz(src_dir, archive_path)
            records.append(
                ArchiveRecord(
                    volume_key=volume_key,
                    filename=filename,
                    size_bytes=archive_path.stat().st_size,
                    sha256=compute_sha256(archive_path),
                    member_count=member_count,
                )
            )

    manifest = BackupManifest(
        schema_version=BACKUP_SCHEMA_VERSION,
        bundle_id=bundle_id,
        created_at_utc=now_utc.isoformat().replace("+00:00", "Z"),
        sqlite_integrity_check=sqlite_integrity,
        archives=tuple(records),
    )
    manifest_path = bundle_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(asdict(manifest), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return bundle_dir, manifest


def _validate_tar_members_safe(archive_path: Path) -> list[str]:
    """Garante que o arquivo `.tar.gz` é legível e não contém path traversal nem symlinks."""
    errors: list[str] = []
    try:
        with tarfile.open(archive_path, mode="r:gz") as tar:
            for member in tar.getmembers():
                name = member.name
                parts = Path(name).parts
                if name.startswith("/") or name.startswith("\\") or ".." in parts:
                    errors.append(
                        f"Caminho inseguro detectado em '{archive_path.name}': {name!r}"
                    )
                if member.issym() or member.islnk():
                    errors.append(
                        f"Link simbólico/hardlink não permitido em '{archive_path.name}': {name!r}"
                    )
    except (OSError, tarfile.TarError) as exc:
        errors.append(f"Arquivo tar corrompido '{archive_path.name}': {exc}")
    return errors


def verify_backup_bundle(bundle_dir: Path) -> BackupVerificationResult:
    """Valida manifesto, presença, tamanho, SHA-256 e segurança tar de todos os arquivos."""
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.is_file():
        return BackupVerificationResult(
            valid=False,
            bundle_dir=str(bundle_dir),
            bundle_id=None,
            created_at_utc=None,
            verified_archives=(),
            errors=(f"Manifesto ausente: '{manifest_path}'.",),
        )

    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return BackupVerificationResult(
            valid=False,
            bundle_dir=str(bundle_dir),
            bundle_id=None,
            created_at_utc=None,
            verified_archives=(),
            errors=(f"Manifesto inválido em '{manifest_path}': {exc}",),
        )

    bundle_id = str(raw.get("bundle_id", ""))
    created_at_utc = str(raw.get("created_at_utc", ""))
    errors: list[str] = []
    verified: list[str] = []

    if not created_at_utc.endswith("Z"):
        errors.append(f"Timestamp UTC inválido no manifesto: {created_at_utc!r}")

    raw_archives = raw.get("archives")
    if not isinstance(raw_archives, list):
        errors.append("Campo 'archives' ausente ou inválido no manifesto.")
        raw_archives = []

    seen_keys: set[str] = set()
    for entry in raw_archives:
        if not isinstance(entry, dict):
            errors.append("Entrada inválida em 'archives'.")
            continue
        vol_key = str(entry.get("volume_key", ""))
        filename = str(entry.get("filename", ""))
        expected_sha = str(entry.get("sha256", "")).lower()
        expected_size = int(entry.get("size_bytes", -1))
        seen_keys.add(vol_key)

        archive_path = bundle_dir / filename
        if not archive_path.is_file():
            errors.append(f"Arquivo de backup ausente: '{filename}'.")
            continue

        actual_size = archive_path.stat().st_size
        if actual_size != expected_size:
            errors.append(
                f"Tamanho divergente em '{filename}': esperado {expected_size}, "
                f"obtido {actual_size}."
            )
            continue

        actual_sha = compute_sha256(archive_path).lower()
        if actual_sha != expected_sha:
            errors.append(
                f"SHA-256 divergente em '{filename}': esperado {expected_sha}, obtido {actual_sha}."
            )
            continue

        tar_errors = _validate_tar_members_safe(archive_path)
        if tar_errors:
            errors.extend(tar_errors)
            continue

        verified.append(vol_key)

    missing_volumes = [k for k in EXPECTED_VOLUME_ARCHIVE_NAMES if k not in seen_keys]
    if missing_volumes:
        errors.append(f"Volume(s) obrigatório(s) ausente(s) no manifesto: {missing_volumes}.")

    return BackupVerificationResult(
        valid=len(errors) == 0,
        bundle_dir=str(bundle_dir),
        bundle_id=bundle_id or None,
        created_at_utc=created_at_utc or None,
        verified_archives=tuple(verified),
        errors=tuple(errors),
    )


def _extract_safe_targz(archive_path: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    resolved_root = target_dir.resolve()
    with tarfile.open(archive_path, mode="r:gz") as tar:
        for member in tar.getmembers():
            dest = (target_dir / member.name).resolve()
            if resolved_root not in dest.parents and dest != resolved_root:
                raise ValueError(f"Path traversal bloqueado durante restauração: {member.name!r}")
            if member.issym() or member.islnk():
                raise ValueError(f"Symlink bloqueado durante restauração: {member.name!r}")
        tar.extractall(path=target_dir, filter="data")


def restore_backup_bundle(
    *,
    bundle_dir: Path,
    target_documents_dir: Path,
    target_chroma_dir: Path,
    target_audit_logs_dir: Path,
    sqlite_filename: str = "catalog.sqlite3",
    confirm_restore: bool = False,
) -> BackupRestoreResult:
    """Restaura bundle após verificação integral de SHA-256 e valida estado pós-restauração."""
    if not confirm_restore:
        raise ValueError(
            "Restauração bloqueada: confirme explicitamente com `confirm_restore=True` "
            "(ou `--confirm-restore` na CLI)."
        )

    verification = verify_backup_bundle(bundle_dir)
    if not verification.valid:
        raise ValueError(
            "Restauração abortada — falha na verificação de integridade do backup: "
            + "; ".join(verification.errors)
        )

    raw_manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    target_map = {
        "regulatory_documents": target_documents_dir,
        "chroma_data": target_chroma_dir,
        "audit_logs": target_audit_logs_dir,
    }
    restored_volumes: list[str] = []
    for entry in raw_manifest["archives"]:
        vol_key = str(entry["volume_key"])
        filename = str(entry["filename"])
        if vol_key in target_map:
            _extract_safe_targz(bundle_dir / filename, target_map[vol_key])
            restored_volumes.append(vol_key)

    # Verificação pós-restauração: SQLite integrity_check + integridade sintática dos JSONL
    sqlite_check = check_sqlite_integrity(target_documents_dir / sqlite_filename)
    if sqlite_check not in {"ok", "not_present"}:
        raise RuntimeError(f"Falha de integridade SQLite pós-restauração: {sqlite_check}")

    jsonl_checked = 0
    if target_audit_logs_dir.exists():
        for jsonl_file in sorted(target_audit_logs_dir.glob("*.jsonl")):
            for line in jsonl_file.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    json.loads(line)
            jsonl_checked += 1

    return BackupRestoreResult(
        restored=True,
        bundle_id=str(raw_manifest["bundle_id"]),
        sqlite_integrity_check=sqlite_check,
        audit_jsonl_files_checked=jsonl_checked,
        restored_volumes=tuple(restored_volumes),
    )
