"""Testes offline completos de backup, verificação SHA-256, restauração e contratos de segurança."""

from __future__ import annotations

import json
import sqlite3
import tarfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from scripts import backup_restore as backup_cli

from rbac_assistant.domain.ports.audit_port import AuditEvent
from rbac_assistant.infrastructure.audit.jsonl_audit_adapter import JsonlAuditAdapter
from rbac_assistant.infrastructure.storage.backup_manager import (
    create_backup_bundle,
    restore_backup_bundle,
    verify_backup_bundle,
)
from rbac_assistant.infrastructure.storage.sqlite_document_repository import (
    SqliteDocumentRepository,
)


def _seed_sample_volumes(base_dir: Path) -> tuple[Path, Path, Path]:
    docs_dir = base_dir / "documents"
    chroma_dir = base_dir / "chroma"
    audit_dir = base_dir / "logs"

    docs_dir.mkdir(parents=True)
    chroma_dir.mkdir(parents=True)
    audit_dir.mkdir(parents=True)

    repo = SqliteDocumentRepository(docs_dir / "catalog.sqlite3")
    repo.initialize()
    repo.close()

    (docs_dir / "raw_pdfs").mkdir(parents=True, exist_ok=True)
    (docs_dir / "raw_pdfs" / "sample.pdf").write_bytes(b"%PDF-1.4\n%sample\n%%EOF\n")
    (chroma_dir / "chroma.sqlite3").write_bytes(b"simulated-chroma-index-bytes")

    audit = JsonlAuditAdapter(logs_dir=audit_dir)
    audit.record(
        AuditEvent(
            timestamp=datetime.now(UTC),
            action="backup_seed_event",
            status="ok",
            actor_id="system",
            details={"sample": True},
        )
    )
    return docs_dir, chroma_dir, audit_dir


@pytest.mark.unit
def test_backup_verify_and_restore_roundtrip_preserves_sqlite_pdf_chroma_and_audit(
    tmp_path: Path,
) -> None:
    docs_dir, chroma_dir, audit_dir = _seed_sample_volumes(tmp_path / "source")
    backup_root = tmp_path / "backups"

    ts1 = datetime(2026, 11, 23, 22, 0, 0, tzinfo=UTC)
    bundle_dir, manifest = create_backup_bundle(
        documents_dir=docs_dir,
        chroma_dir=chroma_dir,
        audit_logs_dir=audit_dir,
        backup_root=backup_root,
        timestamp_utc=ts1,
    )
    assert manifest.bundle_id == "rbac_backup_20261123T220000Z"
    assert manifest.created_at_utc.endswith("Z")
    assert manifest.sqlite_integrity_check == "ok"
    assert len(manifest.archives) == 3

    verification = verify_backup_bundle(bundle_dir)
    assert verification.valid is True
    assert set(verification.verified_archives) == {
        "regulatory_documents",
        "chroma_data",
        "audit_logs",
    }

    # Criar um segundo backup não remove o anterior e recusa sobrescrever o mesmo timestamp
    with pytest.raises(FileExistsError, match="não será sobrescrito"):
        create_backup_bundle(
            documents_dir=docs_dir,
            chroma_dir=chroma_dir,
            audit_logs_dir=audit_dir,
            backup_root=backup_root,
            timestamp_utc=ts1,
        )

    bundle_dir_2, _ = create_backup_bundle(
        documents_dir=docs_dir,
        chroma_dir=chroma_dir,
        audit_logs_dir=audit_dir,
        backup_root=backup_root,
        timestamp_utc=ts1 + timedelta(seconds=10),
    )
    assert bundle_dir.exists() and bundle_dir_2.exists()

    # Restauração exige confirm_restore=True
    restore_root = tmp_path / "restored"
    with pytest.raises(ValueError, match="confirm_restore=True"):
        restore_backup_bundle(
            bundle_dir=bundle_dir,
            target_documents_dir=restore_root / "documents",
            target_chroma_dir=restore_root / "chroma",
            target_audit_logs_dir=restore_root / "logs",
            confirm_restore=False,
        )

    restore_res = restore_backup_bundle(
        bundle_dir=bundle_dir,
        target_documents_dir=restore_root / "documents",
        target_chroma_dir=restore_root / "chroma",
        target_audit_logs_dir=restore_root / "logs",
        confirm_restore=True,
    )
    assert restore_res.restored is True
    assert restore_res.sqlite_integrity_check == "ok"
    assert restore_res.audit_jsonl_files_checked == 1
    assert (restore_root / "documents" / "raw_pdfs" / "sample.pdf").read_bytes().startswith(b"%PDF")
    restored_chroma = (restore_root / "chroma" / "chroma.sqlite3").read_bytes()
    assert restored_chroma == b"simulated-chroma-index-bytes"

    # Confirma que o banco SQLite restaurado possui as tabelas de migração intactas
    conn = sqlite3.connect(str(restore_root / "documents" / "catalog.sqlite3"))
    try:
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
        }
        assert "regulatory_documents" in tables
        assert "normative_chunks_fts" in tables
    finally:
        conn.close()


@pytest.mark.unit
def test_verify_and_restore_detect_sha256_tampering_and_tar_traversal(tmp_path: Path) -> None:
    docs_dir, chroma_dir, audit_dir = _seed_sample_volumes(tmp_path / "src")
    backup_root = tmp_path / "backups"
    bundle_dir, manifest = create_backup_bundle(
        documents_dir=docs_dir,
        chroma_dir=chroma_dir,
        audit_logs_dir=audit_dir,
        backup_root=backup_root,
    )

    # Corrompe 1 byte de um arquivo .tar.gz e confirma detecção imediata
    tampered = bundle_dir / manifest.archives[0].filename
    raw_bytes = bytearray(tampered.read_bytes())
    raw_bytes[-1] ^= 0xFF
    tampered.write_bytes(bytes(raw_bytes))

    ver = verify_backup_bundle(bundle_dir)
    assert ver.valid is False
    assert any("SHA-256 divergente" in err for err in ver.errors)

    target_root = tmp_path / "should_stay_untouched"
    with pytest.raises(ValueError, match="falha na verificação de integridade"):
        restore_backup_bundle(
            bundle_dir=bundle_dir,
            target_documents_dir=target_root / "documents",
            target_chroma_dir=target_root / "chroma",
            target_audit_logs_dir=target_root / "logs",
            confirm_restore=True,
        )
    assert not target_root.exists()

    # Path traversal dentro de tar com SHA-256 recalculado ainda é bloqueado
    evil_bundle = tmp_path / "evil_bundle"
    evil_bundle.mkdir()
    evil_tar = evil_bundle / "regulatory_documents_test.tar.gz"
    dummy_file = tmp_path / "dummy.txt"
    dummy_file.write_text("x", encoding="utf-8")
    with tarfile.open(evil_tar, "w:gz") as tar:
        tar.add(dummy_file, arcname="../escape.txt")
    from rbac_assistant.infrastructure.storage.backup_manager import compute_sha256

    evil_manifest = {
        "schema_version": "1.0",
        "bundle_id": "evil",
        "created_at_utc": "2026-11-23T22:00:00Z",
        "sqlite_integrity_check": "ok",
        "archives": [
            {
                "volume_key": "regulatory_documents",
                "filename": evil_tar.name,
                "size_bytes": evil_tar.stat().st_size,
                "sha256": compute_sha256(evil_tar),
                "member_count": 1,
            },
        ],
    }
    (evil_bundle / "manifest.json").write_text(json.dumps(evil_manifest), encoding="utf-8")
    evil_ver = verify_backup_bundle(evil_bundle)
    assert evil_ver.valid is False
    assert any("Caminho inseguro" in err for err in evil_ver.errors)


@pytest.mark.unit
def test_backup_restore_cli_end_to_end(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    docs_dir, chroma_dir, audit_dir = _seed_sample_volumes(tmp_path / "cli_src")
    backup_root = tmp_path / "cli_backups"

    assert (
        backup_cli.main(
            [
                "backup",
                "--documents-dir",
                str(docs_dir),
                "--chroma-dir",
                str(chroma_dir),
                "--audit-logs-dir",
                str(audit_dir),
                "--backup-root",
                str(backup_root),
                "--json",
            ]
        )
        == 0
    )
    created = json.loads(capsys.readouterr().out)
    bundle_dir = created["bundle_dir"]

    assert backup_cli.main(["verify", "--bundle-dir", bundle_dir, "--json"]) == 0
    verified = json.loads(capsys.readouterr().out)
    assert verified["valid"] is True

    dst = tmp_path / "cli_restored"
    assert (
        backup_cli.main(
            [
                "restore",
                "--bundle-dir",
                bundle_dir,
                "--documents-dir",
                str(dst / "docs"),
                "--chroma-dir",
                str(dst / "chroma"),
                "--audit-logs-dir",
                str(dst / "logs"),
                "--confirm-restore",
                "--json",
            ]
        )
        == 0
    )
    restored = json.loads(capsys.readouterr().out)
    assert restored["restored"] is True
    assert restored["sqlite_integrity_check"] == "ok"


@pytest.mark.unit
def test_powershell_backup_and_restore_scripts_forbid_destructive_commands(repo_root: Path) -> None:
    forbidden_tokens = (
        "down -v",
        "volume prune",
        "volume rm",
        "system prune",
        "rm -v",
        "builder prune",
        "wsl --unregister",
        "wsl --shutdown",
        ".vhdx",
        "n8n_data",
        "ollama",
        "open-webui",
    )
    for script_name in ("backup_docker_volumes.ps1", "restore_docker_volumes.ps1"):
        content = (repo_root / "scripts" / script_name).read_text(encoding="utf-8").lower()
        for token in forbidden_tokens:
            assert token not in content, (
                f"Comando ou referência proibida '{token}' encontrada em scripts/{script_name}"
            )
        assert "docker compose stop" in content
        assert "docker compose start" in content
        assert "sha256" in content
