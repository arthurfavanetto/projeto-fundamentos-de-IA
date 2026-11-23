"""CLI verificável de backup, validação SHA-256 e restauração dos dados do projeto.

Exemplos:
    uv run python scripts/backup_restore.py backup --documents-dir ... \\
        --chroma-dir ... --audit-logs-dir ... --backup-root ...
    uv run python scripts/backup_restore.py verify --bundle-dir <caminho_do_bundle>
    uv run python scripts/backup_restore.py restore --bundle-dir <caminho_do_bundle> \\
        --documents-dir ... --chroma-dir ... --audit-logs-dir ... --confirm-restore
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from rbac_assistant.infrastructure.storage.backup_manager import (
    create_backup_bundle,
    restore_backup_bundle,
    verify_backup_bundle,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Backup, verificação SHA-256 e restauração segura do RBAC AI Assistant."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_backup = sub.add_parser("backup", help="Cria bundle imutável com timestamp UTC e SHA-256.")
    p_backup.add_argument("--documents-dir", type=Path, required=True)
    p_backup.add_argument("--chroma-dir", type=Path, required=True)
    p_backup.add_argument("--audit-logs-dir", type=Path, required=True)
    p_backup.add_argument("--backup-root", type=Path, required=True)
    p_backup.add_argument("--json", action="store_true")

    p_verify = sub.add_parser("verify", help="Valida manifesto e checksums SHA-256 de um bundle.")
    p_verify.add_argument("--bundle-dir", type=Path, required=True)
    p_verify.add_argument("--json", action="store_true")

    p_restore = sub.add_parser(
        "restore",
        help="Valida integridade e restaura bundle nos diretórios alvo (exige --confirm-restore).",
    )
    p_restore.add_argument("--bundle-dir", type=Path, required=True)
    p_restore.add_argument("--documents-dir", type=Path, required=True)
    p_restore.add_argument("--chroma-dir", type=Path, required=True)
    p_restore.add_argument("--audit-logs-dir", type=Path, required=True)
    p_restore.add_argument("--confirm-restore", action="store_true")
    p_restore.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)

    if args.command == "backup":
        bundle_dir, manifest = create_backup_bundle(
            documents_dir=args.documents_dir,
            chroma_dir=args.chroma_dir,
            audit_logs_dir=args.audit_logs_dir,
            backup_root=args.backup_root,
        )
        payload = {"bundle_dir": str(bundle_dir), "manifest": asdict(manifest)}
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(f"[OK] Backup criado em: {bundle_dir}")
            print(
                f"     UTC: {manifest.created_at_utc} | "
                f"SQLite: {manifest.sqlite_integrity_check}"
            )
            for rec in manifest.archives:
                print(f"     - {rec.filename} ({rec.size_bytes} bytes) SHA-256={rec.sha256}")
        return 0

    if args.command == "verify":
        result = verify_backup_bundle(args.bundle_dir)
        if args.json:
            print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
        elif result.valid:
            print(
                f"[OK] Bundle '{result.bundle_id}' ({result.created_at_utc}) integro: "
                f"{', '.join(result.verified_archives)}"
            )
        else:
            print("[ERRO] Falha na verificação de integridade do backup:")
            for err in result.errors:
                print(f"  - {err}")
        return 0 if result.valid else 1

    if args.command == "restore":
        restore_res = restore_backup_bundle(
            bundle_dir=args.bundle_dir,
            target_documents_dir=args.documents_dir,
            target_chroma_dir=args.chroma_dir,
            target_audit_logs_dir=args.audit_logs_dir,
            confirm_restore=bool(args.confirm_restore),
        )
        if args.json:
            print(json.dumps(asdict(restore_res), ensure_ascii=False, indent=2))
        else:
            print(
                f"[OK] Restauração do bundle '{restore_res.bundle_id}' concluída "
                f"(SQLite: {restore_res.sqlite_integrity_check}, "
                f"arquivos JSONL verificados: {restore_res.audit_jsonl_files_checked})."
            )
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
