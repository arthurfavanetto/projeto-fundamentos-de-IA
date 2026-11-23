"""Testes offline completos do adaptador de auditoria persistente JSONL (`JsonlAuditAdapter`)."""

from __future__ import annotations

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from scripts import verify_audit_persistence

from rbac_assistant import main as main_module
from rbac_assistant.application.services.chat_service import NormativeChatService
from rbac_assistant.application.use_cases.check_system_status import CheckSystemStatusUseCase
from rbac_assistant.config.settings import AppSettings, StorageConfig
from rbac_assistant.domain.entities.rag import HybridRetrievalResult
from rbac_assistant.domain.exceptions import AuditWriteError
from rbac_assistant.domain.ports.audit_port import AuditEvent, AuditPort
from rbac_assistant.infrastructure.audit.jsonl_audit_adapter import JsonlAuditAdapter
from rbac_assistant.infrastructure.vector_db.chroma_vector_store import ChromaVectorStore


class _FakeHealthyVectorStore:
    def check_heartbeat(self) -> int:
        return 424242

    def upsert_records(self, collection_name: str, records: Any) -> None:
        return None

    def get_records(self, collection_name: str, record_ids: Any) -> list[Any]:
        return []

    def query_similar(
        self,
        collection_name: str,
        query_embedding: Any,
        *,
        top_k: int = 5,
        where: Any = None,
    ) -> list[Any]:
        return []

    def delete_collection(self, collection_name: str) -> None:
        return None


class _FakeEmptyRetriever:
    def retrieve(self, *, query: str, filters: Any = None, top_k: int | None = None) -> Any:
        return HybridRetrievalResult(
            query=query,
            evidences=(),
            blocked_evidences=(),
            warnings=(),
            index_empty=True,
            diagnostics=None,
        )


@pytest.mark.unit
def test_jsonl_audit_writes_valid_structure_and_utc_timestamp(tmp_path: Path) -> None:
    adapter: AuditPort = JsonlAuditAdapter(logs_dir=tmp_path)
    assert isinstance(adapter, JsonlAuditAdapter)

    # Timestamp com fuso BRT (-03:00) deve ser normalizado para UTC ISO-8601 com 'Z'
    brt = timezone(timedelta(hours=-3))
    local_ts = datetime(2026, 11, 23, 18, 30, 15, 123456, tzinfo=brt)
    adapter.record(
        AuditEvent(
            timestamp=local_ts,
            action="document_ingested",
            status="success",
            actor_id="operator_1",
            role="ADMIN",
            details={"document_id": "anac_rbac_43", "pages_extracted": 32, "verified": True},
        )
    )

    lines = adapter.audit_file_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["schema_version"] == "1.0"
    assert record["timestamp"] == "2026-11-23T21:30:15.123456Z"
    assert record["event_type"] == "document_ingested"
    assert record["outcome"] == "success"
    assert record["actor_id"] == "operator_1"
    assert record["role"] == "ADMIN"
    assert record["metadata"] == {
        "document_id": "anac_rbac_43",
        "pages_extracted": 32,
        "verified": True,
    }


@pytest.mark.unit
def test_jsonl_audit_redacts_secrets_and_blocks_sensitive_prompts_and_full_docs(
    tmp_path: Path,
) -> None:
    adapter = JsonlAuditAdapter(logs_dir=tmp_path, max_detail_string_length=80)
    fake_gemini = "AIzaSyFakeGeminiKeyNeverReal1234567890"
    fake_telegram = "987654321:AAFakeTelegramBotTokenNeverReal0123456789"
    fake_github = "ghp_FakeGitHubToken12345678901234567890"

    adapter.record(
        AuditEvent(
            timestamp=datetime.now(UTC),
            action="security_probe",
            status="blocked",
            actor_id=f"user_{fake_telegram}",
            details={
                "api_key": fake_gemini,
                "token": fake_telegram,
                "embedded_note": f"headers={fake_gemini} tg={fake_telegram} gh={fake_github}",
                "user_prompt": "pergunta confidencial do usuario",
                "system_prompt": "instrucao interna secreta",
                "original_text": "TEXTO INTEGRAL DO REGULAMENTO RBAC 43 QUE NAO PODE VAZAR",
                "document_text": "OUTRO CONTEUDO INTEGRAL",
                "long_safe_note": "A" * 200,
            },
        )
    )

    raw = adapter.audit_file_path.read_text(encoding="utf-8")
    assert fake_gemini not in raw
    assert fake_telegram not in raw
    assert fake_github not in raw
    assert "pergunta confidencial" not in raw
    assert "instrucao interna secreta" not in raw
    assert "TEXTO INTEGRAL DO REGULAMENTO" not in raw
    assert "OUTRO CONTEUDO INTEGRAL" not in raw

    ev = adapter.read_events()[0]
    assert "[REDACTED]" in ev["actor_id"]
    assert ev["metadata"]["api_key"] == "[REDACTED]"
    assert ev["metadata"]["token"] == "[REDACTED]"
    assert ev["metadata"]["user_prompt"] == "[REDACTED]"
    assert ev["metadata"]["system_prompt"] == "[REDACTED]"
    assert ev["metadata"]["original_text"] == "[REDACTED]"
    assert ev["metadata"]["document_text"] == "[REDACTED]"
    assert len(ev["metadata"]["long_safe_note"]) == 80
    assert ev["metadata"]["long_safe_note"].endswith("...")


@pytest.mark.unit
def test_jsonl_audit_rejects_disabling_redaction_and_invalid_filenames(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="não pode ser desabilitada"):
        JsonlAuditAdapter(logs_dir=tmp_path, redact_secrets=False)

    for bad_name in ("", "../escape.jsonl", "sub/audit.jsonl", r"sub\audit.jsonl"):
        with pytest.raises(ValueError, match="inválido"):
            JsonlAuditAdapter(logs_dir=tmp_path, filename=bad_name)


@pytest.mark.unit
def test_jsonl_audit_pseudonymizes_actor_id_deterministically(tmp_path: Path) -> None:
    adapter = JsonlAuditAdapter(logs_dir=tmp_path, pseudonymize_user_ids=True)
    adapter.record(
        AuditEvent(
            timestamp=datetime.now(UTC),
            action="telegram_command",
            status="allowed",
            actor_id="123456789",
            role="USER",
        )
    )
    ev = adapter.read_events()[0]
    assert ev["actor_id"].startswith("usr_")
    assert len(ev["actor_id"]) == 20
    assert "123456789" not in json.dumps(ev)


@pytest.mark.unit
def test_jsonl_audit_persists_across_subprocess_restart(tmp_path: Path) -> None:
    script = (
        "from datetime import UTC, datetime; "
        "from pathlib import Path; "
        "from rbac_assistant.domain.ports.audit_port import AuditEvent; "
        "from rbac_assistant.infrastructure.audit.jsonl_audit_adapter import JsonlAuditAdapter; "
        f"a = JsonlAuditAdapter(logs_dir=Path({str(tmp_path)!r})); "
        "a.record(AuditEvent(timestamp=datetime.now(UTC), action='proc_restart_test', "
        "status='persisted', actor_id='worker_1', details={'seq': 1}))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0

    # Nova instância após término do processo anterior lê o evento durável
    reloaded = JsonlAuditAdapter(logs_dir=tmp_path)
    events = reloaded.read_events()
    assert len(events) == 1
    assert events[0]["event_type"] == "proc_restart_test"
    assert events[0]["outcome"] == "persisted"
    assert events[0]["metadata"]["seq"] == 1


@pytest.mark.unit
def test_jsonl_audit_handles_concurrent_multithreaded_writes(tmp_path: Path) -> None:
    adapter = JsonlAuditAdapter(logs_dir=tmp_path)
    total_threads = 8
    per_thread = 15

    def _worker(worker_idx: int) -> None:
        for i in range(per_thread):
            adapter.record(
                AuditEvent(
                    timestamp=datetime.now(UTC),
                    action="concurrent_write",
                    status="ok",
                    actor_id=f"w{worker_idx}",
                    details={"worker": worker_idx, "seq": i},
                )
            )

    with ThreadPoolExecutor(max_workers=total_threads) as pool:
        futures = [pool.submit(_worker, idx) for idx in range(total_threads)]
        for fut in futures:
            fut.result()

    events = adapter.read_events()
    assert len(events) == total_threads * per_thread
    pairs = {(e["metadata"]["worker"], e["metadata"]["seq"]) for e in events}
    assert len(pairs) == total_threads * per_thread


@pytest.mark.unit
def test_jsonl_audit_rotates_and_enforces_retention_limit(tmp_path: Path) -> None:
    adapter = JsonlAuditAdapter(
        logs_dir=tmp_path,
        filename="rot_audit.jsonl",
        max_file_size_bytes=350,
        max_rotated_files=2,
    )
    for idx in range(12):
        adapter.record(
            AuditEvent(
                timestamp=datetime.now(UTC),
                action="rotation_test",
                status="ok",
                actor_id="system",
                details={"seq": idx, "note": f"evento_numero_{idx:03d}_preenchimento_audit"},
            )
        )

    assert adapter.audit_file_path.exists()
    assert (tmp_path / "rot_audit.jsonl.1").exists()
    assert (tmp_path / "rot_audit.jsonl.2").exists()
    assert not (tmp_path / "rot_audit.jsonl.3").exists()

    # Todos os arquivos ativos e rotacionados contêm exclusivamente linhas JSON válidas
    for path in (
        adapter.audit_file_path,
        tmp_path / "rot_audit.jsonl.1",
        tmp_path / "rot_audit.jsonl.2",
    ):
        for line in path.read_text(encoding="utf-8").splitlines():
            parsed = json.loads(line)
            assert parsed["event_type"] == "rotation_test"


@pytest.mark.unit
def test_jsonl_audit_raises_explicit_error_on_disk_or_permission_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Caso 1: logs_dir aponta para um arquivo regular em vez de diretório
    blocking_file = tmp_path / "not_a_dir"
    blocking_file.write_text("file", encoding="utf-8")
    bad_adapter = JsonlAuditAdapter(logs_dir=blocking_file)
    with pytest.raises(AuditWriteError, match="Falha ao persistir evento de auditoria"):
        bad_adapter.record(
            AuditEvent(
                timestamp=datetime.now(UTC),
                action="disk_fail_probe",
                status="ok",
            )
        )

    # Caso 2: erro de I/O (ex.: disco cheio / permissão negada) durante fsync
    adapter = JsonlAuditAdapter(logs_dir=tmp_path / "logs_fsync_fail")

    def _fail_fsync(_fd: int) -> None:
        raise OSError("No space left on device")

    monkeypatch.setattr("os.fsync", _fail_fsync)
    with pytest.raises(AuditWriteError, match="No space left on device"):
        adapter.record(
            AuditEvent(
                timestamp=datetime.now(UTC),
                action="fsync_fail_probe",
                status="ok",
            )
        )


@pytest.mark.unit
def test_check_system_status_fails_explicitly_when_audit_write_fails(tmp_path: Path) -> None:
    class _FailingAuditPort:
        def record(self, event: AuditEvent) -> None:
            raise AuditWriteError("erro simulado de escrita em /app/logs")

    use_case = CheckSystemStatusUseCase(
        vector_store=_FakeHealthyVectorStore(),
        documents_dir=tmp_path / "docs",
        audit_logs_dir=tmp_path / "logs",
        audit_port=_FailingAuditPort(),
    )
    report = use_case.execute()
    assert report.healthy is False
    assert report.audit_logs_dir_writable is False
    assert report.error is not None
    assert "erro simulado de escrita" in report.error


@pytest.mark.unit
def test_chat_service_audits_query_metadata_without_leaking_question_text(tmp_path: Path) -> None:
    adapter = JsonlAuditAdapter(logs_dir=tmp_path)
    chat = NormativeChatService(
        retriever=_FakeEmptyRetriever(),  # type: ignore[arg-type]
        llm=None,
        audit_port=adapter,
    )
    secret_question = (
        "Pergunta sigilosa com token 123456789:AAFakeSecretToken012345678901234567890"
    )
    chat.ask(question=secret_question, actor_id="tg_42", actor_role="USER")

    raw = adapter.audit_file_path.read_text(encoding="utf-8")
    assert "Pergunta sigilosa" not in raw
    assert "AAFakeSecretToken" not in raw
    ev = adapter.read_events()[0]
    assert ev["event_type"] == "normative_query"
    assert ev["outcome"] == "refused_empty_index"
    assert ev["actor_id"] == "tg_42"
    assert ev["role"] == "USER"
    assert ev["metadata"]["query_length"] == len(secret_question)


@pytest.mark.unit
def test_main_check_only_persists_startup_audit_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logs_dir = tmp_path / "logs"
    settings = AppSettings(
        _env_file=None,  # type: ignore[call-arg]
        storage=StorageConfig(
            documents_dir=tmp_path / "documents",
            audit_logs_dir=logs_dir,
        ),
    )
    monkeypatch.setattr(main_module, "load_settings", lambda: settings)
    monkeypatch.setattr(ChromaVectorStore, "check_heartbeat", lambda _self: 777)

    assert main_module.main(["--check-only"]) == 0
    adapter = JsonlAuditAdapter.from_settings(settings)
    events = adapter.read_events()
    assert len(events) == 1
    assert events[0]["event_type"] == "system_status_check"
    assert events[0]["outcome"] == "healthy"


@pytest.mark.unit
def test_verify_audit_persistence_script_end_to_end(tmp_path: Path) -> None:
    assert (
        verify_audit_persistence.main(["--phase", "write", "--audit-logs-dir", str(tmp_path)])
        == 0
    )
    assert (
        verify_audit_persistence.main(["--phase", "verify", "--audit-logs-dir", str(tmp_path)])
        == 0
    )
    assert (
        verify_audit_persistence.main(["--phase", "cleanup", "--audit-logs-dir", str(tmp_path)])
        == 0
    )
    assert not (tmp_path / verify_audit_persistence.SMOKE_AUDIT_FILENAME).exists()
