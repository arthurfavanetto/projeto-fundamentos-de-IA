"""Testes unitários da infraestrutura ChromaDB 1.5.9, healthcheck e contratos Docker."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml

from rbac_assistant.application.use_cases.check_system_status import CheckSystemStatusUseCase
from rbac_assistant.domain.exceptions import StorageConnectionError
from rbac_assistant.domain.ports.vector_store_port import VectorRecord
from rbac_assistant.infrastructure.vector_db.chroma_vector_store import ChromaVectorStore


@pytest.mark.unit
def test_chroma_heartbeat_v2_success() -> None:
    captured_urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_urls.append(str(request.url))
        return httpx.Response(200, json={"nanosecond heartbeat": 1763664000000000000})

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    store = ChromaVectorStore(
        host="chromadb",
        port=8000,
        heartbeat_path="/api/v2/heartbeat",
        http_client=http_client,
    )

    heartbeat = store.check_heartbeat()
    assert heartbeat == 1763664000000000000
    assert captured_urls == ["http://chromadb:8000/api/v2/heartbeat"]


@pytest.mark.unit
def test_chroma_heartbeat_raises_storage_connection_error_on_http_failure() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "unavailable"})

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    store = ChromaVectorStore(host="chromadb", port=8000, http_client=http_client)

    with pytest.raises(StorageConnectionError, match="/api/v2/heartbeat"):
        store.check_heartbeat()


@pytest.mark.unit
def test_chroma_upsert_read_query_and_delete_are_idempotent(
    fake_chroma_client: Any,
) -> None:
    store = ChromaVectorStore(client=fake_chroma_client)
    record = VectorRecord(
        record_id="rbac_43_43.3_0",
        document="43.3 Pessoas autorizadas a executar manutenção.",
        embedding=(0.1, 0.2, 0.3),
        metadata={"regulation_number": "RBAC 43", "section_number": "43.3"},
    )

    # Dupla execução idempotente (sem duplicação)
    store.upsert_records(collection_name="rbac_test", records=[record])
    store.upsert_records(collection_name="rbac_test", records=[record])

    loaded = store.get_records(collection_name="rbac_test", record_ids=["rbac_43_43.3_0"])
    assert len(loaded) == 1
    assert loaded[0].document == record.document
    assert loaded[0].metadata["section_number"] == "43.3"

    hits = store.query_similar(
        collection_name="rbac_test",
        query_embedding=(0.1, 0.2, 0.3),
        top_k=5,
        where={"regulation_number": "RBAC 43"},
    )
    assert len(hits) == 1
    assert hits[0].record_id == "rbac_43_43.3_0"

    store.delete_collection("rbac_test")
    assert "rbac_test" not in fake_chroma_client.collections


@pytest.mark.unit
def test_check_system_status_use_case_reports_healthy_and_unhealthy(tmp_path: Path) -> None:
    docs_dir = tmp_path / "documents"
    logs_dir = tmp_path / "logs"

    def ok_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"nanosecond heartbeat": 42})

    store_ok = ChromaVectorStore(
        http_client=httpx.Client(transport=httpx.MockTransport(ok_handler))
    )
    report_ok = CheckSystemStatusUseCase(
        vector_store=store_ok,
        documents_dir=docs_dir,
        audit_logs_dir=logs_dir,
    ).execute()

    assert report_ok.healthy is True
    assert report_ok.chroma_healthy is True
    assert report_ok.chroma_heartbeat == 42
    assert report_ok.documents_dir_writable is True
    assert report_ok.audit_logs_dir_writable is True
    assert report_ok.error is None

    def fail_handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    store_fail = ChromaVectorStore(
        http_client=httpx.Client(transport=httpx.MockTransport(fail_handler))
    )
    report_fail = CheckSystemStatusUseCase(
        vector_store=store_fail,
        documents_dir=docs_dir,
        audit_logs_dir=logs_dir,
    ).execute()

    assert report_fail.healthy is False
    assert report_fail.chroma_healthy is False
    assert report_fail.error is not None


@pytest.mark.unit
def test_docker_contracts_enforce_pinning_non_root_and_network_isolation(repo_root: Path) -> None:
    chroma_df = (repo_root / "docker" / "chromadb.Dockerfile").read_text(encoding="utf-8")
    assert "FROM chromadb/chroma:1.5.9" in chroma_df
    assert ":latest" not in chroma_df
    assert "10001" in chroma_df
    assert "USER 10001:10001" in chroma_df
    assert "ENTRYPOINT" not in chroma_df.replace(
        "# Preserva ENTRYPOINT", ""
    ), "chromadb.Dockerfile não deve sobrescrever o ENTRYPOINT oficial."

    app_df = (repo_root / "Dockerfile").read_text(encoding="utf-8")
    assert "FROM python:3.12-slim-bookworm AS builder" in app_df
    assert "FROM python:3.12-slim-bookworm AS runtime" in app_df
    assert "USER 10001:10001" in app_df
    assert "rbac_assistant.healthcheck" in app_df

    compose = yaml.safe_load((repo_root / "docker-compose.yml").read_text(encoding="utf-8"))
    chroma_svc = compose["services"]["chromadb"]
    bot_svc = compose["services"]["rbac-bot"]

    assert "ports" not in chroma_svc
    assert "ports" not in bot_svc
    assert "chroma_data:/data" in chroma_svc["volumes"]
    assert "./config:/app/config:ro" in bot_svc["volumes"]
    assert chroma_svc["networks"] == ["rbac_internal_net"]
    assert compose["networks"]["rbac_internal_net"]["internal"] is True
    assert "/api/v2/heartbeat" in " ".join(chroma_svc["healthcheck"]["test"])
    assert chroma_svc["restart"] == "unless-stopped"
    assert bot_svc["restart"] == "unless-stopped"
    assert bot_svc["depends_on"]["chromadb"]["condition"] == "service_healthy"
    for svc in (chroma_svc, bot_svc):
        assert svc["logging"]["driver"] == "json-file"
        assert svc["logging"]["options"]["max-size"] == "10m"
        assert svc["logging"]["options"]["max-file"] == "5"
