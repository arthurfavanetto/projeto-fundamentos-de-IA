"""Teste de integração real do motor de persistência do ChromaDB 1.5.9 em disco."""

from __future__ import annotations

from pathlib import Path

import chromadb
import pytest

from rbac_assistant.domain.ports.vector_store_port import VectorRecord
from rbac_assistant.infrastructure.vector_db.chroma_vector_store import ChromaVectorStore


@pytest.mark.integration
def test_chromadb_159_write_read_and_persistence_across_reopen(tmp_path: Path) -> None:
    """Valida escrita, leitura, busca e persistência real do ChromaDB 1.5.9 após reabertura."""
    persist_dir = tmp_path / "chroma_data"
    persist_dir.mkdir()

    # Sessão 1: inicialização, escrita e leitura imediata
    client_session_1 = chromadb.PersistentClient(path=str(persist_dir))
    try:
        store_1 = ChromaVectorStore(client=client_session_1)

        record = VectorRecord(
            record_id="rbac_43_persist_001",
            document=(
                "43.3 Pessoas autorizadas a executar manutenção, "
                "manutenção preventiva e alteração."
            ),
            embedding=(0.25, 0.5, 0.75, 1.0),
            metadata={
                "regulation_number": "RBAC 43",
                "section_number": "43.3",
                "extraction_method": "native_pdf",
                "validation_status": "validated",
            },
        )
        store_1.upsert_records(collection_name="rbac_43_integration", records=[record])
        # Reexecução idempotente não deve duplicar
        store_1.upsert_records(collection_name="rbac_43_integration", records=[record])

        immediate = store_1.get_records(
            collection_name="rbac_43_integration",
            record_ids=["rbac_43_persist_001"],
        )
        assert len(immediate) == 1
        assert immediate[0].document == record.document
    finally:
        # Fecha explicitamente handles SQLite/ChromaDB antes de reabrir (evita lock no Windows)
        client_session_1.close()

    # Sessão 2: leitura e consulta semântica após reabertura do volume persistido
    client_session_2 = chromadb.PersistentClient(path=str(persist_dir))
    try:
        store_2 = ChromaVectorStore(client=client_session_2)

        persisted = store_2.get_records(
            collection_name="rbac_43_integration",
            record_ids=["rbac_43_persist_001"],
        )
        assert len(persisted) == 1
        assert persisted[0].record_id == "rbac_43_persist_001"
        assert persisted[0].document == record.document
        assert persisted[0].metadata["section_number"] == "43.3"

        hits = store_2.query_similar(
            collection_name="rbac_43_integration",
            query_embedding=(0.25, 0.5, 0.75, 1.0),
            top_k=3,
            where={"regulation_number": "RBAC 43"},
        )
        assert len(hits) == 1
        assert hits[0].record_id == "rbac_43_persist_001"

        store_2.delete_collection("rbac_43_integration")
    finally:
        client_session_2.close()
