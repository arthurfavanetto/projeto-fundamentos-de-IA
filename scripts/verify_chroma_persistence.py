"""Script operacional para validar inicialização, escrita, leitura e persistência no ChromaDB.

Uso típico no PC Central ou CI (dentro do contêiner `rbac-bot`):
    python scripts/verify_chroma_persistence.py --phase write
    docker compose restart chromadb
    python scripts/verify_chroma_persistence.py --phase verify
    python scripts/verify_chroma_persistence.py --phase cleanup
"""

from __future__ import annotations

import argparse
import sys

from rbac_assistant.config.settings import load_settings
from rbac_assistant.domain.ports.vector_store_port import VectorRecord
from rbac_assistant.infrastructure.vector_db.chroma_vector_store import ChromaVectorStore

SMOKE_COLLECTION = "etapa1_smoke_persistence"
SMOKE_RECORD_ID = "rbac_43_smoke_001"
SMOKE_DOCUMENT = "Registro de smoke test de persistência não-root no ChromaDB 1.5.9."
SMOKE_EMBEDDING = (0.1, 0.2, 0.3, 0.4)
SMOKE_METADATA = {
    "regulation_number": "RBAC 43",
    "section_number": "43.1",
    "extraction_method": "native_pdf",
    "validation_status": "validated",
}


def _build_store() -> ChromaVectorStore:
    settings = load_settings()
    return ChromaVectorStore(
        host=settings.chroma_host,
        port=settings.chroma_port,
        heartbeat_path=settings.vector_db.heartbeat_path,
        timeout_seconds=settings.vector_db.request_timeout_seconds,
    )


def _assert_record_matches(records: list[VectorRecord]) -> None:
    if len(records) != 1:
        raise AssertionError(f"Esperado 1 registro recuperado, encontrado(s): {len(records)}")
    record = records[0]
    if record.record_id != SMOKE_RECORD_ID:
        raise AssertionError(f"ID divergente: {record.record_id!r}")
    if record.document != SMOKE_DOCUMENT:
        raise AssertionError(f"Documento divergente: {record.document!r}")
    for key, expected_value in SMOKE_METADATA.items():
        actual = record.metadata.get(key)
        if actual != expected_value:
            raise AssertionError(f"Metadado '{key}' divergente: {actual!r} != {expected_value!r}")


def run_phase(phase: str) -> int:
    """Executa uma fase do teste de fumaça de persistência contra o ChromaDB."""
    store = _build_store()
    heartbeat = store.check_heartbeat()
    print(f"[OK] Heartbeat /api/v2/heartbeat respondido: {heartbeat}")

    if phase == "write":
        record = VectorRecord(
            record_id=SMOKE_RECORD_ID,
            document=SMOKE_DOCUMENT,
            embedding=SMOKE_EMBEDDING,
            metadata=SMOKE_METADATA,
        )
        store.upsert_records(collection_name=SMOKE_COLLECTION, records=[record])
        loaded = store.get_records(
            collection_name=SMOKE_COLLECTION,
            record_ids=[SMOKE_RECORD_ID],
        )
        _assert_record_matches(loaded)
        print("[OK] Escrita e leitura imediata validadas no ChromaDB.")
        return 0

    if phase == "verify":
        loaded = store.get_records(
            collection_name=SMOKE_COLLECTION,
            record_ids=[SMOKE_RECORD_ID],
        )
        _assert_record_matches(loaded)
        hits = store.query_similar(
            collection_name=SMOKE_COLLECTION,
            query_embedding=SMOKE_EMBEDDING,
            top_k=1,
            where={"regulation_number": "RBAC 43"},
        )
        if not hits or hits[0].record_id != SMOKE_RECORD_ID:
            raise AssertionError(f"Busca vetorial pós-reinício não retornou o registro: {hits!r}")
        print("[OK] Persistência em /data e consulta vetorial pós-reinício validadas.")
        return 0

    if phase == "cleanup":
        store.delete_collection(SMOKE_COLLECTION)
        print("[OK] Coleção temporária de smoke test removida.")
        return 0

    raise ValueError(f"Fase inválida: {phase}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validação de inicialização, escrita, leitura e persistência no ChromaDB."
    )
    parser.add_argument(
        "--phase",
        choices=("write", "verify", "cleanup"),
        required=True,
    )
    args = parser.parse_args()
    sys.exit(run_phase(args.phase))


if __name__ == "__main__":
    main()
