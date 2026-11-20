"""Adaptador HTTP para ChromaDB 1.5.9 (rede interna Docker, API v2)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import chromadb
import httpx

from rbac_assistant.domain.exceptions import StorageConnectionError
from rbac_assistant.domain.ports.vector_store_port import (
    MetadataValue,
    VectorRecord,
    VectorSearchHit,
)


class ChromaVectorStore:
    """Implementação de `VectorStorePort` usando servidor ChromaDB 1.5.9 via HTTP."""

    def __init__(
        self,
        *,
        host: str = "chromadb",
        port: int = 8000,
        heartbeat_path: str = "/api/v2/heartbeat",
        timeout_seconds: float = 5.0,
        client: Any | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._host = host
        self._port = port
        self._heartbeat_path = heartbeat_path
        self._timeout_seconds = timeout_seconds
        self._client: Any | None = client
        self._http_client = http_client

    @property
    def heartbeat_url(self) -> str:
        """Retorna a URL completa do endpoint de heartbeat v2 do ChromaDB."""
        return f"http://{self._host}:{self._port}{self._heartbeat_path}"

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                self._client = chromadb.HttpClient(host=self._host, port=self._port)
            except Exception as exc:
                raise StorageConnectionError(
                    f"Falha ao inicializar cliente HTTP do ChromaDB em "
                    f"{self._host}:{self._port}: {exc}"
                ) from exc
        return self._client

    def check_heartbeat(self) -> int:
        """Consulta `GET /api/v2/heartbeat` e retorna o timestamp em nanossegundos."""
        try:
            if self._http_client is not None:
                response = self._http_client.get(
                    self.heartbeat_url,
                    timeout=self._timeout_seconds,
                )
            else:
                response = httpx.get(
                    self.heartbeat_url,
                    timeout=self._timeout_seconds,
                )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            raise StorageConnectionError(
                f"Falha ao consultar heartbeat do ChromaDB em {self.heartbeat_url}: {exc}"
            ) from exc

        if isinstance(payload, dict):
            heartbeat = payload.get("nanosecond heartbeat")
            if isinstance(heartbeat, int):
                return heartbeat
        if isinstance(payload, int):
            return payload

        raise StorageConnectionError(
            f"Resposta inesperada do endpoint de heartbeat {self.heartbeat_url}: {payload!r}"
        )

    def upsert_records(
        self,
        *,
        collection_name: str,
        records: Sequence[VectorRecord],
    ) -> None:
        """Insere ou atualiza registros vetoriais de forma idempotente no ChromaDB."""
        if not records:
            return
        try:
            collection = self._get_client().get_or_create_collection(name=collection_name)
            collection.upsert(
                ids=[record.record_id for record in records],
                documents=[record.document for record in records],
                embeddings=[list(record.embedding) for record in records],
                metadatas=[dict(record.metadata) for record in records],
            )
        except StorageConnectionError:
            raise
        except Exception as exc:
            raise StorageConnectionError(
                f"Falha ao gravar registros na coleção '{collection_name}': {exc}"
            ) from exc

    def get_records(
        self,
        *,
        collection_name: str,
        record_ids: Sequence[str],
    ) -> list[VectorRecord]:
        """Recupera registros persistidos no ChromaDB pelos seus identificadores."""
        if not record_ids:
            return []
        try:
            collection = self._get_client().get_or_create_collection(name=collection_name)
            raw = collection.get(
                ids=list(record_ids),
                include=["documents", "embeddings", "metadatas"],
            )
        except StorageConnectionError:
            raise
        except Exception as exc:
            raise StorageConnectionError(
                f"Falha ao ler registros da coleção '{collection_name}': {exc}"
            ) from exc

        ids: list[str] = list(raw.get("ids") or [])
        documents: list[str] = list(raw.get("documents") or [])
        raw_embeddings = raw.get("embeddings")
        embeddings_list = list(raw_embeddings) if raw_embeddings is not None else []
        metadatas: list[dict[str, MetadataValue]] = list(raw.get("metadatas") or [])

        results: list[VectorRecord] = []
        for idx, rec_id in enumerate(ids):
            doc = documents[idx] if idx < len(documents) else ""
            emb = (
                tuple(float(v) for v in embeddings_list[idx])
                if idx < len(embeddings_list)
                else ()
            )
            meta = metadatas[idx] if idx < len(metadatas) and metadatas[idx] is not None else {}
            results.append(
                VectorRecord(
                    record_id=rec_id,
                    document=doc,
                    embedding=emb,
                    metadata=meta,
                )
            )
        return results

    def query_similar(
        self,
        *,
        collection_name: str,
        query_embedding: Sequence[float],
        top_k: int = 5,
        where: Mapping[str, MetadataValue] | None = None,
    ) -> list[VectorSearchHit]:
        """Executa busca semântica vetorial filtrada por metadados."""
        try:
            collection = self._get_client().get_or_create_collection(name=collection_name)
            query_kwargs: dict[str, Any] = {
                "query_embeddings": [list(query_embedding)],
                "n_results": top_k,
                "include": ["documents", "metadatas", "distances"],
            }
            if where:
                query_kwargs["where"] = dict(where)
            raw = collection.query(**query_kwargs)
        except StorageConnectionError:
            raise
        except Exception as exc:
            raise StorageConnectionError(
                f"Falha ao consultar coleção '{collection_name}': {exc}"
            ) from exc

        ids_batch = (raw.get("ids") or [[]])[0]
        docs_batch = (raw.get("documents") or [[]])[0]
        metas_batch = (raw.get("metadatas") or [[]])[0]
        dists_batch = (raw.get("distances") or [[]])[0]

        hits: list[VectorSearchHit] = []
        for idx, rec_id in enumerate(ids_batch):
            raw_meta = metas_batch[idx] if idx < len(metas_batch) and metas_batch[idx] else {}
            hits.append(
                VectorSearchHit(
                    record_id=str(rec_id),
                    document=str(docs_batch[idx]) if idx < len(docs_batch) else "",
                    metadata=dict(raw_meta),
                    distance=float(dists_batch[idx]) if idx < len(dists_batch) else 0.0,
                )
            )
        return hits

    def delete_collection(self, collection_name: str) -> None:
        """Remove uma coleção existente no ChromaDB."""
        try:
            self._get_client().delete_collection(name=collection_name)
        except StorageConnectionError:
            raise
        except Exception as exc:
            raise StorageConnectionError(
                f"Falha ao remover coleção '{collection_name}': {exc}"
            ) from exc
