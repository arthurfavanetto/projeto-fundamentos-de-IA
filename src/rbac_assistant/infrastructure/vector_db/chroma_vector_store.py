"""Adaptador HTTP para ChromaDB 1.5.9 (rede interna Docker, API v2)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

import chromadb
import httpx

from rbac_assistant.domain.exceptions import (
    EmbeddingCompatibilityError,
    StorageConnectionError,
)
from rbac_assistant.domain.ports.vector_store_port import (
    CollectionMetadata,
    MetadataFilterValue,
    MetadataValue,
    VectorRecord,
    VectorSearchHit,
)


def _format_chroma_where(
    where: Mapping[str, MetadataFilterValue] | None,
) -> dict[str, Any] | None:
    """Converte mapeamento em filtro compatível com ChromaDB 1.5.9 (`$and` e `$in`)."""
    if not where:
        return None
    items: list[dict[str, Any]] = []
    for key, value in where.items():
        if isinstance(value, (list, tuple)):
            if len(value) == 1:
                items.append({key: value[0]})
            else:
                items.append({key: {"$in": list(value)}})
        else:
            items.append({key: value})
    if len(items) == 1:
        return items[0]
    return {"$and": items}


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

    def _try_get_existing_collection(self, collection_name: str) -> Any | None:
        client = self._get_client()
        if hasattr(client, "get_collection"):
            try:
                return client.get_collection(name=collection_name)
            except Exception as exc:
                msg = str(exc).lower()
                if "does not exist" in msg or "not found" in msg or isinstance(exc, KeyError):
                    return None
                # Se for erro de conexão/rede, propaga como StorageConnectionError
                if isinstance(exc, httpx.HTTPError | ConnectionError | TimeoutError):
                    raise StorageConnectionError(
                        f"Falha ao consultar coleção '{collection_name}' no ChromaDB: {exc}"
                    ) from exc
                # Para erros de coleção inexistente de clientes fakes/ChromaDB (ValueError/NotFound)
                if isinstance(exc, ValueError):
                    return None
                raise StorageConnectionError(
                    f"Falha ao consultar coleção '{collection_name}' no ChromaDB: {exc}"
                ) from exc
        return None

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

    def ensure_collection_compatible(
        self,
        *,
        collection_name: str,
        embedding_model: str,
        embedding_dimension: int,
        instruction_strategy_version: str | None = None,
    ) -> CollectionMetadata:
        """Cria ou valida coleção garantindo compatibilidade de modelo, dimensão e estratégia."""
        existing = self._try_get_existing_collection(collection_name)
        if existing is not None:
            meta = dict(getattr(existing, "metadata", None) or {})
            stored_model = meta.get("embedding_model")
            stored_dim_raw = meta.get("embedding_dimension")
            stored_dim = int(stored_dim_raw) if stored_dim_raw is not None else None
            stored_strategy = meta.get("instruction_strategy_version")
            stored_created_at = meta.get("created_at")
            count = int(existing.count()) if hasattr(existing, "count") else 0

            if count > 0:
                if stored_model is not None and str(stored_model) != embedding_model:
                    raise EmbeddingCompatibilityError(
                        f"Incompatibilidade de modelo de embeddings na coleção "
                        f"'{collection_name}': coleção indexada com '{stored_model}', "
                        f"mas configuração atual exige '{embedding_model}'. "
                        "Espaços vetoriais de modelos distintos são incompatíveis; utilize "
                        "uma coleção versionada ou execute rebuild controlado a partir do SQLite."
                    )
                if stored_dim is not None and stored_dim != embedding_dimension:
                    raise EmbeddingCompatibilityError(
                        f"Incompatibilidade de dimensionalidade na coleção '{collection_name}': "
                        f"coleção possui vetores de {stored_dim} dimensões, mas configuração "
                        f"atual exige {embedding_dimension}. Execute rebuild controlado."
                    )
                if (
                    instruction_strategy_version
                    and stored_strategy is not None
                    and str(stored_strategy) != instruction_strategy_version
                ):
                    raise EmbeddingCompatibilityError(
                        f"Incompatibilidade de estratégia de instrução na coleção "
                        f"'{collection_name}': coleção usa '{stored_strategy}', mas "
                        f"configuração atual exige '{instruction_strategy_version}'. "
                        "Execute rebuild controlado."
                    )
            return CollectionMetadata(
                name=collection_name,
                embedding_model=str(stored_model) if stored_model is not None else embedding_model,
                embedding_dimension=stored_dim if stored_dim is not None else embedding_dimension,
                instruction_strategy_version=(
                    str(stored_strategy)
                    if stored_strategy is not None
                    else instruction_strategy_version
                ),
                created_at=str(stored_created_at) if stored_created_at is not None else None,
                count=count,
            )

        created_at = datetime.now(UTC).isoformat()
        col_metadata: dict[str, MetadataValue] = {
            "hnsw:space": "cosine",
            "embedding_model": embedding_model,
            "embedding_dimension": int(embedding_dimension),
            "created_at": created_at,
        }
        if instruction_strategy_version:
            col_metadata["instruction_strategy_version"] = instruction_strategy_version

        try:
            collection = self._get_client().get_or_create_collection(
                name=collection_name,
                metadata=col_metadata,
            )
            count = int(collection.count()) if hasattr(collection, "count") else 0
            return CollectionMetadata(
                name=collection_name,
                embedding_model=embedding_model,
                embedding_dimension=embedding_dimension,
                instruction_strategy_version=instruction_strategy_version,
                created_at=created_at,
                count=count,
            )
        except (StorageConnectionError, EmbeddingCompatibilityError):
            raise
        except Exception as exc:
            raise StorageConnectionError(
                f"Falha ao criar/validar coleção '{collection_name}': {exc}"
            ) from exc

    def get_collection_metadata(self, collection_name: str) -> CollectionMetadata | None:
        """Retorna metadados de modelo, dimensão, estratégia e contagem de uma coleção."""
        existing = self._try_get_existing_collection(collection_name)
        if existing is None:
            return None
        meta = dict(getattr(existing, "metadata", None) or {})
        stored_model = meta.get("embedding_model")
        stored_dim_raw = meta.get("embedding_dimension")
        stored_strategy = meta.get("instruction_strategy_version")
        stored_created_at = meta.get("created_at")
        count = int(existing.count()) if hasattr(existing, "count") else 0
        return CollectionMetadata(
            name=collection_name,
            embedding_model=str(stored_model) if stored_model is not None else None,
            embedding_dimension=int(stored_dim_raw) if stored_dim_raw is not None else None,
            instruction_strategy_version=(
                str(stored_strategy) if stored_strategy is not None else None
            ),
            created_at=str(stored_created_at) if stored_created_at is not None else None,
            count=count,
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

    def delete_records(
        self,
        *,
        collection_name: str,
        record_ids: Sequence[str],
    ) -> None:
        """Remove registros obsoletos específicos pelos seus identificadores."""
        if not record_ids:
            return
        existing = self._try_get_existing_collection(collection_name)
        if existing is None:
            return
        try:
            existing.delete(ids=list(record_ids))
        except StorageConnectionError:
            raise
        except Exception as exc:
            raise StorageConnectionError(
                f"Falha ao remover registros da coleção '{collection_name}': {exc}"
            ) from exc

    def delete_by_where(
        self,
        *,
        collection_name: str,
        where: Mapping[str, MetadataFilterValue],
    ) -> None:
        """Remove todos os registros que satisfaçam um filtro de metadados."""
        if not where:
            return
        existing = self._try_get_existing_collection(collection_name)
        if existing is None:
            return
        try:
            existing.delete(where=_format_chroma_where(where))
        except StorageConnectionError:
            raise
        except Exception as exc:
            raise StorageConnectionError(
                f"Falha ao remover registros filtrados da coleção '{collection_name}': {exc}"
            ) from exc

    def count_records(self, collection_name: str) -> int:
        """Retorna a quantidade de vetores persistidos na coleção (0 se inexistente)."""
        existing = self._try_get_existing_collection(collection_name)
        if existing is None:
            return 0
        try:
            return int(existing.count())
        except Exception as exc:
            raise StorageConnectionError(
                f"Falha ao contar registros na coleção '{collection_name}': {exc}"
            ) from exc

    def list_record_ids(
        self,
        *,
        collection_name: str,
        where: Mapping[str, MetadataFilterValue] | None = None,
    ) -> list[str]:
        """Lista os IDs de registros persistidos na coleção (opcionalmente filtrados)."""
        existing = self._try_get_existing_collection(collection_name)
        if existing is None:
            return []
        try:
            kwargs: dict[str, Any] = {"include": []}
            formatted_where = _format_chroma_where(where)
            if formatted_where is not None:
                kwargs["where"] = formatted_where
            raw = existing.get(**kwargs)
            return [str(rec_id) for rec_id in (raw.get("ids") or [])]
        except StorageConnectionError:
            raise
        except Exception as exc:
            raise StorageConnectionError(
                f"Falha ao listar IDs da coleção '{collection_name}': {exc}"
            ) from exc

    def query_similar(
        self,
        *,
        collection_name: str,
        query_embedding: Sequence[float],
        top_k: int = 5,
        where: Mapping[str, MetadataFilterValue] | None = None,
    ) -> list[VectorSearchHit]:
        """Executa busca semântica vetorial filtrada por metadados."""
        try:
            collection = self._get_client().get_or_create_collection(name=collection_name)
            if hasattr(collection, "count"):
                total = int(collection.count())
                if total == 0:
                    return []
                effective_top_k = min(top_k, total)
            else:
                effective_top_k = top_k

            query_kwargs: dict[str, Any] = {
                "query_embeddings": [list(query_embedding)],
                "n_results": effective_top_k,
                "include": ["documents", "metadatas", "distances"],
            }
            formatted_where = _format_chroma_where(where)
            if formatted_where is not None:
                query_kwargs["where"] = formatted_where
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
            msg = str(exc).lower()
            if "does not exist" in msg or "not found" in msg:
                return
            raise StorageConnectionError(
                f"Falha ao remover coleção '{collection_name}': {exc}"
            ) from exc
