from __future__ import annotations

import asyncio
from typing import Any

from models.retrieval import EmbeddedChunk
from services.logging_config import get_logger


class PineconeService:
    """Pinecone vector index wrapper."""

    def __init__(self, api_key: str, index_name: str) -> None:
        """Create a Pinecone service and connect to an index.

        Args:
            api_key: Pinecone API key.
            index_name: Pinecone index name.

        Returns:
            None.

        Raises:
            ValueError: If api_key is empty.
            ImportError: If the Pinecone SDK is unavailable.
        """
        if not api_key:
            raise ValueError("PINECONE_API_KEY is required for vector retrieval")
        from pinecone import Pinecone

        self.index_name = index_name
        self._logger = get_logger(__name__)
        self._client = Pinecone(api_key=api_key)
        self._index = self._client.Index(index_name)
        try:
            stats = self._index.describe_index_stats()
            self._logger.info(
                "pinecone_connected",
                index_name=index_name,
                vector_count=self._stats_value(stats, "total_vector_count", 0),
                dimension=self._stats_value(stats, "dimension", None),
            )
        except Exception as exc:
            self._logger.error("pinecone_connection_check_failed", index_name=index_name, error=str(exc))
            raise

    async def upsert_chunks(self, chunks: list[EmbeddedChunk]) -> int:
        """Upsert embedded chunks to Pinecone.

        Args:
            chunks: Embedded chunks to store.

        Returns:
            Count of vectors submitted for upsert.

        Raises:
            Exception: If Pinecone upsert fails.
        """
        upserted = 0
        for start in range(0, len(chunks), 100):
            batch = chunks[start : start + 100]
            vectors = [self._vector_payload(chunk) for chunk in batch]
            await asyncio.to_thread(self._index.upsert, vectors=vectors)
            upserted += len(vectors)
            stats = await self.get_index_stats()
            self._logger.info(
                "pinecone_upsert_batch_completed",
                batch_number=(start // 100) + 1,
                cumulative_upserted=upserted,
                index_size=stats.get("total_vector_count", 0),
            )
        return upserted

    async def query(
        self,
        embedding: list[float],
        top_k: int,
        filter: dict[str, object] | None = None,
    ) -> list[object]:
        """Query Pinecone by embedding vector.

        Args:
            embedding: Query embedding.
            top_k: Number of candidates to retrieve.
            filter: Optional Pinecone metadata filter.

        Returns:
            Raw Pinecone match objects.

        Raises:
            Exception: If Pinecone query fails.
        """
        result = await asyncio.to_thread(
            self._index.query,
            vector=embedding,
            top_k=top_k,
            include_metadata=True,
            filter=filter,
        )
        return list(self._stats_value(result, "matches", []))

    async def get_index_stats(self) -> dict[str, object]:
        """Return Pinecone index stats.

        Args:
            None.

        Returns:
            Normalized stats dictionary.

        Raises:
            Exception: If Pinecone stats retrieval fails.
        """
        stats = await asyncio.to_thread(self._index.describe_index_stats)
        if isinstance(stats, dict):
            return stats
        if hasattr(stats, "to_dict"):
            return dict(stats.to_dict())
        return {
            "total_vector_count": self._stats_value(stats, "total_vector_count", 0),
            "dimension": self._stats_value(stats, "dimension", 0),
            "index_fullness": self._stats_value(stats, "index_fullness", 0.0),
            "namespaces": self._stats_value(stats, "namespaces", {}),
        }

    async def delete_by_pmid(self, pmid: str) -> int:
        """Delete all vectors for one PMID.

        Args:
            pmid: PMID metadata value.

        Returns:
            Number of deleted vectors when reported, otherwise 0.

        Raises:
            Exception: If Pinecone delete fails.
        """
        result = await asyncio.to_thread(self._index.delete, filter={"pmid": {"$eq": pmid}})
        return int(self._stats_value(result, "deleted_count", 0) or 0)

    @staticmethod
    def _vector_payload(item: EmbeddedChunk) -> dict[str, object]:
        chunk = item.chunk
        return {
            "id": chunk.chunk_id,
            "values": item.embedding,
            "metadata": {
                "pmid": chunk.pmid,
                "title": chunk.title[:200],
                "text": chunk.text,
                "journal": chunk.journal,
                "pub_year": chunk.pub_year,
                "doi": chunk.doi or "",
                "mesh_terms": chunk.mesh_terms[:10],
                "chunk_index": chunk.chunk_index,
                "total_chunks": chunk.total_chunks,
            },
        }

    @staticmethod
    def _stats_value(obj: object, key: str, default: Any) -> Any:
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)
