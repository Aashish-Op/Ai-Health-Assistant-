from __future__ import annotations

import asyncio
from time import perf_counter

from models.retrieval import RetrievedChunk
from services.logging_config import get_logger


class RerankerService:
    """Cohere reranker wrapper with vector-score fallback."""

    def __init__(self, api_key: str, model: str, top_n: int) -> None:
        """Create a reranker service.

        Args:
            api_key: Cohere API key.
            model: Cohere reranker model.
            top_n: Final number of chunks to return.

        Returns:
            None.

        Raises:
            ValueError: If api_key is empty.
            ImportError: If the Cohere SDK is unavailable.
        """
        if not api_key:
            raise ValueError("COHERE_API_KEY is required for reranking")
        import cohere

        self.model = model
        self.top_n = top_n
        self._logger = get_logger(__name__)
        try:
            self._client = cohere.AsyncClient(api_key=api_key, timeout=30.0)
            self._is_async = True
        except TypeError:
            self._client = cohere.Client(api_key=api_key, timeout=30.0)
            self._is_async = False

    async def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Rerank candidate chunks with Cohere.

        Args:
            query: Expanded clinical retrieval query.
            chunks: Vector-search candidates.

        Returns:
            Reranked top-N chunks, or vector-score fallback on error.

        Raises:
            None.
        """
        if not chunks:
            return []
        start = perf_counter()
        try:
            documents = [chunk.text for chunk in chunks]
            if self._is_async:
                response = await self._client.rerank(
                    model=self.model,
                    query=query,
                    documents=documents,
                    top_n=min(self.top_n, len(chunks)),
                )
            else:
                response = await asyncio.to_thread(
                    self._client.rerank,
                    model=self.model,
                    query=query,
                    documents=documents,
                    top_n=min(self.top_n, len(chunks)),
                )
            reranked = self._map_results(chunks, response)
            self._logger.info(
                "reranking_completed",
                candidate_count=len(chunks),
                top_n=len(reranked),
                top_score=reranked[0].rerank_score if reranked else None,
                bottom_score=reranked[-1].rerank_score if reranked else None,
                duration_ms=int((perf_counter() - start) * 1000),
            )
            return reranked
        except Exception as exc:
            self._logger.error("reranking_failed_vector_fallback", error=str(exc))
            return sorted(chunks, key=lambda chunk: chunk.vector_score, reverse=True)[: self.top_n]

    def _map_results(self, chunks: list[RetrievedChunk], response: object) -> list[RetrievedChunk]:
        results = getattr(response, "results", response.get("results", []) if isinstance(response, dict) else [])
        mapped: list[RetrievedChunk] = []
        for result in results:
            index = self._field(result, "index", 0)
            score = float(self._field(result, "relevance_score", 0.0))
            mapped.append(chunks[index].model_copy(update={"rerank_score": score}))
        return sorted(mapped, key=lambda chunk: chunk.rerank_score or 0.0, reverse=True)[: self.top_n]

    @staticmethod
    def _field(obj: object, key: str, default: object) -> object:
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)
