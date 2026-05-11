from __future__ import annotations

from time import perf_counter

from models.patient import PatientContext
from models.retrieval import RetrievalResult, RetrievedChunk
from services.embedder import EmbeddingService
from services.logging_config import get_logger
from services.pinecone_service import PineconeService
from services.reranker import RerankerService


class RetrievalService:
    """Orchestrates online vector retrieval and reranking."""

    def __init__(
        self,
        embedder: EmbeddingService,
        pinecone: PineconeService,
        reranker: RerankerService,
        retrieval_top_k: int = 20,
    ) -> None:
        """Create a retrieval service.

        Args:
            embedder: Query embedding service.
            pinecone: Vector database service.
            reranker: Cross-encoder reranker service.
            retrieval_top_k: Number of vector candidates before reranking.

        Returns:
            None.

        Raises:
            None.
        """
        self._embedder = embedder
        self._pinecone = pinecone
        self._reranker = reranker
        self._retrieval_top_k = retrieval_top_k
        self._logger = get_logger(__name__)

    async def retrieve(
        self,
        question: str,
        patient_context: PatientContext | None,
        top_k: int,
    ) -> RetrievalResult:
        """Retrieve and rerank evidence chunks for a clinical question.

        Args:
            question: User clinical question.
            patient_context: Optional patient context for query expansion.
            top_k: Final maximum chunk count.

        Returns:
            Retrieval result with timing metadata.

        Raises:
            Exception: If embedding or vector search fails.
        """
        total_start = perf_counter()
        expanded_query = self.expand_query(question, patient_context)

        query_embedding = await self._embedder.embed_query(expanded_query)
        retrieval_start = perf_counter()
        matches = await self._pinecone.query(query_embedding, top_k=self._retrieval_top_k, filter=None)
        retrieval_duration_ms = int((perf_counter() - retrieval_start) * 1000)

        candidates = [self._match_to_chunk(match) for match in matches]
        rerank_start = perf_counter()
        try:
            reranked = await self._reranker.rerank(expanded_query, candidates)
        except Exception as exc:
            self._logger.error("retrieval_reranker_unexpected_failure", error=str(exc))
            reranked = sorted(candidates, key=lambda chunk: chunk.vector_score, reverse=True)
        reranking_duration_ms = int((perf_counter() - rerank_start) * 1000)
        final_chunks = reranked[: max(top_k, 0)]

        self._logger.info(
            "retrieval_completed",
            patient_id=patient_context.patient_id if patient_context else None,
            question=question[:100],
            candidates_retrieved=len(candidates),
            final_chunks=len(final_chunks),
            total_duration_ms=int((perf_counter() - total_start) * 1000),
        )
        return RetrievalResult(
            query=question,
            expanded_query=expanded_query,
            chunks=final_chunks,
            retrieval_duration_ms=retrieval_duration_ms,
            reranking_duration_ms=reranking_duration_ms,
            total_candidates=len(candidates),
            patient_id=patient_context.patient_id if patient_context else None,
        )

    @staticmethod
    def expand_query(question: str, patient_context: PatientContext | None) -> str:
        """Append patient conditions to improve retrieval relevance.

        Args:
            question: Raw clinical question.
            patient_context: Optional patient context.

        Returns:
            Expanded query.

        Raises:
            None.
        """
        if patient_context is None or not patient_context.active_conditions:
            return question
        condition_names = patient_context.get_condition_names()[:3]
        return f"{question} Patient conditions: {', '.join(condition_names)}"

    @staticmethod
    def _match_to_chunk(match: object) -> RetrievedChunk:
        metadata = RetrievalService._field(match, "metadata", {}) or {}
        return RetrievedChunk(
            chunk_id=str(RetrievalService._field(match, "id", "")),
            pmid=str(metadata.get("pmid", "")),
            title=str(metadata.get("title", "")),
            text=str(metadata.get("text", "")),
            journal=str(metadata.get("journal", "")),
            pub_year=int(metadata.get("pub_year", 0) or 0),
            doi=str(metadata.get("doi") or "") or None,
            mesh_terms=list(metadata.get("mesh_terms", []) or []),
            vector_score=float(RetrievalService._field(match, "score", 0.0) or 0.0),
            rerank_score=None,
        )

    @staticmethod
    def _field(obj: object, key: str, default: object) -> object:
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)
