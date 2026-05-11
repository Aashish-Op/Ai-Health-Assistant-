from __future__ import annotations

from time import perf_counter

from tenacity import RetryCallState, retry, stop_after_attempt, wait_exponential

from models.retrieval import EmbeddedChunk, TextChunk
from services.logging_config import get_logger

_logger = get_logger(__name__)


class EmbeddingService:
    """OpenAI embedding client for chunks and query strings."""

    def __init__(self, api_key: str, model: str, dimensions: int) -> None:
        """Create an embedding service.

        Args:
            api_key: OpenAI API key.
            model: Embedding model name.
            dimensions: Requested embedding dimensionality.

        Returns:
            None.

        Raises:
            ValueError: If api_key is empty.
            ImportError: If the OpenAI SDK is unavailable.
        """
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for embeddings")
        from openai import AsyncOpenAI

        self.model = model
        self.dimensions = dimensions
        self._client = AsyncOpenAI(api_key=api_key, timeout=60.0)
        self._logger = get_logger(__name__)

    async def embed_chunks(self, chunks: list[TextChunk]) -> list[EmbeddedChunk]:
        """Embed text chunks in batches of 100.

        Args:
            chunks: Text chunks to embed.

        Returns:
            Embedded chunks in the same order as input.

        Raises:
            openai.APIError: If embedding requests fail after retries.
        """
        embedded: list[EmbeddedChunk] = []
        cumulative_tokens = 0
        for start in range(0, len(chunks), 100):
            batch = chunks[start : start + 100]
            texts = [chunk.text for chunk in batch]
            estimated_tokens = sum(int(len(text.split()) * 1.3) for text in texts)
            cumulative_tokens += estimated_tokens
            response = await self._create_embeddings(texts)
            vectors = sorted(response.data, key=lambda item: item.index)
            embedded.extend(
                EmbeddedChunk(chunk=chunk, embedding=list(vector.embedding))
                for chunk, vector in zip(batch, vectors, strict=True)
            )
            self._logger.info(
                "embedding_batch_completed",
                batch_size=len(batch),
                estimated_tokens=estimated_tokens,
                cumulative_total=cumulative_tokens,
            )
        return embedded

    async def embed_query(self, text: str) -> list[float]:
        """Embed one retrieval query.

        Args:
            text: Expanded clinical retrieval query.

        Returns:
            Embedding vector.

        Raises:
            openai.APIError: If embedding requests fail after retries.
        """
        start = perf_counter()
        response = await self._create_embeddings([text])
        self._logger.debug(
            "query_embedded",
            duration_ms=int((perf_counter() - start) * 1000),
        )
        return list(response.data[0].embedding)

    @retry(
        wait=wait_exponential(min=1, max=60),
        stop=stop_after_attempt(5),
        before_sleep=lambda retry_state: _log_embedding_retry(retry_state),
        reraise=True,
    )
    async def _create_embeddings(self, texts: list[str]) -> object:
        """Call the OpenAI embeddings endpoint.

        Args:
            texts: Input strings.

        Returns:
            OpenAI embeddings response.

        Raises:
            openai.APIError: If the OpenAI SDK call fails after retries.
        """
        return await self._client.embeddings.create(
            model=self.model,
            input=texts,
            dimensions=self.dimensions,
            encoding_format="float",
            timeout=60.0,
        )


def _log_embedding_retry(retry_state: RetryCallState) -> None:
    exception = retry_state.outcome.exception() if retry_state.outcome else None
    _logger.warning(
        "embedding_retry_scheduled",
        attempt_number=retry_state.attempt_number,
        error_type=type(exception).__name__ if exception else None,
    )
