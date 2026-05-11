from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from unittest.mock import AsyncMock

import pytest

from models.retrieval import EmbeddedChunk, TextChunk, RetrievedChunk
from services.embedder import EmbeddingService
from services.pinecone_service import PineconeService
from services.reranker import RerankerService

pytestmark = pytest.mark.asyncio


@dataclass
class FakeEmbeddingData:
    index: int
    embedding: list[float]


class FakeEmbeddingResponse:
    def __init__(self) -> None:
        self.data = [
            FakeEmbeddingData(index=1, embedding=[0.2, 0.2]),
            FakeEmbeddingData(index=0, embedding=[0.1, 0.1]),
        ]


def chunk(index: int = 0) -> TextChunk:
    return TextChunk(
        chunk_id=f"pmid_chunk_{index}",
        pmid="pmid",
        title="Title",
        text="Title\n\nClinical evidence text",
        token_count=4,
        chunk_index=index,
        total_chunks=2,
        journal="JAMA",
        pub_year=2024,
        mesh_terms=["Diabetes"],
        doi=None,
    )


async def test_embedding_service_embeds_chunks_in_input_order():
    service = EmbeddingService.__new__(EmbeddingService)
    service.model = "text-embedding-3-large"
    service.dimensions = 3072
    service._create_embeddings = AsyncMock(return_value=FakeEmbeddingResponse())
    service._logger = type("Logger", (), {"info": lambda *args, **kwargs: None})()

    embedded = await service.embed_chunks([chunk(0), chunk(1)])
    assert [item.embedding for item in embedded] == [[0.1, 0.1], [0.2, 0.2]]


async def test_embedding_service_embed_query_returns_vector():
    service = EmbeddingService.__new__(EmbeddingService)
    service._create_embeddings = AsyncMock(return_value=FakeEmbeddingResponse())
    service._logger = type("Logger", (), {"debug": lambda *args, **kwargs: None})()
    assert await service.embed_query("query") == [0.2, 0.2]


async def test_embedding_service_requires_api_key():
    with pytest.raises(ValueError):
        EmbeddingService("", "text-embedding-3-large", 3072)


async def test_embedding_service_init_uses_openai_client(monkeypatch: pytest.MonkeyPatch):
    fake_openai = types.SimpleNamespace(
        AsyncOpenAI=lambda api_key, timeout: types.SimpleNamespace(api_key=api_key, timeout=timeout)
    )
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    service = EmbeddingService("sk-test", "model", 16)
    assert service.model == "model"
    assert service.dimensions == 16


class FakeIndex:
    def __init__(self) -> None:
        self.upserted = []

    def describe_index_stats(self):
        return {"total_vector_count": len(self.upserted), "dimension": 2, "index_fullness": 0.0, "namespaces": {}}

    def upsert(self, vectors):
        self.upserted.extend(vectors)

    def query(self, **kwargs):
        return {"matches": [{"id": "pmid_chunk_0", "score": 0.9, "metadata": {"pmid": "pmid", "title": "Title", "text": "Text", "journal": "JAMA", "pub_year": 2024}}]}

    def delete(self, filter):
        return {"deleted_count": 1}


async def test_pinecone_service_methods_with_fake_index():
    service = PineconeService.__new__(PineconeService)
    service.index_name = "clinical-copilot"
    service._index = FakeIndex()
    service._logger = type("Logger", (), {"info": lambda *args, **kwargs: None})()

    embedded = [EmbeddedChunk(chunk=chunk(0), embedding=[0.1, 0.2])]
    assert await service.upsert_chunks(embedded) == 1
    assert (await service.query([0.1, 0.2], 1))[0]["id"] == "pmid_chunk_0"
    assert (await service.get_index_stats())["dimension"] == 2
    assert await service.delete_by_pmid("pmid") == 1


async def test_pinecone_service_requires_api_key():
    with pytest.raises(ValueError):
        PineconeService("", "clinical-copilot")


async def test_pinecone_service_init_uses_client(monkeypatch: pytest.MonkeyPatch):
    fake_index = FakeIndex()
    fake_pinecone_module = types.SimpleNamespace(
        Pinecone=lambda api_key: types.SimpleNamespace(Index=lambda index_name: fake_index)
    )
    monkeypatch.setitem(sys.modules, "pinecone", fake_pinecone_module)
    service = PineconeService("pcsk-test", "clinical-copilot")
    assert service.index_name == "clinical-copilot"


class FakeAsyncRerankClient:
    async def rerank(self, **kwargs):
        return types.SimpleNamespace(
            results=[
                types.SimpleNamespace(index=1, relevance_score=0.99),
                types.SimpleNamespace(index=0, relevance_score=0.5),
            ]
        )


class FakeFailingRerankClient:
    async def rerank(self, **kwargs):
        raise RuntimeError("cohere down")


def retrieved(score: float, suffix: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"{suffix}_chunk_0",
        pmid=suffix,
        title="Title",
        text="Clinical evidence",
        journal="JAMA",
        pub_year=2024,
        vector_score=score,
        mesh_terms=[],
    )


async def test_reranker_service_success_maps_scores():
    service = RerankerService.__new__(RerankerService)
    service.model = "rerank-v3.5"
    service.top_n = 2
    service._client = FakeAsyncRerankClient()
    service._is_async = True
    service._logger = type("Logger", (), {"info": lambda *args, **kwargs: None, "error": lambda *args, **kwargs: None})()

    results = await service.rerank("query", [retrieved(0.1, "a"), retrieved(0.2, "b")])
    assert results[0].pmid == "b"
    assert results[0].rerank_score == 0.99


async def test_reranker_service_falls_back_on_error():
    service = RerankerService.__new__(RerankerService)
    service.model = "rerank-v3.5"
    service.top_n = 2
    service._client = FakeFailingRerankClient()
    service._is_async = True
    service._logger = type("Logger", (), {"info": lambda *args, **kwargs: None, "error": lambda *args, **kwargs: None})()

    results = await service.rerank("query", [retrieved(0.1, "a"), retrieved(0.9, "b")])
    assert results[0].pmid == "b"


async def test_reranker_service_requires_api_key():
    with pytest.raises(ValueError):
        RerankerService("", "rerank-v3.5", 5)


async def test_reranker_service_init_uses_async_client(monkeypatch: pytest.MonkeyPatch):
    fake_cohere = types.SimpleNamespace(
        AsyncClient=lambda api_key, timeout: types.SimpleNamespace(api_key=api_key, timeout=timeout)
    )
    monkeypatch.setitem(sys.modules, "cohere", fake_cohere)
    service = RerankerService("co-test", "rerank-v3.5", 5)
    assert service._is_async is True
    assert service.model == "rerank-v3.5"


async def test_reranker_service_init_falls_back_to_sync_client(monkeypatch: pytest.MonkeyPatch):
    def broken_async_client(api_key, timeout):
        raise TypeError("no async client")

    fake_cohere = types.SimpleNamespace(
        AsyncClient=broken_async_client,
        Client=lambda api_key, timeout: types.SimpleNamespace(api_key=api_key, timeout=timeout),
    )
    monkeypatch.setitem(sys.modules, "cohere", fake_cohere)
    service = RerankerService("co-test", "rerank-v3.5", 5)
    assert service._is_async is False


async def test_reranker_service_empty_chunks_returns_empty():
    service = RerankerService.__new__(RerankerService)
    assert await service.rerank("query", []) == []


class FakeSyncRerankClient:
    def rerank(self, **kwargs):
        return {"results": [{"index": 0, "relevance_score": 0.77}]}


async def test_reranker_service_sync_client_branch():
    service = RerankerService.__new__(RerankerService)
    service.model = "rerank-v3.5"
    service.top_n = 1
    service._client = FakeSyncRerankClient()
    service._is_async = False
    service._logger = type("Logger", (), {"info": lambda *args, **kwargs: None, "error": lambda *args, **kwargs: None})()

    results = await service.rerank("query", [retrieved(0.1, "a")])
    assert results[0].rerank_score == 0.77
