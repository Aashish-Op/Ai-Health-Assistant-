from __future__ import annotations

import pytest

from models.retrieval import RetrievedChunk
from services.retrieval_service import RetrievalService

pytestmark = pytest.mark.asyncio


class FakeEmbedder:
    async def embed_query(self, text: str) -> list[float]:
        self.last_text = text
        return [0.1, 0.2, 0.3]


class FakePinecone:
    def __init__(self, matches: list[dict]) -> None:
        self.matches = matches

    async def query(self, embedding: list[float], top_k: int, filter: dict | None = None) -> list[dict]:
        self.last_top_k = top_k
        return self.matches


class FakeReranker:
    async def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        return sorted(
            [
                chunk.model_copy(update={"rerank_score": 1.0 - index / 10})
                for index, chunk in enumerate(chunks)
            ],
            key=lambda chunk: chunk.rerank_score or 0,
            reverse=True,
        )


class FailingReranker:
    async def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        raise RuntimeError("cohere down")


def matches() -> list[dict]:
    return [
        {
            "id": "111_chunk_0",
            "score": 0.7,
            "metadata": {
                "pmid": "111",
                "title": "Diabetes therapy",
                "text": "Metformin evidence",
                "journal": "JAMA",
                "pub_year": 2022,
                "doi": "",
                "mesh_terms": ["Diabetes"],
            },
        },
        {
            "id": "222_chunk_0",
            "score": 0.9,
            "metadata": {
                "pmid": "222",
                "title": "Hypertension therapy",
                "text": "Blood pressure evidence",
                "journal": "NEJM",
                "pub_year": 2023,
                "doi": "10.1/test",
                "mesh_terms": ["Hypertension"],
            },
        },
    ]


async def test_query_expansion_appends_patient_conditions(patient_context):
    service = RetrievalService(FakeEmbedder(), FakePinecone([]), FakeReranker())
    result = await service.retrieve("is this medication safe?", patient_context, 5)
    assert "Type 2 diabetes mellitus" in result.expanded_query
    assert "Essential hypertension" in result.expanded_query


async def test_query_expansion_skipped_when_no_patient():
    service = RetrievalService(FakeEmbedder(), FakePinecone([]), FakeReranker())
    result = await service.retrieve("question", None, 5)
    assert result.expanded_query == "question"


async def test_pinecone_results_converted_to_retrieved_chunks():
    service = RetrievalService(FakeEmbedder(), FakePinecone(matches()), FakeReranker())
    result = await service.retrieve("question", None, 5)
    assert result.chunks[0].pmid == "111"
    assert result.chunks[0].text == "Metformin evidence"


async def test_reranked_chunks_sorted_by_rerank_score():
    service = RetrievalService(FakeEmbedder(), FakePinecone(matches()), FakeReranker())
    result = await service.retrieve("question", None, 5)
    assert result.chunks[0].rerank_score >= result.chunks[-1].rerank_score


async def test_reranker_failure_falls_back_to_vector_score():
    service = RetrievalService(FakeEmbedder(), FakePinecone(matches()), FailingReranker())
    result = await service.retrieve("question", None, 5)
    assert result.chunks[0].pmid == "222"


async def test_retrieval_result_contains_timing_fields():
    service = RetrievalService(FakeEmbedder(), FakePinecone(matches()), FakeReranker())
    result = await service.retrieve("question", None, 5)
    assert result.retrieval_duration_ms >= 0
    assert result.reranking_duration_ms >= 0
    assert result.total_candidates == 2


async def test_empty_pinecone_results_returns_empty_chunks():
    service = RetrievalService(FakeEmbedder(), FakePinecone([]), FakeReranker())
    result = await service.retrieve("question", None, 5)
    assert result.chunks == []


async def test_patient_conditions_capped_at_three_in_expansion(patient_context):
    service = RetrievalService(FakeEmbedder(), FakePinecone([]), FakeReranker())
    result = await service.retrieve("question", patient_context, 5)
    assert result.expanded_query.count(",") == 2
