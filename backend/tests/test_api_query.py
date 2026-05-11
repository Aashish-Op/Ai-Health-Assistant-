from __future__ import annotations

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from db.repository import PatientRepository
from models.retrieval import RetrievalResult, RetrievedChunk

pytestmark = pytest.mark.asyncio


class FakeRetrievalService:
    def __init__(self, chunks: list[RetrievedChunk] | None = None) -> None:
        self.chunks = chunks if chunks is not None else [
            RetrievedChunk(
                chunk_id="111_chunk_0",
                pmid="111",
                title="Diabetes therapy",
                text="Metformin evidence",
                journal="JAMA",
                pub_year=2022,
                vector_score=0.8,
                rerank_score=0.9,
                mesh_terms=["Diabetes"],
            )
        ]

    async def retrieve(self, question: str, patient_context, top_k: int) -> RetrievalResult:
        return RetrievalResult(
            query=question,
            expanded_query=f"{question} Patient conditions: {', '.join(patient_context.get_condition_names()[:3])}",
            chunks=self.chunks[:top_k],
            retrieval_duration_ms=10,
            reranking_duration_ms=5,
            total_candidates=len(self.chunks),
            patient_id=patient_context.patient_id,
        )


class FakePineconeService:
    async def get_index_stats(self) -> dict[str, object]:
        return {
            "total_vector_count": 123,
            "dimension": 3072,
            "index_fullness": 0.01,
            "namespaces": {},
        }


async def persist_patient(db_session: AsyncSession, patient_context) -> None:
    await PatientRepository(db_session).upsert(patient_context)


async def test_retrieve_returns_200_with_valid_patient(
    test_client: AsyncClient,
    test_app,
    db_session: AsyncSession,
    patient_context,
):
    await persist_patient(db_session, patient_context)
    test_app.state.retrieval_service = FakeRetrievalService()
    response = await test_client.post(
        "/query/retrieve",
        json={"patient_id": patient_context.patient_id, "question": "Is metformin safe?"},
    )
    assert response.status_code == 200
    assert response.json()["results"][0]["pmid"] == "111"


async def test_retrieve_returns_404_for_unknown_patient(test_client: AsyncClient, test_app):
    test_app.state.retrieval_service = FakeRetrievalService()
    response = await test_client.post(
        "/query/retrieve",
        json={"patient_id": "missing", "question": "Question?"},
    )
    assert response.status_code == 404


async def test_retrieve_returns_structured_response(
    test_client: AsyncClient,
    test_app,
    db_session: AsyncSession,
    patient_context,
):
    await persist_patient(db_session, patient_context)
    test_app.state.retrieval_service = FakeRetrievalService()
    response = await test_client.post(
        "/query/retrieve",
        json={"patient_id": patient_context.patient_id, "question": "Question?", "top_k": 1},
    )
    payload = response.json()
    assert payload["patient_id"] == patient_context.patient_id
    assert payload["expanded_query"]
    assert payload["results"][0]["rerank_score"] == 0.9


async def test_index_stats_endpoint_returns_vector_count(test_client: AsyncClient, test_app):
    test_app.state.pinecone_service = FakePineconeService()
    response = await test_client.get("/query/index/stats")
    assert response.status_code == 200
    assert response.json()["vector_count"] == 123


async def test_retrieve_with_zero_results_returns_empty_list(
    test_client: AsyncClient,
    test_app,
    db_session: AsyncSession,
    patient_context,
):
    await persist_patient(db_session, patient_context)
    test_app.state.retrieval_service = FakeRetrievalService(chunks=[])
    response = await test_client.post(
        "/query/retrieve",
        json={"patient_id": patient_context.patient_id, "question": "Question?"},
    )
    assert response.status_code == 200
    assert response.json()["results"] == []


async def test_retrieve_request_missing_question_returns_422(test_client: AsyncClient, test_app):
    test_app.state.retrieval_service = FakeRetrievalService()
    response = await test_client.post("/query/retrieve", json={"patient_id": "patient-001"})
    assert response.status_code == 422
