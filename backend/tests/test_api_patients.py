from __future__ import annotations

import json

import pytest
from httpx import AsyncClient


async def upload_sample(client: AsyncClient, fhir_bundle: dict[str, object]):
    """Upload the sample FHIR bundle."""
    data = json.dumps(fhir_bundle).encode("utf-8")
    files = {"file": ("sample_patient.json", data, "application/json")}
    return await client.post("/fhir/patient/load", files=files)


@pytest.mark.asyncio
async def test_upload_valid_fhir_bundle_returns_200(test_client: AsyncClient, fhir_bundle):
    response = await upload_sample(test_client, fhir_bundle)
    assert response.status_code == 200
    payload = response.json()
    assert payload["patient_id"] == "patient-001"
    assert payload["conditions_count"] == 3
    assert payload["medications_count"] == 3
    assert payload["allergies_count"] == 3
    assert payload["labs_count"] == 6
    assert payload["has_critical_labs"] is True


@pytest.mark.asyncio
async def test_upload_invalid_json_returns_422(test_client: AsyncClient):
    response = await test_client.post(
        "/fhir/patient/load",
        files={"file": ("bad.json", b"{bad-json", "application/json")},
    )
    assert response.status_code == 422
    assert response.json()["error_code"] == "FHIR_PARSE_ERROR"
    assert response.json()["request_id"] is not None


@pytest.mark.asyncio
async def test_upload_non_json_file_returns_422(test_client: AsyncClient):
    response = await test_client.post(
        "/fhir/patient/load",
        files={"file": ("bad.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 422
    assert response.json()["error_code"] == "INVALID_FILE_TYPE"


@pytest.mark.asyncio
async def test_upload_duplicate_patient_returns_200(test_client: AsyncClient, fhir_bundle):
    first = await upload_sample(test_client, fhir_bundle)
    second = await upload_sample(test_client, fhir_bundle)
    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["patient_id"] == first.json()["patient_id"]


@pytest.mark.asyncio
async def test_get_patient_list_returns_paginated(test_client: AsyncClient, fhir_bundle):
    await upload_sample(test_client, fhir_bundle)
    response = await test_client.get("/patients?page=1&page_size=10")
    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 1
    assert payload["patients"][0]["full_name"] == "Ada Lovelace"


@pytest.mark.asyncio
async def test_get_patient_list_with_search_filter(test_client: AsyncClient, fhir_bundle):
    await upload_sample(test_client, fhir_bundle)
    response = await test_client.get("/patients?search=Lovelace")
    assert response.status_code == 200
    assert response.json()["total"] == 1

    empty = await test_client.get("/patients?search=NoMatch")
    assert empty.status_code == 200
    assert empty.json()["total"] == 0


@pytest.mark.asyncio
async def test_get_patient_by_id_returns_full_context(test_client: AsyncClient, fhir_bundle):
    await upload_sample(test_client, fhir_bundle)
    response = await test_client.get("/patients/patient-001")
    assert response.status_code == 200
    payload = response.json()
    assert payload["patient_id"] == "patient-001"
    assert len(payload["active_conditions"]) == 3
    assert len(payload["recent_vitals"]) == 6


@pytest.mark.asyncio
async def test_get_patient_by_id_not_found_returns_404(test_client: AsyncClient):
    response = await test_client.get("/patients/missing")
    assert response.status_code == 404
    assert response.json()["error_code"] == "PATIENT_NOT_FOUND"


@pytest.mark.asyncio
async def test_delete_patient_returns_204(test_client: AsyncClient, fhir_bundle):
    await upload_sample(test_client, fhir_bundle)
    response = await test_client.delete("/patients/patient-001")
    assert response.status_code == 204


@pytest.mark.asyncio
async def test_delete_nonexistent_patient_returns_404(test_client: AsyncClient):
    response = await test_client.delete("/patients/missing")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_get_patient_summary_returns_clinical_text(test_client: AsyncClient, fhir_bundle):
    await upload_sample(test_client, fhir_bundle)
    response = await test_client.get("/patients/patient-001/summary")
    assert response.status_code == 200
    payload = response.json()
    assert payload["patient_id"] == "patient-001"
    assert "Ada Lovelace" in payload["clinical_summary"]
    assert "Potassium" in payload["clinical_summary"]


@pytest.mark.asyncio
async def test_health_endpoint_returns_200(test_client: AsyncClient):
    response = await test_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.asyncio
class FakePineconeHealth:
    async def get_index_stats(self) -> dict[str, object]:
        return {"total_vector_count": 1}


@pytest.mark.asyncio
async def test_health_ready_returns_200_when_db_up(
    test_client: AsyncClient,
    test_app,
    monkeypatch: pytest.MonkeyPatch,
):
    import routers.health as health_router

    async def healthy() -> bool:
        return True

    monkeypatch.setattr(health_router, "check_db_connection", healthy)
    test_app.state.pinecone_service = FakePineconeHealth()
    response = await test_client.get("/health/ready")
    assert response.status_code == 200
    assert response.json()["postgres"] is True
    assert response.json()["pinecone"] is True
