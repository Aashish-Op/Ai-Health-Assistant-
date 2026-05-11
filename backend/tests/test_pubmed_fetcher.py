from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest

from models.retrieval import PubMedAbstract
from services.pubmed_fetcher import PubMedFetcher


class FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text

    def raise_for_status(self) -> None:
        return None


def sample_xml() -> str:
    return (Path(__file__).parent / "fixtures" / "sample_pubmed_xml.xml").read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_search_returns_pmid_list(tmp_path):
    fetcher = PubMedFetcher("research@example.edu", None, tmp_path)
    client = type("Client", (), {"get": AsyncMock(return_value=FakeResponse("<eSearchResult><IdList><Id>1</Id><Id>2</Id></IdList></eSearchResult>"))})()
    assert await fetcher._search(client, "diabetes", 2) == ["1", "2"]


@pytest.mark.asyncio
async def test_fetch_batch_parses_abstracts_correctly(tmp_path):
    fetcher = PubMedFetcher("research@example.edu", None, tmp_path)
    client = type("Client", (), {"get": AsyncMock(return_value=FakeResponse(sample_xml()))})()
    abstracts = await fetcher._fetch_batch(client, ["11111111", "22222222"])
    assert len(abstracts) == 3
    assert abstracts[0].pmid == "11111111"
    assert abstracts[0].authors == ["Smith JA", "Patel R"]
    assert abstracts[0].doi == "10.1056/example.111"


def test_structured_abstract_sections_joined_with_newline(tmp_path):
    fetcher = PubMedFetcher("research@example.edu", None, tmp_path)
    abstract = fetcher._parse_xml_response(sample_xml())[0]
    assert "BACKGROUND:" in abstract.abstract
    assert "\nMETHODS:" in abstract.abstract
    assert "\nRESULTS:" in abstract.abstract


def test_missing_abstract_text_skipped_not_raised(tmp_path):
    fetcher = PubMedFetcher("research@example.edu", None, tmp_path)
    xml = "<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>1</PMID><Article><ArticleTitle>No abstract</ArticleTitle></Article></MedlineCitation></PubmedArticle></PubmedArticleSet>"
    assert fetcher._parse_xml_response(xml) == []


@pytest.mark.asyncio
async def test_rate_limiting_respected_between_batches(monkeypatch: pytest.MonkeyPatch, tmp_path):
    fetcher = PubMedFetcher("research@example.edu", None, tmp_path)
    monkeypatch.setattr(fetcher, "_search", AsyncMock(return_value=[str(i) for i in range(101)]))
    monkeypatch.setattr(
        fetcher,
        "_fetch_batch",
        AsyncMock(return_value=[PubMedAbstract(
            pmid="1",
            title="Title",
            abstract="Text",
            authors=[],
            journal="JAMA",
            pub_year=2020,
            mesh_terms=[],
            fetched_at=__import__("datetime").datetime.utcnow(),
        )]),
    )
    sleep = AsyncMock()
    monkeypatch.setattr("services.pubmed_fetcher.asyncio.sleep", sleep)
    client = object()
    results = []
    async for abstract in fetcher.fetch_all(101):
        results.append(abstract)
        if len(results) == 2:
            break
    assert sleep.await_args_list[0].args[0] == 1.1


@pytest.mark.asyncio
async def test_cache_hit_skips_network_call(tmp_path):
    fetcher = PubMedFetcher("research@example.edu", None, tmp_path)
    params = {"db": "pubmed", "id": "1"}
    path = fetcher._cache_path("efetch.fcgi", params)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("<cached />", encoding="utf-8")
    client = type("Client", (), {"get": AsyncMock(side_effect=AssertionError("network called"))})()
    assert await fetcher._request_text(client, "efetch.fcgi", params) == "<cached />"


def test_xml_parse_handles_missing_doi_gracefully(tmp_path):
    fetcher = PubMedFetcher("research@example.edu", None, tmp_path)
    abstracts = fetcher._parse_xml_response(sample_xml())
    assert abstracts[1].doi is None


@pytest.mark.asyncio
async def test_retry_on_http_error(tmp_path):
    fetcher = PubMedFetcher("research@example.edu", None, tmp_path)
    client = type(
        "Client",
        (),
        {
            "get": AsyncMock(
                side_effect=[
                    httpx.ConnectError("temporary"),
                    FakeResponse("<ok />"),
                ]
            )
        },
    )()
    assert await fetcher._request_text(client, "efetch.fcgi", {"db": "pubmed", "id": "1"}) == "<ok />"
    assert client.get.await_count == 2
