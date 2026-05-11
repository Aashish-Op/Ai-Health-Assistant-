from __future__ import annotations

import asyncio
import hashlib
import re
from collections.abc import AsyncIterator
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode
from xml.etree import ElementTree

import aiofiles
import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from models.retrieval import PubMedAbstract
from services.logging_config import get_logger

PUBMED_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
HTTP_TIMEOUT = httpx.Timeout(30.0, connect=10.0)
MESH_QUERIES: list[str] = [
    "diabetes mellitus type 2 treatment management",
    "hypertension pharmacotherapy guidelines",
    "heart failure management therapy",
    "chronic kidney disease treatment",
    "major depressive disorder treatment",
    "COPD exacerbation management",
    "acute coronary syndrome therapy",
    "antibiotic resistance clinical management",
    "thyroid disorder treatment guidelines",
    "anticoagulation therapy clinical guidelines",
]


class PubMedFetcher:
    """Fetch and parse PubMed abstracts through NCBI E-utilities."""

    def __init__(self, email: str, api_key: str | None, cache_dir: Path) -> None:
        """Create a PubMed fetcher.

        Args:
            email: NCBI contact email.
            api_key: Optional NCBI API key for higher rate limits.
            cache_dir: Directory for raw XML response cache files.

        Returns:
            None.

        Raises:
            ValueError: If email is empty.
        """
        if not email:
            raise ValueError("NCBI_EMAIL is required for PubMed fetching")
        self.email = email
        self.api_key = api_key or None
        self.cache_dir = cache_dir
        self._logger = get_logger(__name__)

    async def fetch_all(self, max_per_query: int) -> AsyncIterator[PubMedAbstract]:
        """Fetch abstracts for every configured MeSH query.

        Args:
            max_per_query: Maximum PMIDs to fetch per query.

        Returns:
            Async iterator yielding parsed PubMed abstracts.

        Raises:
            httpx.HTTPError: If NCBI requests fail after retries.
        """
        async with httpx.AsyncClient(base_url=PUBMED_BASE_URL, timeout=HTTP_TIMEOUT) as client:
            for query in MESH_QUERIES:
                self._logger.info("pubmed_query_started", query=query, max_results=max_per_query)
                pmids = await self._search(client, query, max_per_query)
                yielded = 0
                for start in range(0, len(pmids), 100):
                    batch = pmids[start : start + 100]
                    abstracts = await self._fetch_batch(client, batch)
                    self._logger.debug(
                        "pubmed_batch_fetched",
                        query=query,
                        batch_start=start,
                        pmid_count=len(batch),
                        abstract_count=len(abstracts),
                    )
                    for abstract in abstracts:
                        yielded += 1
                        yield abstract
                    await asyncio.sleep(self._rate_limit_seconds)
                self._logger.info(
                    "pubmed_query_completed",
                    query=query,
                    pmid_count=len(pmids),
                    abstract_count=yielded,
                )

    async def _search(self, client: httpx.AsyncClient, query: str, max_results: int) -> list[str]:
        """Search PubMed for relevant PMIDs.

        Args:
            client: Shared HTTP client.
            query: MeSH/free-text query.
            max_results: Maximum PMID count.

        Returns:
            PMID strings in relevance order.

        Raises:
            httpx.HTTPError: If NCBI requests fail after retries.
        """
        params = self._base_params(
            {
                "db": "pubmed",
                "term": f"({query}) AND hasabstract[text] AND freetext[filter]",
                "retmax": str(max_results),
                "retmode": "xml",
                "sort": "relevance",
                "mindate": "2015",
                "maxdate": "2025",
            }
        )
        xml_text = await self._request_text(client, "esearch.fcgi", params)
        root = ElementTree.fromstring(xml_text)
        return [self._node_text(node) for node in root.findall(".//Id") if self._node_text(node)]

    async def _fetch_batch(
        self,
        client: httpx.AsyncClient,
        pmids: list[str],
    ) -> list[PubMedAbstract]:
        """Fetch and parse a batch of PubMed abstracts.

        Args:
            client: Shared HTTP client.
            pmids: PubMed IDs to fetch.

        Returns:
            Parsed abstracts with empty abstracts skipped.

        Raises:
            httpx.HTTPError: If NCBI requests fail after retries.
        """
        if not pmids:
            return []
        params = self._base_params(
            {
                "db": "pubmed",
                "id": ",".join(pmids),
                "rettype": "abstract",
                "retmode": "xml",
            }
        )
        xml_text = await self._request_text(client, "efetch.fcgi", params)
        return self._parse_xml_response(xml_text)

    def _parse_xml_response(self, xml_text: str) -> list[PubMedAbstract]:
        """Parse a PubMed efetch XML response.

        Args:
            xml_text: XML response body from PubMed.

        Returns:
            Parsed abstracts.

        Raises:
            xml.etree.ElementTree.ParseError: If XML is malformed.
        """
        root = ElementTree.fromstring(xml_text)
        abstracts: list[PubMedAbstract] = []
        for article in root.findall(".//PubmedArticle"):
            pmid = self._node_text(article.find("./MedlineCitation/PMID"))
            title = self._iter_text(article.find("./MedlineCitation/Article/ArticleTitle"))
            abstract_text = self._abstract_text(article)
            if not abstract_text:
                self._logger.debug("pubmed_abstract_skipped_empty", pmid=pmid, title=title[:80])
                continue
            abstracts.append(
                PubMedAbstract(
                    pmid=pmid,
                    title=title,
                    abstract=abstract_text,
                    authors=self._authors(article),
                    journal=self._journal(article),
                    pub_year=self._pub_year(article),
                    mesh_terms=self._mesh_terms(article),
                    doi=self._doi(article),
                    fetched_at=datetime.utcnow(),
                )
            )
        return abstracts

    @retry(
        retry=retry_if_exception_type(httpx.HTTPError),
        wait=wait_exponential(min=1, max=60),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def _request_text(
        self,
        client: httpx.AsyncClient,
        endpoint: str,
        params: dict[str, str],
    ) -> str:
        """Return cached response text or fetch it from NCBI.

        Args:
            client: Shared HTTP client.
            endpoint: E-utilities endpoint name.
            params: Request query parameters.

        Returns:
            Raw XML response text.

        Raises:
            httpx.HTTPError: If NCBI returns an error after retries.
        """
        cache_path = self._cache_path(endpoint, params)
        if cache_path.exists():
            async with aiofiles.open(cache_path, encoding="utf-8") as cached:
                return await cached.read()

        response = await client.get(endpoint, params=params, timeout=HTTP_TIMEOUT)
        response.raise_for_status()
        text = response.text
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(cache_path, "w", encoding="utf-8") as cached:
            await cached.write(text)
        return text

    def _base_params(self, extra: dict[str, str]) -> dict[str, str]:
        params = {"tool": "clinical_copilot", "email": self.email, **extra}
        if self.api_key:
            params["api_key"] = self.api_key
        return params

    def _cache_path(self, endpoint: str, params: dict[str, str]) -> Path:
        key = f"{endpoint}?{urlencode(sorted(params.items()))}"
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.xml"

    @property
    def _rate_limit_seconds(self) -> float:
        return 0.35 if self.api_key else 1.1

    @staticmethod
    def _abstract_text(article: ElementTree.Element) -> str:
        sections = []
        for node in article.findall("./MedlineCitation/Article/Abstract/AbstractText"):
            label = node.attrib.get("Label")
            text = PubMedFetcher._iter_text(node)
            if not text:
                continue
            sections.append(f"{label}: {text}" if label else text)
        return "\n".join(sections).strip()

    @staticmethod
    def _authors(article: ElementTree.Element) -> list[str]:
        authors = []
        for author in article.findall("./MedlineCitation/Article/AuthorList/Author"):
            last_name = PubMedFetcher._node_text(author.find("LastName"))
            initials = PubMedFetcher._node_text(author.find("Initials"))
            if last_name:
                authors.append(f"{last_name} {initials}".strip())
        return authors

    @staticmethod
    def _journal(article: ElementTree.Element) -> str:
        journal = PubMedFetcher._node_text(
            article.find("./MedlineCitation/Article/Journal/ISOAbbreviation")
        )
        if journal:
            return journal
        return PubMedFetcher._node_text(article.find("./MedlineCitation/Article/Journal/Title"))

    @staticmethod
    def _pub_year(article: ElementTree.Element) -> int:
        year = PubMedFetcher._node_text(
            article.find("./MedlineCitation/Article/Journal/JournalIssue/PubDate/Year")
        )
        if year.isdigit():
            return int(year)
        medline_date = PubMedFetcher._node_text(
            article.find("./MedlineCitation/Article/Journal/JournalIssue/PubDate/MedlineDate")
        )
        match = re.search(r"\d{4}", medline_date)
        return int(match.group(0)) if match else datetime.utcnow().year

    @staticmethod
    def _mesh_terms(article: ElementTree.Element) -> list[str]:
        return [
            PubMedFetcher._node_text(node)
            for node in article.findall("./MedlineCitation/MeshHeadingList/MeshHeading/DescriptorName")
            if PubMedFetcher._node_text(node)
        ]

    @staticmethod
    def _doi(article: ElementTree.Element) -> str | None:
        for node in article.findall("./PubmedData/ArticleIdList/ArticleId"):
            if node.attrib.get("IdType") == "doi":
                text = PubMedFetcher._node_text(node)
                return text or None
        return None

    @staticmethod
    def _iter_text(node: ElementTree.Element | None) -> str:
        if node is None:
            return ""
        return " ".join("".join(node.itertext()).split())

    @staticmethod
    def _node_text(node: ElementTree.Element | None) -> str:
        return "" if node is None or node.text is None else node.text.strip()
