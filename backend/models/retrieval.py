from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class RetrievalModel(BaseModel):
    """Base retrieval schema."""

    model_config = ConfigDict(frozen=True)


class PubMedAbstract(RetrievalModel):
    pmid: str
    title: str
    abstract: str
    authors: list[str]
    journal: str
    pub_year: int
    mesh_terms: list[str]
    doi: str | None = None
    fetched_at: datetime


class TextChunk(RetrievalModel):
    chunk_id: str
    pmid: str
    title: str
    text: str
    token_count: int
    chunk_index: int
    total_chunks: int
    journal: str
    pub_year: int
    mesh_terms: list[str]
    doi: str | None = None


class EmbeddedChunk(RetrievalModel):
    chunk: TextChunk
    embedding: list[float]


class RetrievedChunk(RetrievalModel):
    chunk_id: str
    pmid: str
    title: str
    text: str
    journal: str
    pub_year: int
    doi: str | None = None
    mesh_terms: list[str] = Field(default_factory=list)
    vector_score: float
    rerank_score: float | None = None

    @property
    def citation(self) -> str:
        """Return a formatted citation string for prompt injection.

        Args:
            None.

        Returns:
            Citation with title, journal, year, PMID, and DOI when present.

        Raises:
            None.
        """
        doi = f", doi:{self.doi}" if self.doi else ""
        return f"{self.title} ({self.journal}, {self.pub_year}) PMID:{self.pmid}{doi}"


class RetrievalResult(RetrievalModel):
    query: str
    expanded_query: str
    chunks: list[RetrievedChunk]
    retrieval_duration_ms: int
    reranking_duration_ms: int
    total_candidates: int
    patient_id: str | None = None
