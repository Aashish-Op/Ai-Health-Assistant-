from __future__ import annotations

import html
import re
import unicodedata

import tiktoken

from models.retrieval import PubMedAbstract, TextChunk
from services.logging_config import get_logger


class TextChunker:
    """Token-aware chunker for PubMed abstracts."""

    def __init__(self, chunk_size: int, overlap: int) -> None:
        """Create a text chunker.

        Args:
            chunk_size: Maximum tokens in each abstract body chunk.
            overlap: Number of body tokens to overlap between adjacent chunks.

        Returns:
            None.

        Raises:
            ValueError: If chunk sizing is invalid.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be non-negative and smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap
        try:
            self._encoding = tiktoken.encoding_for_model("text-embedding-3-large")
        except KeyError:
            self._encoding = tiktoken.get_encoding("cl100k_base")
        self._logger = get_logger(__name__)

    def chunk_abstract(self, abstract: PubMedAbstract) -> list[TextChunk]:
        """Split one PubMed abstract into embedding chunks.

        Args:
            abstract: Source PubMed abstract.

        Returns:
            Chunk list with deterministic IDs.

        Raises:
            None.
        """
        cleaned = self.clean_text(abstract.abstract)
        if not cleaned:
            return []

        tokens = self._encoding.encode(cleaned)
        step = self.chunk_size - self.overlap
        windows = [
            tokens[start : start + self.chunk_size]
            for start in range(0, len(tokens), step)
            if tokens[start : start + self.chunk_size]
        ]

        chunks: list[TextChunk] = []
        total_chunks = len(windows)
        for index, window in enumerate(windows):
            body = self._encoding.decode(window)
            chunks.append(
                TextChunk(
                    chunk_id=f"{abstract.pmid}_chunk_{index}",
                    pmid=abstract.pmid,
                    title=abstract.title,
                    text=f"{abstract.title}\n\n{body}",
                    token_count=len(window),
                    chunk_index=index,
                    total_chunks=total_chunks,
                    journal=abstract.journal,
                    pub_year=abstract.pub_year,
                    mesh_terms=abstract.mesh_terms,
                    doi=abstract.doi,
                )
            )
        return chunks

    def chunk_abstracts(self, abstracts: list[PubMedAbstract]) -> list[TextChunk]:
        """Split multiple abstracts into embedding chunks.

        Args:
            abstracts: PubMed abstracts to chunk.

        Returns:
            Flattened chunk list.

        Raises:
            None.
        """
        chunks: list[TextChunk] = []
        for abstract in abstracts:
            chunks.extend(self.chunk_abstract(abstract))
        self._logger.info("abstracts_chunked", abstract_count=len(abstracts), chunk_count=len(chunks))
        return chunks

    def token_count(self, text: str) -> int:
        """Return exact token count for configured embedding tokenizer.

        Args:
            text: Text to tokenize.

        Returns:
            Token count.

        Raises:
            None.
        """
        return len(self._encoding.encode(text))

    @staticmethod
    def clean_text(text: str) -> str:
        """Clean PubMed abstract text without changing medical meaning.

        Args:
            text: Raw abstract text.

        Returns:
            Normalized text.

        Raises:
            None.
        """
        decoded = html.unescape(text)
        without_tags = re.sub(r"<[^>]+>", " ", decoded)
        normalized = unicodedata.normalize("NFC", without_tags)
        without_controls = "".join(
            char for char in normalized if char == "\n" or unicodedata.category(char)[0] != "C"
        )
        return re.sub(r"\s+", " ", without_controls).strip()
