from __future__ import annotations

from datetime import datetime

from models.retrieval import PubMedAbstract
from services.chunker import TextChunker


def make_abstract(text: str) -> PubMedAbstract:
    return PubMedAbstract(
        pmid="123",
        title="Diabetes medication safety",
        abstract=text,
        authors=["Smith JA"],
        journal="JAMA",
        pub_year=2024,
        mesh_terms=["Diabetes Mellitus"],
        doi="10.1000/test",
        fetched_at=datetime.utcnow(),
    )


def test_short_abstract_yields_single_chunk():
    chunks = TextChunker(512, 50).chunk_abstract(make_abstract("Metformin lowers A1c."))
    assert len(chunks) == 1


def test_long_abstract_yields_multiple_chunks():
    text = " ".join(["hypertension pharmacotherapy"] * 80)
    chunks = TextChunker(40, 5).chunk_abstract(make_abstract(text))
    assert len(chunks) > 1


def test_chunk_overlap_is_correct():
    text = " ".join(f"token{i}" for i in range(100))
    chunker = TextChunker(30, 8)
    chunks = chunker.chunk_abstract(make_abstract(text))
    first_body = chunks[0].text.split("\n\n", 1)[1]
    second_body = chunks[1].text.split("\n\n", 1)[1]
    first_tokens = chunker._encoding.encode(first_body)
    second_tokens = chunker._encoding.encode(second_body)
    assert first_tokens[-8:] == second_tokens[:8]


def test_title_prepended_to_every_chunk():
    chunks = TextChunker(10, 2).chunk_abstract(make_abstract(" ".join(["heart failure"] * 20)))
    assert all(chunk.text.startswith("Diabetes medication safety\n\n") for chunk in chunks)


def test_html_stripped_from_abstract():
    chunks = TextChunker(512, 50).chunk_abstract(make_abstract("<b>Metformin</b> &amp; safety"))
    assert "<b>" not in chunks[0].text
    assert "&amp;" not in chunks[0].text


def test_chunk_id_format_is_correct():
    chunk = TextChunker(512, 50).chunk_abstract(make_abstract("Clinical text"))[0]
    assert chunk.chunk_id == "123_chunk_0"


def test_token_count_accurate_for_medical_text():
    chunker = TextChunker(512, 50)
    chunk = chunker.chunk_abstract(make_abstract("HbA1c, eGFR, and SGLT2 inhibitors were reviewed."))[0]
    body = chunk.text.split("\n\n", 1)[1]
    assert chunk.token_count == chunker.token_count(body)


def test_empty_abstract_yields_no_chunks():
    assert TextChunker(512, 50).chunk_abstract(make_abstract(" \n\t ")) == []


def test_chunk_metadata_populated_from_abstract():
    chunk = TextChunker(512, 50).chunk_abstract(make_abstract("Clinical text"))[0]
    assert chunk.pmid == "123"
    assert chunk.journal == "JAMA"
    assert chunk.pub_year == 2024
    assert chunk.mesh_terms == ["Diabetes Mellitus"]
