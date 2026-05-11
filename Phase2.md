# Phase 2 Build Prompt — Vector RAG Pipeline
### Paste this entire prompt into your agentic IDE composer window.

---

## ROLE & ENGINEERING STANDARD

Same standard as Phase 1. Senior Staff Engineer. Production healthcare system. Every file reviewed by the most senior engineers before touching a system used by real clinicians. No shortcuts, no hacks, no half-measures.

Phase 2 builds the intelligence layer on top of Phase 1's data foundation. Where Phase 1 was about structured data correctness, Phase 2 is about retrieval precision and latency. The two failure modes here are: retrieving irrelevant literature (hurts clinical quality) and slow retrieval (hurts usability). Both are unacceptable. Every architectural decision must optimise against one or both.

---

## PROJECT CONTEXT

Phase 1 is complete. You have:
- `PatientContext` Pydantic model with `to_clinical_summary()` and helper methods
- PostgreSQL with `patients` table and pre-computed `clinical_summary` text
- `GET /patients/{id}` endpoint returning full `PatientContext`
- `FHIRParser` producing clean structured clinical data

Phase 2 adds the **Vector RAG pipeline** — a two-phase system:

**Indexing phase (offline, run once as a script):**
PubMed E-utilities API → clean and chunk abstracts → embed with OpenAI → upsert to Pinecone

**Query phase (online, per user request):**
Clinical question + patient conditions → query expansion → embed → Pinecone similarity search (top-20) → Cohere cross-encoder reranking (top-5) → structured evidence context for LLM

Read the attached `clinical-copilot-handoff.md` Phase 2 section fully before writing any code.

**New tech added this phase:**
- `openai` SDK (embeddings only — LLM synthesis is Phase 4)
- `pinecone-client` v3+ (serverless)
- `cohere` SDK (reranker)
- `httpx` async (already installed — for PubMed API calls)
- `tiktoken` (token counting for chunking)
- `tenacity` (retry logic for external API calls)
- `aiofiles` (async file I/O for caching)

Add all to `backend/requirements.txt` with pinned versions.

---

## WHAT YOU ARE BUILDING — FULL DELIVERABLE LIST

```
backend/services/pubmed_fetcher.py
backend/services/chunker.py
backend/services/embedder.py
backend/services/pinecone_service.py
backend/services/reranker.py
backend/services/retrieval_service.py
backend/routers/query.py
backend/scripts/index_pubmed.py
backend/tests/test_pubmed_fetcher.py
backend/tests/test_chunker.py
backend/tests/test_retrieval_service.py
backend/tests/test_api_query.py
```

Plus modifications to:
```
backend/config.py             — add Phase 2 env vars
backend/main.py               — mount query router
backend/requirements.txt      — add new dependencies
```

---

## CONFIGURATION — `backend/config.py`

Add these fields to the existing `Settings` class:

```
openai_api_key: str
pinecone_api_key: str
pinecone_index_name: str = "clinical-copilot"
pinecone_environment: str = "us-east-1-aws"
cohere_api_key: str
ncbi_email: str
ncbi_api_key: str | None = None     # optional — raises rate limit from 3 to 10 req/s
embedding_model: str = "text-embedding-3-large"
embedding_dimensions: int = 3072
reranker_model: str = "rerank-v3.5"
reranker_top_n: int = 5
retrieval_top_k: int = 20           # candidates before reranking
chunk_size_tokens: int = 512
chunk_overlap_tokens: int = 50
pubmed_max_per_query: int = 200
```

All optional in development — use `Field(default="")` with a validator that logs a warning if empty when the retrieval router is called.

---

## DATA CONTRACTS — DEFINE THESE FIRST

Before building any service, define these Pydantic models in `backend/models/retrieval.py`. Every service in this phase communicates through these types — no raw dicts between layers.

```python
class PubMedAbstract(BaseModel):
    pmid: str
    title: str
    abstract: str
    authors: list[str]
    journal: str
    pub_year: int
    mesh_terms: list[str]
    doi: str | None = None
    fetched_at: datetime

class TextChunk(BaseModel):
    chunk_id: str           # f"{pmid}_chunk_{index}"
    pmid: str
    title: str
    text: str               # the actual chunk content
    token_count: int
    chunk_index: int
    total_chunks: int
    journal: str
    pub_year: int
    mesh_terms: list[str]
    doi: str | None = None

class EmbeddedChunk(BaseModel):
    chunk: TextChunk
    embedding: list[float]  # 3072-dim for text-embedding-3-large

class RetrievedChunk(BaseModel):
    chunk_id: str
    pmid: str
    title: str
    text: str
    journal: str
    pub_year: int
    doi: str | None = None
    mesh_terms: list[str]
    vector_score: float     # raw cosine score from Pinecone
    rerank_score: float | None = None   # set after reranking

    @property
    def citation(self) -> str:
        """Returns formatted citation string for prompt injection."""

class RetrievalResult(BaseModel):
    query: str
    expanded_query: str
    chunks: list[RetrievedChunk]    # top-N after reranking
    retrieval_duration_ms: int
    reranking_duration_ms: int
    total_candidates: int
    patient_id: str | None = None
```

Also add to `backend/models/api.py`:

```python
class RetrieveRequest(BaseModel):
    patient_id: str
    question: str
    top_k: int = 5      # how many final chunks to return

class RetrieveResponse(BaseModel):
    patient_id: str
    question: str
    expanded_query: str
    results: list[RetrievedChunk]
    retrieval_duration_ms: int
    reranking_duration_ms: int
    total_candidates: int
```

---

## SERVICE SPECIFICATIONS

### `backend/services/pubmed_fetcher.py`

Purpose: Fetch abstracts from PubMed E-utilities API by MeSH term. Rate-limit safe. Resumable. Idempotent.

**Constants:**

```
PUBMED_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

MESH_QUERIES: list[str] — ten disease-category queries:
  "diabetes mellitus type 2 treatment management"
  "hypertension pharmacotherapy guidelines"
  "heart failure management therapy"
  "chronic kidney disease treatment"
  "major depressive disorder treatment"
  "COPD exacerbation management"
  "acute coronary syndrome therapy"
  "antibiotic resistance clinical management"
  "thyroid disorder treatment guidelines"
  "anticoagulation therapy clinical guidelines"
```

**`PubMedFetcher` class:**

Constructor: `(email: str, api_key: str | None, cache_dir: Path)`

The cache is critical — PubMed fetching takes 20–40 minutes for 2000 abstracts. Cache raw XML responses to disk so the indexing script is resumable if it crashes. Cache key: SHA256 of the request URL. If cache hit, deserialise from disk. If miss, fetch and write to disk before returning.

Methods to implement:

`fetch_all(max_per_query: int) -> AsyncIterator[PubMedAbstract]`
Iterates over all MESH_QUERIES. For each: search → get PMIDs → fetch in batches of 100 → parse XML → yield abstracts. Rate limit: sleep `0.35s` between batch requests if api_key present, `1.1s` if not. Log progress: INFO per query start/complete, DEBUG per batch.

`_search(client, query, max_results) -> list[str]`
Calls `esearch.fcgi`. Returns list of PMIDs. Filter: `hasabstract[text]`, `freetext[filter]`, `mindate=2015`, `maxdate=2025`, `sort=relevance`.

`_fetch_batch(client, pmids: list[str]) -> list[PubMedAbstract]`
Calls `efetch.fcgi` with `rettype=abstract&retmode=xml`. Parses XML response. Any abstract with empty text is skipped and logged at DEBUG.

`_parse_xml_response(xml_text: str) -> list[PubMedAbstract]`
Parses `PubmedArticleSet` XML. For each `PubmedArticle`:
- PMID: `MedlineCitation/PMID`
- Title: `Article/ArticleTitle`
- Abstract: `Article/Abstract/AbstractText` — join multiple sections with newline if structured abstract
- Authors: `Article/AuthorList/Author` — format as "LastName Initials"
- Journal: `Article/Journal/ISOAbbreviation`
- Year: `Article/Journal/JournalIssue/PubDate/Year` — fall back to MedlineDate parsing
- MeSH: `MeshHeadingList/MeshHeading/DescriptorName`
- DOI: `ArticleIdList/ArticleId[@IdType="doi"]`

Wrap retry logic with `tenacity`: `retry=retry_if_exception_type(httpx.HTTPError)`, `wait=wait_exponential(min=1, max=60)`, `stop=stop_after_attempt(3)`.

---

### `backend/services/chunker.py`

Purpose: Split `PubMedAbstract` objects into `TextChunk` objects suitable for embedding. Token-accurate chunking using `tiktoken`.

**Key design decisions to implement:**

Use `tiktoken.encoding_for_model("text-embedding-3-large")` for token counting. Do not estimate tokens with word count — it will be inaccurate for medical terminology.

Prepend title to every chunk: `f"{abstract.title}\n\n{chunk_text}"`. This ensures every chunk carries enough context to be retrieved even when the question matches the title but not the abstract body.

Chunk strategy: sliding window. Start at token 0, step by `chunk_size - overlap`. For abstracts shorter than `chunk_size`, yield as a single chunk — do not pad.

Clean text before chunking:
- Strip HTML tags (some PubMed abstracts contain `<b>`, `<i>` markup)
- Normalise Unicode (NFD → NFC)
- Collapse multiple whitespace to single space
- Remove control characters
- Do NOT strip medical abbreviations or expand them — the embedding model handles these

`TextChunker` class:

Constructor: `(chunk_size: int, overlap: int)`

`chunk_abstract(abstract: PubMedAbstract) -> list[TextChunk]`
Returns list of `TextChunk`. Sets `chunk_id = f"{abstract.pmid}_chunk_{i}"`. Populates all metadata from the source abstract. Token count on the chunk text (excluding the prepended title).

`chunk_abstracts(abstracts: list[PubMedAbstract]) -> list[TextChunk]`
Calls `chunk_abstract` for each. Logs total chunks produced at INFO.

---

### `backend/services/embedder.py`

Purpose: Convert `TextChunk` objects to `EmbeddedChunk` objects using OpenAI's embedding API. Batch-optimised. Rate-limit safe.

**`EmbeddingService` class:**

Constructor: `(api_key: str, model: str, dimensions: int)`

`embed_chunks(chunks: list[TextChunk]) -> list[EmbeddedChunk]`
Batches chunks into groups of 100 (OpenAI limit). For each batch, calls `client.embeddings.create()`. Maps results back to chunks by index. Returns `EmbeddedChunk` list in same order as input.

`embed_query(text: str) -> list[float]`
Embeds a single query string. Used at retrieval time. Must use the same model and dimensions as `embed_chunks` — inconsistency here silently breaks retrieval.

Use `tenacity` retry on `openai.RateLimitError` and `openai.APIError`: exponential backoff, max 5 attempts. Log each retry attempt at WARNING with the error type and attempt number.

Log embedding cost estimate at INFO after each batch: `estimated_tokens`, `batch_size`, `cumulative_total`. Token estimate: split on whitespace and multiply by 1.3.

---

### `backend/services/pinecone_service.py`

Purpose: Upsert `EmbeddedChunk` objects to Pinecone at index time; query Pinecone with an embedding vector at retrieval time.

**`PineconeService` class:**

Constructor: `(api_key: str, index_name: str)`

Initialise the Pinecone client and connect to the serverless index. Store the index handle as `self._index`. Log at INFO on successful connection including index stats (total vector count, dimension).

`upsert_chunks(chunks: list[EmbeddedChunk]) -> int`
Upserts in batches of 100. Each vector:
- `id`: `chunk.chunk.chunk_id`
- `values`: `chunk.embedding`
- `metadata`: `{ pmid, title, journal, pub_year, doi, mesh_terms (list), chunk_index, total_chunks }`

Metadata must be kept under 40KB per vector (Pinecone limit). Truncate `title` to 200 chars and mesh_terms to first 10 if necessary.

Returns total vectors upserted. Log progress at INFO: batch number, cumulative upserted, index size after upsert.

`query(embedding: list[float], top_k: int, filter: dict | None) -> list[dict]`
Queries the index. Returns raw Pinecone match objects. Supports optional metadata filter dict (used later when filtering by pub_year or mesh_term).

`get_index_stats() -> dict`
Returns `describe_index_stats()` result. Used by health endpoint and indexing script.

`delete_by_pmid(pmid: str) -> int`
Deletes all vectors for a PMID. Used when re-indexing updated abstracts.

---

### `backend/services/reranker.py`

Purpose: Take a list of retrieved candidate chunks and re-score them using Cohere's cross-encoder reranker. Return top-N by rerank score.

**`RerankerService` class:**

Constructor: `(api_key: str, model: str, top_n: int)`

`rerank(query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]`
Calls `cohere.Client.rerank()` with:
- `query`: the expanded clinical query
- `documents`: list of `chunk.text` strings
- `model`: `self.model`
- `top_n`: `self.top_n`

Map rerank scores back to the input chunks by index. Set `chunk.rerank_score`. Return top-N sorted by rerank score descending.

If Cohere API is unavailable (network error, auth error), log ERROR and fall back to returning the top-N chunks sorted by `vector_score` instead. Do not raise — degraded retrieval is better than a 500 error.

Log at INFO: original candidate count, top_n after reranking, top score, bottom score, reranking duration.

---

### `backend/services/retrieval_service.py`

Purpose: Orchestrates the full query-time pipeline. The only service the router calls directly.

**`RetrievalService` class:**

Constructor: `(embedder: EmbeddingService, pinecone: PineconeService, reranker: RerankerService)`

`retrieve(question: str, patient_context: PatientContext | None, top_k: int) -> RetrievalResult`

This method does the following in order:

1. **Query expansion:** If `patient_context` is provided, augment the raw question with the patient's top-3 active condition names appended as context. Example: raw question "is this medication safe?" becomes "is this medication safe? Patient conditions: type 2 diabetes mellitus, hypertension, chronic kidney disease stage 3". This dramatically improves retrieval relevance without changing the semantic intent.

2. **Embed expanded query:** Call `embedder.embed_query(expanded_query)`. Time this call.

3. **Pinecone search:** Call `pinecone.query(embedding, top_k=config.retrieval_top_k)`. Converts raw Pinecone matches to `RetrievedChunk` objects (populate `vector_score` from match score, `rerank_score=None`). Time this call.

4. **Rerank:** Call `reranker.rerank(expanded_query, candidates)`. Updates `rerank_score` on returned chunks. Time this call.

5. **Return `RetrievalResult`:** Include the question, expanded query, final chunks, all timing fields, total candidate count, patient_id if provided.

Run steps 2 and 3 — the embed and the Pinecone call — sequentially (embed must complete before query). Do NOT parallelise them; they are dependent.

Log the full retrieval at INFO: patient_id, question (truncated to 100 chars), candidates retrieved, final chunks returned, total duration.

---

### `backend/routers/query.py`

Mount at `/query` in `main.py`.

**`POST /query/retrieve`**

Request body: `RetrieveRequest`
Response: `RetrieveResponse`

Steps:
1. Load `PatientContext` from DB using patient_id (call `PatientRepository.get_by_id()`, deserialise from context_json)
2. If patient not found, raise `PatientNotFoundError` → 404
3. Instantiate services from app state (not per-request construction — see dependency injection note below)
4. Call `retrieval_service.retrieve(question, patient_context, top_k)`
5. Return `RetrieveResponse`

**Dependency injection pattern for services:**
Services like `EmbeddingService`, `PineconeService`, `RerankerService` are expensive to instantiate (they open connections and load configs). Do not construct them inside the route handler or in `Depends`. Instead, attach them to `app.state` in the `lifespan` function in `main.py`, and create a `Depends` function `get_retrieval_service()` that reads from `app.state`. This means one instance per application lifetime, not per request.

**`GET /query/index/stats`**

Returns Pinecone index stats: vector count, dimension, index fullness. Used to monitor indexing progress. Response: `{ vector_count: int, dimension: int, index_fullness: float, namespaces: dict }`.

**`GET /health/ready`** (modify existing)

Add Pinecone connectivity check alongside the existing PostgreSQL check. Return `{ postgres: bool, pinecone: bool, message: str }`.

---

### `backend/scripts/index_pubmed.py`

This is the one-shot indexing script. It is not part of the API — run it manually before Phase 3 begins.

CLI interface using `argparse`:
```
--max-per-query INT     Max abstracts per MeSH query (default: 200)
--cache-dir PATH        Directory to cache raw PubMed XML (default: data/pubmed/cache)
--workers INT           Concurrent embedding batches (default: 2)
--dry-run               Fetch and chunk but do not upsert to Pinecone
--resume                Skip PMIDs already present in Pinecone (check by querying existing IDs)
--stats                 Print index stats and exit
```

Execution flow:
1. Print config summary (index name, embedding model, max abstracts, etc.)
2. Instantiate all services
3. Fetch all abstracts via `PubMedFetcher.fetch_all()` — stream as async generator
4. Chunk each abstract immediately as it arrives (don't buffer all abstracts in memory)
5. Accumulate chunks into batches of 100, embed each batch, upsert to Pinecone
6. Every 500 chunks: log progress summary and write checkpoint to `data/pubmed/checkpoint.json`
7. On completion: print final report — total abstracts fetched, total chunks indexed, total tokens estimated, total time, Pinecone index size

Checkpoint format: `{ "completed_pmids": [...], "total_chunks": int, "last_updated": ISO datetime }`

If `--resume` flag: load checkpoint, skip PMIDs already in `completed_pmids`.

If the script crashes mid-run: on restart with `--resume`, it picks up from the checkpoint without re-fetching already-indexed abstracts. This is critical — PubMed fetching takes 30+ minutes and must be resumable.

---

## TESTING SPECIFICATIONS

### `backend/tests/test_pubmed_fetcher.py`

All tests mock `httpx.AsyncClient` — never make real network calls in tests.

```
test_search_returns_pmid_list
test_fetch_batch_parses_abstracts_correctly
test_structured_abstract_sections_joined_with_newline
test_missing_abstract_text_skipped_not_raised
test_rate_limiting_respected_between_batches     (assert sleep called with correct interval)
test_cache_hit_skips_network_call
test_xml_parse_handles_missing_doi_gracefully
test_retry_on_http_error
```

Fixture: `sample_pubmed_xml.xml` — a realistic PubMed efetch XML response with 3 articles, one structured abstract, one missing DOI, one with multiple MeSH terms.

### `backend/tests/test_chunker.py`

No mocking needed — pure logic.

```
test_short_abstract_yields_single_chunk
test_long_abstract_yields_multiple_chunks
test_chunk_overlap_is_correct
test_title_prepended_to_every_chunk
test_html_stripped_from_abstract
test_chunk_id_format_is_correct         (f"{pmid}_chunk_{i}")
test_token_count_accurate_for_medical_text
test_empty_abstract_yields_no_chunks
test_chunk_metadata_populated_from_abstract
```

### `backend/tests/test_retrieval_service.py`

Mock `EmbeddingService`, `PineconeService`, `RerankerService` with `unittest.mock.AsyncMock`.

```
test_query_expansion_appends_patient_conditions
test_query_expansion_skipped_when_no_patient
test_pinecone_results_converted_to_retrieved_chunks
test_reranked_chunks_sorted_by_rerank_score
test_reranker_failure_falls_back_to_vector_score
test_retrieval_result_contains_timing_fields
test_empty_pinecone_results_returns_empty_chunks
test_patient_conditions_capped_at_three_in_expansion
```

### `backend/tests/test_api_query.py`

Mock `RetrievalService` at the app state level.

```
async def test_retrieve_returns_200_with_valid_patient
async def test_retrieve_returns_404_for_unknown_patient
async def test_retrieve_returns_structured_response
async def test_index_stats_endpoint_returns_vector_count
async def test_retrieve_with_zero_results_returns_empty_list
async def test_retrieve_request_missing_question_returns_422
```

---

## CODE QUALITY STANDARDS — SAME AS PHASE 1, PLUS THESE

**External API calls must always have timeouts.** Every `httpx` call: `timeout=httpx.Timeout(30.0, connect=10.0)`. Every OpenAI call: `timeout=60.0`. Every Cohere call: `timeout=30.0`. A hung external call must never hang a user request indefinitely.

**Memory management during indexing.** The indexing script processes up to 5000 abstracts × 3 chunks each = 15,000 chunks. Do not accumulate all embeddings in memory simultaneously. Process in streaming batches: fetch → chunk → embed → upsert → discard. Peak memory usage should stay under 500MB.

**Idempotent upserts.** Calling the indexing script twice must not duplicate vectors in Pinecone. Pinecone upsert is idempotent by vector ID — since chunk IDs are deterministic (`{pmid}_chunk_{i}`), re-running the script with the same data is safe.

**No API keys in logs.** When logging service initialisation, log the first 8 characters of API keys at DEBUG for debugging (e.g., `"sk-ant-ap..."`) — never the full key.

**Graceful degradation.** If Pinecone is unavailable at startup, log ERROR but do not crash the application. Mark the retrieval service as unavailable in app state. The `/query/retrieve` endpoint returns 503 with a clear error message. The FHIR and patient endpoints must continue working.

---

## BUILD ORDER — FOLLOW THIS SEQUENCE

```
1.  backend/models/retrieval.py           (data contracts first)
2.  backend/config.py                     (add Phase 2 fields)
3.  backend/requirements.txt              (add new dependencies)
4.  backend/services/pubmed_fetcher.py
5.  backend/services/chunker.py
6.  backend/services/embedder.py
7.  backend/services/pinecone_service.py
8.  backend/services/reranker.py
9.  backend/services/retrieval_service.py
10. backend/routers/query.py
11. backend/main.py                       (mount query router, attach services to app.state in lifespan)
12. backend/scripts/index_pubmed.py
13. backend/tests/fixtures/sample_pubmed_xml.xml
14. backend/tests/test_pubmed_fetcher.py
15. backend/tests/test_chunker.py
16. backend/tests/test_retrieval_service.py
17. backend/tests/test_api_query.py
```

---

## DO NOT DO ANY OF THE FOLLOWING

- Do not call the OpenAI or PubMed APIs in tests — mock everything
- Do not construct services inside route handlers — use app.state
- Do not buffer all 15,000 chunks in memory — process in streaming batches
- Do not use synchronous OpenAI or Cohere clients — use async where available, `asyncio.to_thread()` where not
- Do not hardcode MeSH queries — define them as a named constant, not inline strings
- Do not ignore Pinecone upsert errors — log and raise
- Do not skip the `--resume` flag in the indexing script — it is required for a production-grade tool
- Do not use cosine similarity as the final ranking signal — reranking is mandatory
- Do not return more than `top_n` chunks after reranking — the fusion step in Phase 4 expects exactly this many
- Do not log the full `abstract` text — log `pmid` and `title[:80]` only
- Do not swallow `tenacity.RetryError` silently — log it at ERROR with the original exception

---

## ACCEPTANCE CRITERIA — PHASE 2 IS COMPLETE WHEN

1. `pytest -x` passes — all Phase 1 tests still green, all Phase 2 tests green
2. `pytest --cov=backend --cov-report=term-missing` shows ≥75% on all new service files
3. `python scripts/index_pubmed.py --dry-run --max-per-query 5` runs without error and logs chunk counts
4. `python scripts/index_pubmed.py --max-per-query 10` successfully upserts vectors to Pinecone and logs final index size > 0
5. `POST /query/retrieve` with a valid patient_id and question returns a `RetrieveResponse` with `results` list containing at least 1 chunk
6. Each `RetrievedChunk` in the response has: non-empty `text`, non-null `rerank_score`, valid `pmid`, `journal`, `pub_year`
7. `GET /query/index/stats` returns `vector_count > 0` after indexing
8. `GET /health/ready` returns both `postgres: true` and `pinecone: true`
9. Running `index_pubmed.py` twice produces the same Pinecone vector count (idempotency)
10. Killing the indexing script mid-run and restarting with `--resume` does not re-fetch already-cached PMIDs
11. If `COHERE_API_KEY` is invalid, retrieval degrades to vector-score ranking — does not 500
12. Query expansion is verified in logs: the expanded query string contains patient condition names

---

## COMPLETION OUTPUT

When all files are built and all criteria pass, output:

```
## Phase 2 Complete

### Files created
[list every file with relative path]

### Files modified
[list modified files]

### Test results
[paste pytest -x output]

### Coverage
[paste coverage for new service files]

### Index stats
[paste GET /query/index/stats response after running index_pubmed.py]

### Sample retrieval
[paste one example POST /query/retrieve request + response]

### Deviations from spec
[any intentional deviations and justification]
```