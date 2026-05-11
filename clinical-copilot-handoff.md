# FHIR-Integrated Clinical Copilot — Build Handoff Document

**Project type:** AI Clinical Decision Support System  
**Architecture:** Dual-Retrieval RAG (Vector + Knowledge Graph) with FHIR Interoperability  
**Stack:** FastAPI · Next.js · PostgreSQL · Pinecone · Neo4j · deepseek / medgemma model  

---

## Table of Contents

1. [System Architecture Overview](#1-system-architecture-overview)
2. [Repository Structure](#2-repository-structure)
3. [Environment & Prerequisites](#3-environment--prerequisites)
4. [Data Sources & Licensing](#4-data-sources--licensing)
5. [Phase 1 — Data Foundation ](#5-phase-1--data-foundation-weeks-12)
6. [Phase 2 — Vector RAG Pipeline](#6-phase-2--vector-rag-pipeline-weeks-35)
7. [Phase 3 — Knowledge Graph](#7-phase-3--knowledge-graph-weeks-69)
8. [Phase 4 — Dual-Retrieval Fusion & LLM Synthesis (Weeks 10–12)](#8-phase-4--dual-retrieval-fusion--llm-synthesis-weeks-1012)
9. [Phase 5 — Frontend Dashboard](#9-phase-5--frontend-dashboard-weeks-1113)
10. [Phase 6 — Clinical Safety Layer](#10-phase-6--clinical-safety-layer)
11. [Phase 7 — Deployment (AWS)](#11-phase-7--deployment-aws)
12. [API Contract Reference](#12-api-contract-reference)
13. [Inter-Service Data Contracts](#13-inter-service-data-contracts)
14. [Testing Strategy](#14-testing-strategy)
15. [Known Risks & Mitigations](#15-known-risks--mitigations)

---

## 1. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Next.js Dashboard                        │
│         (Patient Profile · Query Interface · Graph View)    │
└────────────────────────┬────────────────────────────────────┘
                         │ REST + SSE streaming
┌────────────────────────▼────────────────────────────────────┐
│                  FastAPI Backend                            │
│            Orchestration · Auth · Streaming                 │
└───────┬────────────────┬──────────────────┬─────────────────┘
        │                │                  │
┌───────▼──────┐ ┌───────▼──────┐ ┌────────▼──────────┐
│ FHIR Parser  │ │  Vector RAG  │ │  Knowledge Graph   │
│  (Synthea)   │ │  (Pinecone)  │ │  (Neo4j/SNOMED)   │
│  PostgreSQL  │ │  PubMed      │ │  DrugBank/RxNorm  │
└───────┬──────┘ └───────┬──────┘ └────────┬──────────┘
        │                │                  │
┌───────▼────────────────▼──────────────────▼─────────────────┐
│                  Dual-Retrieval Fusion                      │
│         Patient Context + Evidence + Graph Facts            │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                   LLM Synthesis                             │
│ MedGemma 1.5 4B (GCP) · DeepSeek-Chat (fallback)      │     │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                Clinical Safety Layer                        │
│       Drug interactions · Allergy flags · Dose checks       │
└─────────────────────────────────────────────────────────────┘
```

### Key design decisions

| Decision | Choice | Reason |
|---|---|---|
| Primary model | Medgemma | Superior structured clinical reasoning |
| Fallback LLM | deepseek | Redundancy + comparison baseline |
| Vector DB | Pinecone (serverless) | Production-grade, free tier sufficient |
| Graph DB | Neo4j Community (Docker) | Free, Cypher is expressive for ontologies |
| Embeddings | `text-embedding-3-large` | Best semantic accuracy for medical text |
| Reranker | Cohere Rerank v3 | Significant precision improvement over raw cosine |
| Patient data | Synthea (synthetic FHIR R4) | Realistic, safe, 1000s of records |
| FHIR standard | HL7 FHIR R4 | Current hospital standard |

---

## 2. Repository Structure

```
clinical-copilot/
├── backend/                        # FastAPI service
│   ├── main.py                     # App entrypoint, router mounting
│   ├── config.py                   # Settings via pydantic-settings
│   ├── models/
│   │   ├── patient.py              # PatientContext, Condition, Medication etc.
│   │   └── api.py                  # Request/response schemas
│   ├── services/
│   │   ├── fhir_parser.py          # FHIR R4 bundle → PatientContext
│   │   ├── pubmed_fetcher.py       # PubMed E-utilities API client
│   │   ├── chunker.py              # Text cleaning + semantic chunking
│   │   ├── embedder.py             # OpenAI embedding with batching
│   │   ├── pinecone_service.py     # Upsert + query operations
│   │   ├── neo4j_service.py        # Graph query service
│   │   ├── fusion.py               # Dual-retrieval context assembler
│   │   ├── llm_service.py          llm_service.py # GCP AI Platform SDK (Primary) + DeepSeek API (Fallback)
│   │   └── safety_layer.py         # Clinical rules engine
│   ├── routers/
│   │   ├── fhir.py                 # /fhir/* endpoints
│   │   ├── query.py                # /query/* endpoints
│   │   └── patients.py             # /patients/* endpoints
│   ├── db/
│   │   ├── session.py              # Async SQLAlchemy session
│   │   ├── models.py               # ORM models
│   │   └── patient_store.py        # DB read/write operations
│   ├── scripts/
│   │   ├── index_pubmed.py         # One-shot indexing script
│   │   ├── load_neo4j.py           # Graph loading script
│   │   └── generate_synthea.sh     # Synthea generation helper
│   ├── requirements.txt
│   └── Dockerfile
│
├── frontend/                       # Next.js application
│   ├── app/
│   │   ├── layout.tsx
│   │   ├── page.tsx                # Patient selector / home
│   │   ├── patients/[id]/
│   │   │   ├── page.tsx            # Patient profile view
│   │   │   └── query/page.tsx      # Clinical query interface
│   │   └── graph/page.tsx          # Knowledge graph visualizer
│   ├── components/
│   │   ├── PatientCard.tsx         # FHIR data display
│   │   ├── ChatInterface.tsx       # Streaming query UI
│   │   ├── GraphViewer.tsx         # D3 force-directed graph
│   │   ├── SafetyBadge.tsx         # Drug interaction flags
│   │   └── SourceCitation.tsx      # RAG source display
│   ├── lib/
│   │   ├── api.ts                  # Backend API client
│   │   └── types.ts                # Shared TypeScript types
│   └── Dockerfile
│
├── data/
│   ├── synthea/                    # Generated FHIR patient files
│   ├── pubmed/                     # Raw PubMed XML cache
│   └── ontologies/                 # DrugBank XML, SNOMED snapshot
│
├── docker-compose.yml              # Local dev: all services
├── docker-compose.prod.yml         # Production overrides
└── .env.example
```

---

## 3. Environment & Prerequisites

### Local tools required

```
Python          3.11+
Node.js         20+
Docker          24+
Java            17+    (for Synthea)
Git
```

### Environment variables (`.env`)

```bash
# LLM APIs
DEEPSEEK_API_KEY=sk-...
GCP_PROJECT_ID=ai-essential
GCP_LOCATION=us-central1
GCP_ENDPOINT_ID=... # Populate after GCP deployment
# Pinecone
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=clinical-copilot
PINECONE_ENVIRONMENT=us-east-1-aws

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=...

# PostgreSQL
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/clinical_copilot

# Cohere (reranker)
COHERE_API_KEY=...

# PubMed
NCBI_EMAIL=you@university.edu
NCBI_API_KEY=...           # optional, gets you 10 req/s instead of 3

# DrugBank
DRUGBANK_API_KEY=...        # free academic license

# App
ENVIRONMENT=development
SECRET_KEY=...             # for JWT if adding auth
```

### `docker-compose.yml` — local services

```yaml
version: "3.9"
services:
  postgres:
    image: postgres:16
    environment:
      POSTGRES_DB: clinical_copilot
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
    ports: ["5432:5432"]
    volumes: ["pgdata:/var/lib/postgresql/data"]

  neo4j:
    image: neo4j:5-community
    environment:
      NEO4J_AUTH: neo4j/yourpassword
      NEO4J_PLUGINS: '["apoc"]'
    ports:
      - "7474:7474"    # Browser UI
      - "7687:7687"    # Bolt protocol
    volumes: ["neo4jdata:/data"]

  backend:
    build: ./backend
    ports: ["8000:8000"]
    env_file: .env
    depends_on: [postgres, neo4j]

  frontend:
    build: ./frontend
    ports: ["3000:3000"]
    environment:
      NEXT_PUBLIC_API_URL: http://localhost:8000

volumes:
  pgdata:
  neo4jdata:
```

---

## 4. Data Sources & Licensing

| Source | What it provides | License | Access |
|---|---|---|---|
| **Synthea** | Synthetic FHIR R4 patient records | Apache 2.0 | GitHub, no registration |
| **PubMed E-utilities** | Medical abstracts via API | Public domain | No key needed (register for higher limits) |
| **RxNorm** | Drug codes + relationships | Public domain (NLM) | Free download, no registration |
| **SNOMED CT** | Clinical terminology ontology | SNOMED license (free for education) | Register at NLM |
| **DrugBank** | Drug-drug interactions, mechanisms | Academic license (free) | Register at drugbank.com |

**Important:** Do not use real patient data at any stage. Synthea exists precisely for this purpose. All demo and testing must use Synthea-generated records.

---

## 5. Phase 1 — Data Foundation (Weeks 1–2)

**Deliverable:** 10000+ synthetic FHIR patients parsed and stored in PostgreSQL, with a working `/fhir/patient/load` endpoint.

### Step 1.1 — Generate Synthea patients

```bash
git clone https://github.com/synthetichealth/synthea
cd synthea
./gradlew build
./run_synthea -p 10000 \
  --exporter.fhir.export=true \
  --exporter.fhir.transaction_bundle=false \
  Massachusetts
```

Output goes to `output/fhir/` — one JSON bundle per patient. Copy these to `data/synthea/`.

### Step 1.2 — Build the FHIR parser

Extract from each bundle: `Patient` (demographics), `Condition` (active only, ICD-10 + SNOMED), `MedicationRequest` (active, RxNorm), `AllergyIntolerance` (SNOMED), `Observation` (labs split from vitals by LOINC code).

Key parsing rules:
- Only include `Condition` resources where `clinicalStatus.coding[0].code` is `active`, `relapse`, or `recurrence`
- Only include `MedicationRequest` where `status == "active"`
- Separate Observations into labs vs. vitals using the LOINC vital signs panel codes (8867-4, 8480-6, 8462-4, etc.)
- Pre-compute `to_clinical_summary()` as a plain text string and store it in PostgreSQL alongside the JSON — this string is what gets injected into prompts at query time (avoids recomputing on every request)

### Step 1.3 — PostgreSQL schema

Two tables are sufficient for Phase 1:

```
patients
  patient_id        TEXT PRIMARY KEY
  first_name        TEXT
  last_name         TEXT
  context_json      JSONB          -- full PatientContext
  clinical_summary  TEXT           -- pre-rendered prompt text
  created_at        TIMESTAMP
  updated_at        TIMESTAMP

patient_uploads
  id                UUID PRIMARY KEY
  patient_id        TEXT REFERENCES patients
  filename          TEXT
  status            TEXT           -- pending | parsed | failed
  error_msg         TEXT
  uploaded_at       TIMESTAMP
```

### Step 1.4 — FastAPI endpoints (Phase 1)

```
POST  /fhir/patient/load         Upload FHIR bundle JSON → parse → store
GET   /patients                  List all patients (paginated)
GET   /patients/{id}             Get PatientContext for a patient
GET   /patients/{id}/summary     Get pre-computed clinical summary text
```

### Phase 1 acceptance criteria

- [ ] 500 Synthea patients parsed without errors
- [ ] All active conditions, medications, and allergies correctly extracted
- [ ] Labs separated from vitals correctly
- [ ] `GET /patients/{id}` returns structured JSON in under 100ms
- [ ] `to_clinical_summary()` produces readable, accurate clinical text

---

## 6. Phase 2 — Vector RAG Pipeline (Weeks 3–5)

**Deliverable:** PubMed abstracts indexed in Pinecone; `/query/retrieve` endpoint returning grounded medical evidence for a clinical question.

### Step 2.1 — Pinecone index setup

Create a serverless index with these settings:
- Dimensions: `3072` (matches `text-embedding-3-large`)
- Metric: `cosine`
- Cloud: AWS, region: `us-east-1`

Metadata schema per vector:
```json
{
  "pmid": "38291045",
  "title": "...",
  "journal": "NEJM",
  "pub_year": 2024,
  "mesh_terms": ["diabetes mellitus", "metformin"],
  "chunk_index": 0,
  "total_chunks": 3,
  "doi": "10.1056/..."
}
```

### Step 2.2 — Indexing pipeline (run as script, not API)

```
PubMed esearch (by MeSH term)
  → pmid list
  → efetch in batches of 100
  → parse XML → PubMedAbstract objects
  → clean text (strip HTML, normalise whitespace)
  → chunk at 512 tokens with 50-token overlap
  → embed with text-embedding-3-large in batches of 100
  → upsert to Pinecone with metadata
```

Target corpus: ~2,000–5,000 abstracts across 10 disease categories. This is enough for a strong demo without hitting cost or rate limits.

MeSH query categories to cover: diabetes type 2, hypertension, heart failure, CKD, depression, COPD, acute coronary syndrome, antibiotic resistance, thyroid disorders, anticoagulation.

### Step 2.3 — Query-time retrieval

At query time, the retrieval call does three things in sequence:

1. **Query expansion:** Augment the user's question with the patient's active conditions. Example: user asks "is this medication safe?" → expand to "medication safety [condition1] [condition2] [patient age]yo". This dramatically improves retrieval relevance.

2. **Pinecone search:** Embed the expanded query, search top-20 chunks.

3. **Cross-encoder reranking:** Pass all 20 (query, chunk) pairs to Cohere Rerank v3. Take top-5. This step typically increases precision significantly — the reranker understands medical co-occurrence that cosine similarity misses.

### Step 2.4 — Output format for fusion step

Each retrieved chunk should carry: `text`, `source` (PMID + title + journal + year), `relevance_score`, `chunk_index`. The fusion step assembles these into the LLM prompt.

### Phase 2 acceptance criteria

- [ ] ≥2,000 PubMed abstracts indexed without duplicates
- [ ] Query returns top-5 reranked chunks in under 2 seconds
- [ ] Retrieved chunks are topically relevant to the clinical question (manual spot-check 20 queries)
- [ ] Source metadata (journal, year, PMID) correctly attached to each result

---

## 7. Phase 3 — Knowledge Graph (Weeks 6–9)

**Deliverable:** Neo4j graph populated with drug-drug interactions and drug-condition contraindications; graph query service returning structured safety facts for a given patient's medication and condition list.

### Step 3.1 — Node types and schema

```cypher
// Node labels
(:Drug   {rxnorm_id, drugbank_id, name, generic_name})
(:Condition  {snomed_id, icd10_id, name})
(:Symptom    {snomed_id, name})

// Relationship types with properties
(:Drug)-[:INTERACTS_WITH {
  severity: "major"|"moderate"|"minor",
  mechanism: String,
  effect: String,
  evidence_level: String
}]->(:Drug)

(:Drug)-[:CONTRAINDICATED_FOR {
  reason: String,
  severity: String
}]->(:Condition)

(:Drug)-[:TREATS {
  evidence_level: String,
  guideline_source: String
}]->(:Condition)

(:Drug)-[:CAUSES_SIDE_EFFECT {
  frequency: String,
  severity: String
}]->(:Symptom)

(:Symptom)-[:INDICATES]->(:Condition)
(:Condition)-[:IS_A]->(:Condition)
(:Condition)-[:COMPLICATES]->(:Condition)
```

### Step 3.2 — Loading sequence

Load in this order — each step depends on the previous:

1. **RxNorm drugs** — create all Drug nodes first (these are the anchor)
2. **DrugBank interactions** — add `INTERACTS_WITH` edges between Drug nodes
3. **DrugBank contraindications** — add `CONTRAINDICATED_FOR` and `TREATS` edges
4. **SNOMED conditions** — filtered to ICD-10 codes seen in your Synthea population only
5. **SNOMED symptoms** — filtered similarly; add `IS_A` and `INDICATES` edges
6. **Side effects** — from DrugBank, add `CAUSES_SIDE_EFFECT` edges

Do not try to load all of SNOMED (~350,000 concepts). Query your PostgreSQL `patients` table first to extract all unique ICD-10 and SNOMED codes that actually appear in your generated patient population. Load only those.

### Step 3.3 — Runtime query pattern

At query time, pass the patient's medication list (as RxNorm codes) and condition list (as SNOMED or ICD-10 codes) and run three Cypher queries in parallel:

- **Drug-drug interactions:** Find all `INTERACTS_WITH` edges between any pair of medications in the patient's list. Filter to `severity IN ["major", "moderate"]`.
- **Contraindications:** Find all `CONTRAINDICATED_FOR` edges from any of the patient's medications to any of their conditions.
- **Side-effect overlap:** Find medications whose `CAUSES_SIDE_EFFECT` output `INDICATES` one of the patient's existing conditions. This surfaces subtle masking — e.g. a drug causing fatigue in a patient with depression.

Return results as structured objects, not free text. These become hard facts in the prompt.

### Step 3.4 — APOC plugin

Install the APOC plugin in Neo4j (it's included in the docker-compose above). You'll need it for batch import from JSON/CSV during the loading scripts, and for `apoc.path.expand` during multi-hop graph traversal.

### Phase 3 acceptance criteria

- [ ] Graph contains ≥ 5,000 Drug nodes with RxNorm IDs
- [ ] ≥ 50,000 `INTERACTS_WITH` relationships (DrugBank has ~130k, filtered to major/moderate)
- [ ] All drug-condition contraindications from DrugBank loaded
- [ ] Graph query for a 5-drug patient returns in under 500ms
- [ ] Contraindication detection correctly flags known dangerous pairs (test with warfarin + aspirin, metformin + contrast dye, etc.)

---

## 8. Phase 4 — Dual-Retrieval Fusion & LLM Synthesis (Weeks 10–12)

**Deliverable:** Single `/query/ask` endpoint that accepts a clinical question + patient ID, runs both retrieval systems in parallel, assembles a grounded prompt, and streams a structured clinical response.

### Step 4.1 — Parallel retrieval

Run the vector search and graph query concurrently using `asyncio.gather`. Do not run them sequentially — both take 200–800ms and sequential execution doubles latency.

```python
vector_results, graph_facts = await asyncio.gather(
    vector_service.retrieve(query, patient_context),
    graph_service.query_patient(patient_context),
)
```

### Step 4.2 — Prompt assembly structure

Assemble the prompt in this exact order — the LLM performs best when patient context comes first (it anchors interpretation), then evidence, then hard safety facts, then the question.

```
SYSTEM:
You are a clinical decision support assistant. You receive structured patient data,
evidence-based medical literature, and verified clinical safety information. You do
not diagnose. You surface relevant evidence and flag safety concerns to support
clinical decision-making. Always cite your sources.

USER:
--- PATIENT CONTEXT ---
{patient_context.to_clinical_summary()}

--- EVIDENCE-BASED LITERATURE (from PubMed, ranked by relevance) ---
[1] {chunk_1.title} ({chunk_1.journal}, {chunk_1.year}) — {chunk_1.text}
[2] ...

--- CLINICAL SAFETY FACTS (verified from drug database) ---
{graph_facts formatted as bullet list}

--- SAFETY LAYER FLAGS ---
{safety_layer_output}

--- CLINICAL QUESTION ---
{user_question}

Respond with:
1. Relevant clinical considerations (cite literature by number)
2. Patient-specific factors from their record
3. Safety flags (if any — must list ALL flagged interactions)
4. Suggested next steps
5. Limitations of this assessment
```

### Step 4.3 — LLM call with streaming

Use Google Cloud AI Platform SDK to call the MedGemma endpoint as primary. Implement a try/except block that falls back to the DeepSeek API (using the standard openai Python package pointing to https://api.deepseek.com) if the GCP endpoint is spun down or returns an error. Stream the response back to the client via Server-Sent Events (SSE)."
### Step 4.4 — Response schema

Even though the response streams as text, enforce structured output by prompting for numbered sections (as above). The frontend parses these sections to render the response with appropriate formatting — safety flags render as red badges, citations render as clickable source cards.

### Phase 4 acceptance criteria

- [ ] End-to-end query (patient load → dual retrieval → LLM → response) completes in under 10 seconds
- [ ] First token streams within 3 seconds of query submission
- [ ] All retrieved literature correctly cited in response
- [ ] All graph safety flags appear in response when contraindications exist
- [ ] Response degrades gracefully if Neo4j is unavailable (falls back to vector-only)

---

## 9. Phase 5 — Frontend Dashboard (Weeks 11–13)

**Deliverable:** Next.js application with three functional views.

### View 1 — Patient list (`/`)

Simple searchable list of loaded patients. Each row shows: name, age, number of active conditions, number of active medications. Click to open patient profile.

### View 2 — Patient profile (`/patients/[id]`)

Four sections:
- **Demographics card:** name, age, gender, DOB
- **Active conditions:** list with ICD-10 codes, onset dates
- **Current medications:** list with dosage, any flagged interactions shown as amber/red badges
- **Recent labs:** table with reference range and flag column

At the bottom: a query input box. This is the entry point to View 3.

### View 3 — Clinical query interface (`/patients/[id]/query`)

Split-panel layout:
- **Left:** The streaming LLM response. Safety flags render first as sticky alert cards before the main response text. Citations render as numbered superscripts linked to source cards.
- **Right:** Source panel showing the retrieved PubMed abstracts with title, journal, year, and relevance score. Clicking a source highlights the citation in the response text.

Implement streaming with the `EventSource` browser API. Show a typing indicator while waiting for first token.

### View 4 — Knowledge graph visualizer (`/graph`)

D3.js force-directed graph. Show the drug-condition-symptom subgraph for the currently selected patient. Nodes colored by type (Drug = blue, Condition = coral, Symptom = amber). Edges colored by relationship type (red for contraindication, orange for interaction, green for treats).

Use `d3-force` with charge, link, and center forces. Keep it to the patient's immediate subgraph (2 hops max) — rendering the full graph is not useful.

### Frontend stack

```
Next.js 14+         App Router
TypeScript          Strict mode
Tailwind CSS        Styling
shadcn/ui           Component primitives
D3.js v7            Graph visualization
EventSource API     SSE streaming
```

---

## 10. Phase 6 — Clinical Safety Layer

**Deliverable:** Python rules engine that validates LLM output and patient context independently of the graph, providing a deterministic safety net.

### Rule categories

The safety layer runs *before* the LLM (its output is injected into the prompt as a flag section) and the results are non-negotiable — the LLM cannot override them.

**Drug-allergy cross-check:** For every active medication, check if the drug class appears in the patient's allergy list. Use a pre-built drug class mapping (penicillins, cephalosporins, sulfonamides, NSAIDs, etc.) to catch class-level allergies, not just exact matches.

**Renal dose adjustment:** Cross-check active medications against a list of renally-cleared drugs that require dose reduction. If the patient has a CKD condition and takes metformin, contrast agents, NSAIDs, or aminoglycosides, flag immediately with severity level.

**Age-based flags:** Pediatric dosing if age < 18. Geriatric flags (Beers Criteria high-risk medications) if age ≥ 65.

**Duplicate therapy:** Flag when two medications in the same therapeutic class are both active — e.g. two ACE inhibitors, two antidepressants of the same class.

**Pregnancy contraindications:** If gender is female and age is 15–50, flag all Category X medications (warfarin, isotretinoin, thalidomide, etc.) with a note that pregnancy status should be confirmed.

### Safety layer output format

```json
{
  "flags": [
    {
      "severity": "critical",
      "type": "drug_allergy",
      "message": "Patient allergic to penicillin — amoxicillin is a penicillin-class antibiotic",
      "medication": "amoxicillin",
      "allergy": "penicillin",
      "recommendation": "Consider azithromycin or doxycycline as alternatives"
    }
  ],
  "flag_count": 1,
  "highest_severity": "critical"
}
```

---

## 11. Phase 7 — Deployment (AWS)

### Architecture

```
Route 53
  → CloudFront (CDN for Next.js static assets)
  → ALB (Application Load Balancer)
      → ECS Fargate: Next.js frontend (2 tasks)
      → ECS Fargate: FastAPI backend (2 tasks)
  → RDS PostgreSQL (db.t3.small)
  → EC2 t3.medium: Neo4j Community
  → Pinecone (managed — no deployment needed)
```

### Containerisation

Both services have Dockerfiles. The backend Dockerfile:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
```

Use ECR to store images. Build and push with `aws ecr get-login-password` + `docker push`.

### ECS task sizing

| Service | CPU | Memory | Tasks |
|---|---|---|---|
| FastAPI backend | 512 | 1024 MB | 2 |
| Next.js frontend | 256 | 512 MB | 2 |
| Neo4j (EC2) | 2 vCPU | 4 GB | 1 |

### Secrets management

Use AWS Secrets Manager for all API keys. Mount as environment variables in ECS task definitions — never bake secrets into container images or commit to Git.

---

## 12. API Contract Reference

### FHIR endpoints

```
POST  /fhir/patient/load
  Body: multipart/form-data — file: FHIR bundle JSON
  Response: { patient_id, status, conditions_count, medications_count }

GET   /patients
  Query: page=1&limit=20&search=string
  Response: { patients: [...], total, page }

GET   /patients/{id}
  Response: PatientContext (full JSON)

GET   /patients/{id}/summary
  Response: { patient_id, clinical_summary: string }
```

### Query endpoints

```
POST  /query/ask
  Body: { patient_id: string, question: string, stream: boolean }
  Response: EventStream (SSE) or JSON

POST  /query/retrieve
  Body: { patient_id: string, question: string }
  Response: {
    vector_results: [{ text, source, score }],
    graph_facts: [{ type, severity, message }],
    safety_flags: [{ severity, type, message }]
  }
```

### Health endpoints

```
GET  /health             → { status: "ok", version }
GET  /health/db          → { postgres: bool, neo4j: bool, pinecone: bool }
```

---

## 13. Inter-Service Data Contracts

### PatientContext (core shared type)

```typescript
interface PatientContext {
  patient_id: string
  first_name: string
  last_name: string
  birth_date: string       // ISO date
  gender: string
  age: number
  active_conditions: Condition[]
  active_medications: Medication[]
  allergies: Allergy[]
  recent_labs: LabResult[]
  recent_vitals: Vital[]
}
```

### ClinicalQueryResponse (SSE event shape)

```typescript
// Events emitted on the SSE stream:
{ event: "safety_flags", data: SafetyFlag[] }   // emitted first, always
{ event: "token", data: string }                  // LLM stream tokens
{ event: "sources", data: Source[] }              // emitted after completion
{ event: "done", data: { tokens_used: number } }
```

---

## 14. Testing Strategy

### Unit tests

- FHIR parser: test against 5–10 known Synthea bundles, assert correct field extraction
- Safety layer: test all rule categories against hand-crafted patient fixtures with known violations
- Chunker: assert chunk sizes never exceed 512 tokens, overlap is correct

### Integration tests

- Full pipeline: load patient → query → assert response contains citations and safety section
- Graph queries: assert known drug-drug interactions are detected (warfarin + aspirin, SSRIs + MAOIs)
- Pinecone retrieval: assert that diabetes query returns diabetes-relevant chunks

### Evaluation metrics

- **RAG precision:** Manual review of 50 queries — are top-5 chunks topically relevant? Target ≥80%.
- **Safety recall:** Against a test set of 20 patients with planted contraindications — does the safety layer catch all of them? Must be 100%.
- **Latency:** End-to-end query under 10 seconds. First token under 3 seconds.

---

## 15. Known Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| PubMed rate limiting during indexing | High | Low | Use NCBI API key, add sleep(0.35) between batches |
| SNOMED CT license approval delay | Medium | High | Start registration in Week 1. Use ICD-10 + DrugBank only as fallback. |
| Neo4j memory pressure on t3.medium | Medium | Medium | Filter SNOMED to patient population only; cap heap at 2GB in neo4j.conf |
| Pinecone cold-start latency on free tier | Low | Medium | Use serverless tier; pre-warm with a dummy query on service start |
| LLM hallucination in clinical context | High | High | Safety layer is deterministic; graph facts are injected as hard constraints, not suggestions |
| Synthea data not reflecting real clinical complexity | Low | Low | Known limitation — document in project write-up; use 500+ patients for diversity |

---

## Quick-start checklist (Day 1)

- [ ] Clone repo, copy `.env.example` → `.env`, fill in API keys
- [ ] `docker compose up -d` — starts PostgreSQL and Neo4j
- [ ] Register at NCBI for PubMed API key
- [ ] Apply for DrugBank academic license (takes 1–2 days)
- [ ] Register at NLM for SNOMED CT access (takes 1–3 days)
- [ ] Generate Synthea patients: `./scripts/generate_synthea.sh`
- [ ] Run FHIR loader: `python scripts/load_patients.py data/synthea/`
- [ ] Verify: `curl http://localhost:8000/patients` returns patient list

---

*Document version: 1.0 — generated for coursework build reference*  
*Stack versions: FastAPI 0.111 · Next.js 14 · Neo4j 5 · Pinecone serverless · Claude claude-sonnet-4-20250514*
