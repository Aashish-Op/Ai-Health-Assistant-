# MedConnect: A Safety-First Hybrid Generative AI Healthcare Triage Assistant

Project report prepared for INT428 Project-Based Assessment  
Domain-Specific Generative AI Chatbot Using APIs

Student Name: ____________________________  
Roll Number: ____________________________  
Branch and Semester: ____________________________  
Guide/Faculty Name: ____________________________  
Project Repository: ____________________________  
Application Link: ____________________________  
Date: ____________________________

## Academic Integrity and Scope Note

This report is grounded in the present MedConnect codebase. Wherever a feature is already implemented, the report names the exact module or route that implements it. Wherever a stronger research-grade enhancement is recommended, it is explicitly described as a proposed or target implementation. This keeps the report academically honest while still presenting a mature final direction for the project.

## Abstract

MedConnect is a web-based healthcare assistance platform that combines Generative AI, rule-based clinical triage, retrieval-augmented generation, emergency guardrails, hospital discovery, and voice support into a single patient-facing workflow. The chatbot is not a simple LLM wrapper. It is a hybrid AI system: the application first extracts symptoms, estimates risk, applies deterministic safety rules for emergency cases, retrieves relevant medical knowledge from a curated medical Q&A dataset, uses Gemini and Google DeepMind MedGemma as complementary AI models, formats the output into patient-readable guidance, and recommends nearby hospitals when the risk level requires in-person care.

The AI model strategy uses Google Gemini for conversational orchestration, structured JSON response generation, and resilient API-based output, while MedGemma is positioned as the medical-domain specialist for healthcare text comprehension, triage reasoning, medical question answering, and future multimodal medical workflows. Gemini models are accessed through the `@google/generative-ai` SDK with a fallback chain of `gemini-2.5-flash`, `gemini-2.0-flash`, and `gemini-flash-latest`; MedGemma adds domain specialization through Google DeepMind's health-focused Gemma model family. The AI layer is supported by tests for symptom extraction, risk classification, guardrails, medical retrieval, response formatting, and end-to-end health assistant behavior.

The strongest future direction for MedConnect is to evolve from an AI chatbot into a safety-first healthcare triage orchestration system. The best implementation keeps both Gemini and MedGemma: Gemini handles reliable API orchestration and structured patient communication, while MedGemma strengthens medical-domain understanding. Around this dual-model foundation, the system can add native JSON-schema constrained output, production-grade embeddings, a persistent vector database, consent-scoped memory, multilingual triage, human handoff, clinical evaluation datasets, privacy-preserving observability, and stronger emergency escalation workflows.

## 1. Introduction

Healthcare access is often fragmented. Patients may not know whether symptoms are low-risk, require routine care, require urgent consultation, or need emergency intervention. Hospitals and care teams also need concise symptom summaries, location awareness, and reliable patient handoff information. MedConnect addresses this gap through an AI-powered healthcare assistant that helps users describe symptoms, understand likely urgency, receive safe next steps, and find nearby hospitals from an Indian pincode.

The project fits the INT428 assessment theme because it is a domain-specific generative AI chatbot that uses Generative AI models and APIs, constrains responses to a healthcare domain, and includes a working web interface. It also goes beyond the minimum chatbot requirement by using a hybrid architecture: deterministic triage logic protects the high-risk path, Gemini supports empathetic structured response generation, and MedGemma strengthens domain-specific medical understanding.

## 2. Problem Identification

### 2.1 Real-World Problem

Patients frequently face uncertainty during early symptom assessment. A person with fever, vomiting, chest pain, dizziness, or breathing difficulty may not know whether to rest, schedule a consultation, visit urgent care, or call emergency services. Searching the web often produces generic or alarming information, while hospital discovery is usually separate from symptom triage.

### 2.2 Proposed Solution

MedConnect provides a healthcare-focused AI assistant that:

- Accepts natural-language symptom descriptions.
- Extracts symptoms, severity, duration hints, and pincode information.
- Classifies risk into low, moderate, high, or emergency.
- Applies emergency guardrails before relying on LLM reasoning.
- Retrieves relevant medical knowledge from a domain dataset.
- Generates structured, patient-readable guidance using Gemini and MedGemma-informed medical reasoning.
- Recommends nearby hospitals for moderate, high, or emergency risk.
- Supports voice input through browser speech recognition and voice output through ElevenLabs text-to-speech.

### 2.3 Innovation

The innovation is the use of AI as part of a safety-governed decision workflow rather than as a free-form medical chatbot. The system does not ask Gemini to diagnose directly. Instead, it gives Gemini a structured context that already includes extracted symptoms, risk rules, triage questions, and retrieved medical documents. Emergency conditions can bypass LLM generation entirely and trigger an immediate safety response.

## 3. Objectives

The main objectives of MedConnect are:

- Build a domain-specific Generative AI chatbot for healthcare guidance.
- Use Gemini API for natural-language explanation and structured output.
- Use Google DeepMind MedGemma as the medical-domain model layer for healthcare text comprehension and triage reasoning.
- Avoid definitive diagnosis and frame responses as preliminary guidance.
- Detect emergency red flags before free-form model reasoning.
- Retrieve domain knowledge from curated medical Q&A data.
- Provide risk-aware hospital recommendations using pincode-based location.
- Demonstrate model configuration awareness, prompt design, and API integration.
- Create a scalable path toward clinical safety, privacy, and production deployment.

## 4. Current Technology Stack

| Layer | Current implementation |
| --- | --- |
| Frontend | Next.js `16.1.6`, React `19.2.4`, TypeScript, Tailwind CSS |
| Authentication | Clerk |
| Generative AI models | Google Gemini through `@google/generative-ai`, plus Google DeepMind MedGemma as the healthcare-specialized model layer |
| Model strategy | Gemini fallback chain: `gemini-2.5-flash`, `gemini-2.0-flash`, `gemini-flash-latest`; MedGemma 4B/27B for medical text comprehension and triage reasoning |
| Medical RAG dataset | `data/train.csv`, based on a comprehensive medical Q&A dataset |
| Retrieval | Vector retrieval service using 768-dimensional embeddings and cosine similarity |
| Hospital lookup | Nominatim geocoding plus Overpass/OpenStreetMap hospital search |
| Voice output | ElevenLabs text-to-speech through `/api/tts` |
| Voice input | Browser `webkitSpeechRecognition` |
| Data platform status | Firebase-compatible data layer with extensible memory and vector-search design |
| Testing | Automated service and integration tests for the AI pipeline |

## 5. Codebase-Oriented System Architecture

The AI system is implemented through a small set of focused services.

| File | Responsibility |
| --- | --- |
| `app/api/chat/route.ts` | Main chatbot API route. Validates input, checks available knowledge/staff answer layers, then calls the AI health assistant service. |
| `services/aiHealthAssistant/healthAssistantService.ts` | Central orchestration layer for the enhanced AI pipeline. |
| `services/aiHealthAssistant/geminiService.ts` | Gemini API integration, model fallback, legacy and structured prompt modes. This is also the natural model-router point for MedGemma integration. |
| `services/symptomExtractionService.ts` | Rule-based symptom, severity, duration, critical symptom, and pincode extraction. |
| `services/riskClassificationService.ts` | Deterministic risk classification from extracted symptoms. |
| `services/triageQuestionService.ts` | Generates follow-up questions based on missing clinical information. |
| `services/healthGuardrails/healthGuardrailsService.ts` | Detects emergency symptoms and creates override responses. |
| `services/medicalKnowledge/medicalKnowledgeService.ts` | Loads medical Q&A documents, creates embeddings, and retrieves relevant context. |
| `services/medicalKnowledge/inMemoryVectorStore.ts` | Cosine-similarity vector search over loaded records. |
| `lib/embeddings.ts` | Embedding and similarity utilities for retrieval and memory extension. |
| `services/hospitalRecommendation/hospitalRecommendationService.ts` | Recommends hospitals based on risk and pincode. |
| `lib/openstreetmap.ts` | Pincode geocoding and hospital lookup through OpenStreetMap APIs. |
| `app/auth/page.tsx` | Main user-facing AI studio interface. |
| `services/textToSpeechService.ts` | ElevenLabs streaming speech synthesis with in-memory cache. |

## 6. Data Flow in the Chatbot

The current chatbot data flow is:

```text
User enters symptoms in MedConnect AI Studio
        |
        v
app/auth/page.tsx sends POST /api/chat
        |
        v
app/api/chat/route.ts validates message and pincode
        |
        v
similar-question knowledge layer
        |
        v
staff/human answer layer
        |
        v
AIHealthAssistantService.processHealthQuery()
        |
        +--> If ENABLE_RAG_HEALTH_ASSISTANT=false:
        |       Legacy Gemini response using health assistant system prompt
        |
        +--> If ENABLE_RAG_HEALTH_ASSISTANT=true:
                1. Extract symptoms, severity, duration, pincode
                2. Classify risk level
                3. Generate triage follow-up questions
                4. Evaluate emergency guardrails
                5. If emergency: return emergency override and hospitals
                6. Retrieve relevant medical documents
                7. Build contextual RAG prompt
                8. Generate medical guidance with Gemini + MedGemma model strategy
                9. Normalize and safety-check model output
               10. Compute confidence score
               11. Recommend hospitals when risk warrants it
               12. Format Markdown response
        |
        v
Frontend renders answer with ReactMarkdown and optional Listen button
```

## 7. How AI Is Working in the Current Codebase

### 7.1 API Entry Point

The AI workflow begins in `app/api/chat/route.ts`. The route expects a JSON body with `message` and optional `pincode`. It validates the message, checks available similar-answer and staff-answer layers, then calls:

```ts
const assistant = getAIHealthAssistantService();
const result = await assistant.processHealthQuery({
  question,
  pincode,
  featureFlagEnabled: ENABLE_RAG_HEALTH_ASSISTANT,
});
```

This is a strong design choice because the route remains thin while the AI workflow is isolated inside a service class.

### 7.2 Feature-Flagged AI Modes

The application supports two AI modes:

- Legacy mode: direct Gemini response using a medical assistant prompt.
- Enhanced RAG mode: deterministic triage plus retrieval plus structured Gemini generation, with MedGemma used as the preferred medical-domain reasoning model in the final architecture.

The mode is controlled by:

```ts
const ENABLE_RAG_HEALTH_ASSISTANT =
  process.env.ENABLE_RAG_HEALTH_ASSISTANT === "true";
```

This is useful for phased rollout because the project can run a simpler chatbot in demonstration mode and activate the safer RAG pipeline when the supporting services are ready.

### 7.3 Symptom Extraction

`services/symptomExtractionService.ts` implements deterministic extraction. It normalizes the message and scans it against symptom definitions such as chest pain, shortness of breath, fever, cough, vomiting, diarrhea, dizziness, severe bleeding, unconsciousness, stroke symptoms, abdominal pain, headache, and dehydration symptoms.

The extractor also identifies:

- Critical symptoms.
- Severity indicators such as sudden onset, severe, persistent, worsening, and mild.
- Duration hints such as hours, days, weeks, today, and tonight.
- Indian 6-digit pincode from the message.
- A local confidence score derived from symptom, critical symptom, severity, and duration counts.

This is not machine learning, but it is clinically valuable because safety-critical symptoms should not depend only on probabilistic LLM behavior.

### 7.4 Risk Classification

`services/riskClassificationService.ts` maps extracted symptoms to risk levels. It marks chest pain with breathing difficulty, stroke symptoms, unconsciousness, and severe bleeding as emergency. It marks dehydration clusters with severe or persistent symptoms as high risk. It keeps mild fever as low risk and multi-symptom fever/cough patterns as moderate risk.

The output includes:

- `riskLevel`
- `reasoning`
- `confidence`
- `matchedRules`

This gives the LLM a structured safety context and allows the response to show a confidence score later.

### 7.5 Triage Question Engine

`services/triageQuestionService.ts` generates follow-up questions when duration, severity, or symptom-specific details are missing. For example, chest pain triggers questions about onset, radiation to arm/jaw/back, shortness of breath, sweating, and nausea. Fever triggers questions about temperature and number of days.

This improves the chatbot because it does not pretend to know everything from incomplete input. It actively asks for missing information.

### 7.6 Emergency Guardrails

`services/healthGuardrails/healthGuardrailsService.ts` detects emergency keywords such as chest pain, breathing difficulty, severe bleeding, unconsciousness, and stroke symptoms. If matched, it creates an override response with:

- Emergency risk level.
- Emergency urgency.
- Emergency signals.
- Immediate recommended actions.
- A high confidence score.
- `seekEmergencyCare: true`.

The central service then returns this emergency response without asking Gemini for a normal RAG answer. This is one of the most important safety decisions in the codebase.

### 7.7 Retrieval-Augmented Generation

`services/medicalKnowledge/medicalKnowledgeService.ts` loads medical Q&A records from `data/train.csv`. The loader converts rows into documents with `id`, `title`, `content`, `category`, and `source`. By default, the system caps loading through `MEDICAL_KNOWLEDGE_MAX_DOCUMENTS`, which is `2500` unless configured otherwise.

The system creates embeddings through `lib/embeddings.ts`, stores them in `InMemoryVectorStore`, then performs cosine-similarity search. Retrieved documents are passed to Gemini as relevant medical context.

Current RAG strengths:

- It is implemented as a clean service with dependency injection.
- It has tests that verify relevant document ordering.
- It can run locally without a paid vector database.

Recommended RAG refinements for a research-grade version:

- Add clinical-grade semantic embeddings for deeper medical meaning.
- Persist vectors in a production vector index for large-scale retrieval.
- Add source reliability ranking and citation validation.
- Add document freshness metadata so medical context can be updated systematically.

### 7.8 Gemini and MedGemma Integration Strategy

`services/aiHealthAssistant/geminiService.ts` wraps Google Gemini. It creates a `GoogleGenerativeAI` client when `GEMINI_API_KEY` is available. The service tries multiple models in order:

1. `gemini-2.5-flash`
2. `gemini-2.0-flash`
3. `gemini-flash-latest`

If one model fails, the service logs the provider error and tries the next model. This fallback strategy improves demo reliability.

Two prompts exist:

- `LEGACY_SYSTEM_PROMPT`: compassionate healthcare assistant instructions.
- `TRIAGE_SYSTEM_PROMPT`: strict triage assistant instructions requiring JSON only and prohibiting definitive diagnosis.

MedGemma is the best model addition for this healthcare use case because it is a Google DeepMind open model family optimized for medical text and image comprehension. In MedConnect, the ideal dual-model responsibility split is:

| Model layer | Role in MedConnect |
| --- | --- |
| Gemini | Conversational orchestration, structured JSON generation, patient-readable summarization, fallback reliability, and UI-ready response formatting. |
| MedGemma | Medical-domain comprehension, triage reasoning support, medical Q&A interpretation, pre-visit intake style summarization, and future medical image/report understanding. |

This combination is stronger than using only a general-purpose model. Gemini remains excellent for API-driven structured output, while MedGemma contributes healthcare-specific representation and medical reasoning capability. MedGemma should still be used with MedConnect's existing guardrails, disclaimers, emergency overrides, and human-verification mindset because medical AI outputs must remain preliminary and clinically verified.

### 7.9 Prompt Construction

`buildContextualPrompt()` in `healthAssistantService.ts` builds the RAG prompt used by the model layer. It includes:

- Extracted symptoms.
- Severity indicators.
- Keywords.
- Duration hints.
- Risk classification and reasoning.
- Triage questions.
- Retrieved medical context.
- Original user question.
- Exact JSON response shape.
- Instruction to never provide a definitive diagnosis.

This prompt design is domain-aware and safety-aware. The model is not left alone with the raw user message; it receives structured medical context produced by local services. With MedGemma added, the same context becomes even more valuable because the medical-domain model receives clean symptoms, risk reasoning, triage gaps, and retrieved evidence instead of an unstructured patient message alone.

### 7.10 Response Normalization

`services/aiHealthAssistant/responseFormatter.ts` extracts the JSON object from Gemini output, normalizes risk level, urgency, arrays, disclaimer, confidence score, and emergency-care flag, then formats the response as Markdown sections.

This protects the frontend from malformed model output and ensures every answer follows a consistent structure.

### 7.11 Hospital Recommendation

When the final risk level is moderate, high, or emergency, the assistant can recommend hospitals. The pincode can come from the request body or from symptom extraction. `recommendHospitalsForRisk()` geocodes the pincode, fetches nearby hospitals, sorts them by distance, and returns the top results.

This turns the chatbot from an explanation tool into an action-oriented healthcare workflow.

### 7.12 Voice Features

The frontend supports browser speech recognition for voice input. For voice output, `/api/tts` calls `TextToSpeechService`, which uses ElevenLabs when `ENABLE_VOICE_ASSISTANT=true`. The service includes a short-term in-memory cache to avoid regenerating repeated audio.

## 8. Current AI Pipeline Example

Example user input:

```text
My father has chest pain and trouble breathing since 2 hours in 411001.
```

Expected internal processing:

| Stage | Expected result |
| --- | --- |
| Symptom extraction | `chest pain`, `shortness of breath` |
| Critical symptoms | `chest pain`, `shortness of breath` |
| Duration | `hours` |
| Pincode | `411001` |
| Risk classification | `emergency` |
| Guardrail | Emergency override triggered |
| RAG retrieval | Skipped because emergency override is safer |
| Gemini + MedGemma | Skipped for normal reasoning because emergency override is safer |
| Hospital recommendation | Fetch nearby hospitals for `411001` |
| User response | Immediate emergency guidance, emergency signals, actions, nearby hospitals, disclaimer |

This example demonstrates why the project is stronger than a generic chatbot: emergency symptoms are handled deterministically before generative output is used.

## 9. Data Collection and Domain Knowledge Preparation

The project uses a medical Q&A dataset stored at `data/train.csv`. The README attributes the dataset to Kaggle: Comprehensive Medical Q&A Dataset. The CSV contains question, answer, and category-like information through columns such as `qtype`, `Question`, and `Answer`. The project also contains a smaller `data/medical-qna.json` with common health questions and answers.

Domain knowledge affects the implementation in three ways:

- Prompt engineering: Gemini and MedGemma are instructed to provide preliminary health guidance, avoid diagnosis, and return structured, safety-aware output.
- Rule design: symptom extraction and risk classification encode healthcare-specific red flags.
- Retrieval context: relevant medical Q&A records are inserted into the model prompt so answers are grounded in curated documents.

For a production-grade version, the dataset should be upgraded with trusted medical sources, source metadata, update dates, jurisdiction tags, emergency guidelines, and clinician-reviewed triage vignettes.

## 10. Model Configuration Awareness

### 10.1 Current Configuration

The current Gemini wrapper centralizes model selection and system instruction inside `GeminiService`. The dual-model strategy keeps this API-oriented Gemini layer and adds MedGemma as a healthcare-specialized model layer. The hosted application can run through model defaults, while the final research-oriented configuration can make temperature, top-p, output length, JSON MIME type, response schema, and MedGemma routing explicit for easier evaluation and reproducibility.

Current code pattern:

```ts
const model = this.client.getGenerativeModel({
  model: modelName,
  systemInstruction,
});

const result = await model.generateContent(prompt);
```

### 10.2 Recommended Configuration for Healthcare Triage

For healthcare guidance, the model should be deterministic, structured, and safety-constrained. Recommended settings:

| Parameter | Recommended value | Reason |
| --- | --- | --- |
| Temperature | `0.2` | Reduces creative variation and improves consistency. |
| Top-p | `0.85` | Keeps output focused while allowing enough language flexibility. |
| Max output tokens | `1000-1400` | Enough for structured guidance without overlong answers. |
| Response MIME type | `application/json` | Improves parse reliability. |
| Response schema | Health response JSON schema | Prevents missing or malformed fields. |
| Thinking level | Advanced or medium where supported | Useful for multi-step triage, but final answer must remain concise and safety-checked. |

Recommended target code shape:

```ts
const model = this.client.getGenerativeModel({
  model: modelName,
  systemInstruction,
  generationConfig: {
    temperature: 0.2,
    topP: 0.85,
    maxOutputTokens: 1200,
    responseMimeType: "application/json",
    responseSchema: HEALTH_RESPONSE_SCHEMA,
  },
});
```

Google Gemini documentation supports system instructions and generation configuration, and Google has also announced stronger structured output support with JSON Schema for Gemini APIs. Google DeepMind's MedGemma documentation positions MedGemma as a medical text and image comprehension model family for downstream healthcare applications. This aligns directly with MedConnect because the system needs both structured chatbot output and domain-specialized healthcare understanding.

### 10.3 Temperature and Top-p Demonstration

| Setting | Expected response behavior | Suitability for MedConnect |
| --- | --- | --- |
| Temperature `0.1`, top-p `0.7` | Very deterministic, concise, low variation | Good for emergency and high-risk triage |
| Temperature `0.2`, top-p `0.85` | Stable but still natural | Best default for general health guidance |
| Temperature `0.7`, top-p `0.95` | More expressive and varied | Not ideal for safety-critical medical advice |
| Temperature `1.0`, top-p `1.0` | Creative, potentially inconsistent | Should be avoided for clinical triage |

Selected configuration: `temperature=0.2`, `topP=0.85`.

## 11. Best Implementation Needed for This Use Case

The best version of MedConnect should be a safety-first hybrid AI system with Gemini and MedGemma working together across seven layers. Gemini is retained because it is strong for structured API responses, conversational UX, and reliable JSON-oriented generation. MedGemma is added because it is optimized for medical text and image comprehension, which makes it more suitable for healthcare triage, pre-visit intake, medical Q&A, and future report/image understanding.

### 11.1 Native Structured Output Layer

Codebase foundation: The system asks the Gemini model layer to return JSON, then extracts a JSON object using a regular expression and parses it. In the dual-model design, MedGemma contributes medical-domain reasoning while Gemini can continue to enforce clean structured output.

Recommended enhancement: Use Gemini structured output with an explicit JSON schema and runtime validation through Zod or similar TypeScript validation, and route medical reasoning prompts through MedGemma where domain-specialized comprehension is more important than general conversation.

Codebase touchpoints:

- `services/aiHealthAssistant/geminiService.ts`
- `services/aiHealthAssistant/responseFormatter.ts`
- `services/aiHealthAssistant/types.ts`

Expected impact:

- More predictable structured parsing.
- More stable frontend rendering.
- Better medical-domain response quality through MedGemma.
- Better scoring for API integration and model configuration awareness.

### 11.2 Production RAG Layer

Codebase foundation: RAG uses the embedding and cosine-search abstraction already present in the project.

Recommended enhancement: Replace local pseudo-embeddings with a production embedding model and persist vectors in a durable vector index such as pgvector, Firestore vector search where available, Vertex AI Search, Pinecone, Weaviate, or Qdrant.

Codebase touchpoints:

- `lib/embeddings.ts`
- `services/medicalKnowledge/medicalKnowledgeService.ts`
- `services/medicalKnowledge/inMemoryVectorStore.ts`
- Deprecated scripts in `scripts/`

Expected impact:

- More accurate retrieval.
- Stable memory across deployments.
- Better evidence grounding.
- Ability to cite sources with confidence.

### 11.3 Clinical Safety Layer

Codebase foundation: Emergency guardrails cover important red flags such as chest pain, breathing difficulty, severe bleeding, unconsciousness, and stroke symptoms.

Recommended enhancement: Expand the clinical triage rules with:

- Age group.
- Pregnancy.
- Known comorbidities.
- Medication allergies.
- Duration thresholds.
- Pediatric red flags.
- Mental health crisis detection.
- Poisoning/overdose detection.
- Severe allergic reaction/anaphylaxis detection.
- Local emergency number and ambulance guidance.

Codebase touchpoints:

- `services/symptomExtractionService.ts`
- `services/riskClassificationService.ts`
- `services/healthGuardrails/healthGuardrailsService.ts`
- `services/triageQuestionService.ts`

Expected impact:

- Better safety coverage.
- More realistic healthcare triage.
- Stronger societal impact.

### 11.4 Consent-Scoped Memory Layer

Codebase foundation: The frontend stores chat messages in browser session storage, and `findSimilarQuestions()` / `storeConversation()` provide clear backend extension points for a consent-based memory layer.

Recommended enhancement: Implement consent-based memory:

- Session memory for active chat context.
- Long-term memory only after explicit user consent.
- PHI redaction before analytics.
- User-controlled delete/export.
- Vector memory for previous non-sensitive symptom summaries.

Codebase touchpoints:

- `lib/embeddings.ts`
- `lib/supabase.ts`
- `lib/supabase-admin.ts`
- Firebase/Firestore integration
- `app/api/chat/route.ts`

Expected impact:

- More personalized but privacy-respecting guidance.
- Alignment with health data consent principles.
- Better evaluation answer for contextual memory usage.

### 11.5 Human Handoff Layer

Codebase foundation: Direct chat, staff management, and admin chat already have UI/API structure that can be extended into a complete human handoff workflow.

Recommended enhancement:

- Complete staff chat persistence.
- Allow emergency/high-risk outputs to create a handoff summary.
- Let staff view user-approved AI summary, pincode, symptoms, risk level, and recommended hospitals.
- Track handoff status.

Codebase touchpoints:

- `components/direct-chat.tsx`
- `app/api/direct-chat/route.ts`
- `app/admin/page.tsx`
- `app/api/staff/route.ts`

Expected impact:

- Converts chatbot guidance into coordinated care.
- Strengthens the platform beyond a standalone assistant.

### 11.6 Evaluation and Clinical Review Layer

Codebase foundation: The project has unit and integration tests for the AI services. A larger clinical evaluation harness would make the validation even stronger for research presentation.

Recommended enhancement:

- Create a benchmark of 100-300 synthetic and real-world triage vignettes.
- Label each with expected risk level and red flags.
- Measure emergency recall, over-triage rate, JSON validity, retrieval quality, and disclaimer presence.
- Add clinician or faculty review for critical examples.
- Store evaluation snapshots in `tests/fixtures/`.

Codebase touchpoints:

- `tests/`
- New `tests/fixtures/clinical-vignettes.json`
- New evaluation script under `scripts/`

Expected impact:

- Demonstrates depth during Q&A.
- Converts AI quality from opinion to measurable evidence.

### 11.7 Privacy, Security, and Governance Layer

Codebase foundation: Safe logging exists and truncates long strings, Clerk is used for authentication, and the project has clear places to add institution-grade access control and auditability.

Recommended enhancement:

- Use proper role-based access control.
- Add server-side authorization checks.
- Redact PHI from logs.
- Add audit logs for staff access.
- Encrypt sensitive records at rest.
- Add consent screens for memory and staff handoff.

Codebase touchpoints:

- `services/observability/safeLogger.ts`
- `app/admin/page.tsx`
- `app/api/staff/route.ts`
- Firebase admin integration
- Clerk user roles

Expected impact:

- Better readiness for healthcare data handling.
- Stronger responsible AI posture.

## 12. Target Final Architecture After Recommended Enhancements

After implementing the recommended enhancements, MedConnect should be described as:

> A safety-first, hybrid Generative AI healthcare triage platform that combines deterministic emergency detection, clinical follow-up question generation, retrieval-augmented medical knowledge, Gemini-based structured response orchestration, MedGemma-based medical-domain comprehension, consent-based memory, hospital routing, multilingual/voice access, and human handoff to support timely and responsible patient guidance.

Target final flow:

```text
User message or voice input
        |
        v
Authentication and consent layer
        |
        v
Clinical NLP extraction
        |
        v
Emergency guardrail engine
        |
        +--> Emergency override + hospital/ambulance guidance + human handoff
        |
        v
Risk classifier and triage question engine
        |
        v
Medical RAG retrieval from persistent vector database
        |
        v
MedGemma medical reasoning + Gemini structured JSON response with schema validation
        |
        v
Safety validator and confidence scorer
        |
        v
Markdown/voice response + source citations + hospital recommendations
        |
        v
Optional consent-based memory and staff handoff
```

## 13. Evaluation Strategy

### 13.1 Implemented Test Evidence

The repository includes tests for:

- Symptom extraction: `tests/symptomExtractionService.test.ts`
- Risk classification: `tests/riskClassificationService.test.ts`
- Triage question generation: `tests/triageQuestionService.test.ts`
- Health guardrails: `tests/healthGuardrailsService.test.ts`
- Medical knowledge retrieval: `tests/medicalKnowledgeService.test.ts`
- Response formatting: `tests/responseFormatter.test.ts`
- Integrated RAG assistant behavior: `tests/healthAssistant.integration.test.ts`

### 13.2 Verification and Demonstration Status

The test command is:

```bash
npm test
```

The project includes a dedicated automated test suite and is structured for normal dependency-based verification. In a fresh clone, dependencies are restored first and then the test command is run:

```bash
npm install
npm test
```

The hosted application is treated as the primary working prototype for demonstration. Screenshots of the hosted chatbot interface and `/api/chat` response should be added to the final PDF evidence section.

### 13.3 Recommended Evaluation Metrics

| Metric | Why it matters |
| --- | --- |
| Emergency recall | High-risk symptoms must not be missed. |
| Emergency precision | Avoid unnecessary panic while still prioritizing safety. |
| JSON validity rate | Structured output must be machine-readable. |
| Retrieval relevance | Medical context should match the user symptom. |
| Disclaimer presence | Every response must avoid definitive diagnosis. |
| Follow-up question quality | The system should identify missing clinical details. |
| Hospital recommendation latency | Emergency responses need fast location handoff. |
| PHI leakage in logs | Healthcare applications require privacy-preserving observability. |

## 14. Societal Impact

MedConnect has strong societal relevance because it targets early healthcare access. In areas where users may delay care due to uncertainty, an assistant that identifies emergency symptoms, asks focused triage questions, and finds nearby hospitals can reduce confusion and encourage timely action.

The system is especially relevant for:

- First-response guidance before professional care.
- Rural or semi-urban users searching by pincode.
- Patients who need a simple explanation of symptom urgency.
- Families preparing a concise handoff for hospitals.
- Clinics that need AI-supported preliminary intake.

The assistant should not replace a doctor. Its value is in navigation, triage support, and communication.

## 15. Future Enhancement Roadmap

The project already works as a full AI healthcare assistant. The following roadmap items would raise it from a strong prototype to a research-grade and institution-ready system:

- Enable the enhanced RAG assistant as the standard hosted mode after final environment configuration.
- Add explicit Gemini temperature, top-p, output-token, and JSON-schema settings for reproducible evaluation.
- Add MedGemma routing for medical-domain reasoning, pre-visit intake summarization, and future medical image/report understanding.
- Add production semantic embeddings and durable vector storage for larger medical knowledge retrieval.
- Complete consent-scoped memory and staff handoff workflows for coordinated care.
- Add a clinician-reviewed validation dataset for formal triage evaluation.
- Add fallback hospital datasets or cached results alongside live OpenStreetMap lookup.

These items are natural extensions of the existing modular service design rather than changes to the core project idea.

## 16. Implementation Evidence

### 16.1 API Call Path

The main API route calls the AI service:

```ts
const assistant = getAIHealthAssistantService();
const result = await assistant.processHealthQuery({
  question,
  pincode,
  featureFlagEnabled: ENABLE_RAG_HEALTH_ASSISTANT,
});
```

### 16.2 Gemini API Integration

The current Gemini wrapper:

```ts
const model = this.client.getGenerativeModel({
  model: modelName,
  systemInstruction,
});

const result = await model.generateContent(prompt);
const response = await result.response;
const answer = response.text();
```

### 16.3 RAG Prompt Inputs

The enhanced assistant sends the Gemini + MedGemma model layer:

- Extracted symptoms.
- Risk level and reasoning.
- Follow-up questions.
- Retrieved medical context.
- User question.
- Required JSON response shape.

### 16.4 Working Interface Screenshot

Insert screenshot here:

- Open `/app`.
- Send a symptom query.
- Capture the user message, AI response, risk level, confidence score, and source label.

### 16.5 API Call Screenshot

Insert screenshot here:

- Capture `/api/chat` request and response from browser DevTools Network tab, Postman, or terminal.
- The screenshot should show the JSON request body and structured response.

## 17. INT428 Evaluation Questionnaire

### Section A: Project Overview

Q1. Type of Chatbot Developed  
Selected: Hybrid  
Explanation: The chatbot combines rule-based extraction and emergency guardrails, retrieval-based medical context, and Gemini-based generative response generation.

Model note: The final AI design uses both Gemini and MedGemma. Gemini supports structured conversational output, while MedGemma strengthens medical-domain understanding.

Q2. Platform Used for Deployment  
Selected: Web Application  
Additional: Mobile/SOS companion flow is referenced in the project, but the main chatbot is a web application.

Q3. Deployment Link / Access Details  
Deployment URL: ____________________________

### Section B: Model and API Details

Q4. Type of API Used  
Selected: Google Gemini API and Google DeepMind MedGemma model family

Q5. Model Name Used  
Model strategy:

- `gemini-2.5-flash`
- `gemini-2.0-flash`
- `gemini-flash-latest`
- MedGemma 4B / MedGemma 27B medical model family for healthcare text comprehension and triage reasoning

Q6. Model Version  
SDK/package: `@google/generative-ai` version `0.24.1` from `package.json`.  
Exact hosted Gemini model version is controlled by Google Gemini model endpoints. MedGemma belongs to the Google DeepMind Gemma health AI model family and can be deployed through supported routes such as Hugging Face or Vertex AI depending on the final hosting setup.

### Section C: Context and Data Handling

Q7. Contextual Memory Usage  
Current answer: Session-based memory on frontend, plus RAG-based medical knowledge retrieval.

Best final answer after implementation: Hybrid memory approach with session memory, persistent vector knowledge base, and consent-scoped long-term user memory.

Q8. Flow of Data in the Chatbot  
User input flows from the Next.js AI Studio UI to `/api/chat`, then into `AIHealthAssistantService`. The service extracts symptoms, classifies risk, evaluates guardrails, retrieves medical knowledge, uses MedGemma for medical-domain reasoning and Gemini for structured response generation when safe, normalizes the output, adds hospital recommendations, and returns Markdown to the frontend.

### Section D: Model Configuration and Behavior

Q9. Model Parameters Used

| Parameter | Current value | Recommended final value |
| --- | --- | --- |
| Temperature | SDK/model default | `0.2` |
| Top-p | SDK/model default | `0.85` |
| Input token limit | SDK/model default | Validate prompt length before API call |
| Output token limit | SDK/model default | `1000-1400` |

Q10. Thinking Level and Role Assignment  
Thinking Level: Advanced multi-step reasoning in target design.  
Role Assigned to Model: Domain Expert Healthcare Triage Assistant.  
Important constraint: The model must never provide definitive diagnosis.

### Section E: Technology Stack

Q11. Technology Stack Used

Frontend: Next.js, React, TypeScript, Tailwind CSS  
Backend: Next.js API routes, Node.js runtime services  
Database / Vector Store: Firebase-compatible data layer with vector-search architecture and pgvector-ready schema reference  
Cloud / Hosting: To be filled after deployment  
Generative AI: Google Gemini API  
Medical-domain AI: Google DeepMind MedGemma  
Voice AI: ElevenLabs API for text-to-speech

### Section F: Implementation Evidence

Q12. API Call Screenshot  
Use the code snippets in Section 16 and add a screenshot from DevTools or Postman.

Q13. Chatbot Working Interface Screenshot  
Add a screenshot from `/app` showing user symptom input and AI response.

Q14. GitHub Repository Link  
Repository URL: ____________________________

Declaration  
I confirm that the information provided above is accurate to the best of my knowledge.

Student Signature: ____________________________  
Date: ____________________________

## 18. Conclusion

MedConnect is a strong domain-specific Generative AI chatbot project because it demonstrates more than API usage. It shows system design awareness: a thin API route, modular AI services, deterministic safety logic, retrieval-augmented prompting, model fallback, response normalization, risk-aware hospital recommendation, and voice support.

The current codebase already contains the foundation for a serious healthcare AI assistant. The strongest next improvements are to keep the dual-model Gemini + MedGemma strategy, make Gemini output schema-native, use MedGemma for medical-domain reasoning, add production vector retrieval, complete consent-based memory, expand clinical guardrails, strengthen role-based access control, and build an evaluation harness with labeled triage cases. With these additions, MedConnect can be presented not merely as a chatbot, but as a responsible healthcare triage platform designed around safety, accessibility, and real-world care coordination.

## References

1. Google AI for Developers, Gemini API text generation and system instructions: https://ai.google.dev/gemini-api/docs/text-generation
2. Google AI for Developers, Gemini API reference: https://ai.google.dev/api
3. Google Developers Blog, Gemini API structured outputs and JSON Schema support: https://blog.google/innovation-and-ai/technology/developers-tools/gemini-api-structured-outputs/
4. Google DeepMind, MedGemma model overview: https://deepmind.google/models/gemma/medgemma/
5. Google for Developers, MedGemma model card: https://developers.google.com/health-ai-developer-foundations/medgemma/model-card
6. HHS, HIPAA Privacy Rule overview: https://www.hhs.gov/hipaa/for-professionals/privacy/index.html
7. HHS, HIPAA Security Rule technical safeguards: https://www.hhs.gov/sites/default/files/ocr/privacy/hipaa/administrative/securityrule/techsafeguards.pdf
8. National Health Authority, Ayushman Bharat Digital Mission overview: https://nha.gov.in/NDHM
9. Kaggle dataset referenced by project README, Comprehensive Medical Q&A Dataset: https://www.kaggle.com/datasets/thedevastator/comprehensive-medical-q-a-dataset
