# Multi-Agent RAG System for Legal Contract Analysis

A console-based interactive system that analyzes legal contracts using a multi-agent Retrieval-Augmented Generation (RAG) architecture. Users ask natural language questions and receive grounded, cited answers with risk indicators.

---

## Architecture Overview

![Architecture Diagram](assets/architecture_diagram.png)

### System Flow (Text Version)

```
                              User Query
                                  │
                    ┌─────────────┴──────────────┐
                    │       ORCHESTRATOR          │
                    │                             │
                    │  1. Rewrite Guard           │
                    │     ├─ Numeric? → Direct map│
                    │     ├─ Short/referential?   │
                    │     │  → LLM Rewriter       │
                    │     └─ Self-contained?      │
                    │        → Pass through       │
                    │                             │
                    │  2. Query Classifier        │
                    │     → analysis / risk /     │
                    │       out_of_scope          │
                    └─────────┬───────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        ┌──────────┐   ┌──────────┐   ┌─────────────┐
        │ ANALYSIS │   │   RISK   │   │ OUT OF SCOPE│
        │  AGENT   │   │  AGENT   │   │  (hardcoded │
        │ temp=0.0 │   │ temp=0.2 │   │   refusal)  │
        └────┬─────┘   └────┬─────┘   └─────────────┘
             │              │
             └──────┬───────┘
                    │
                    ▼
            ┌──────────────┐
            │  RETRIEVER   │
            │              │
            │ Stage 1:     │
            │ Bi-encoder   │
            │ Single doc?  │
            │ → Metadata   │
            │   filter     │
            │ Multi/none?  │
            │ → Semantic   │
            │   search     │
            │   (top-10)   │
            │              │
            │ Stage 2:     │
            │ Cross-encoder│
            │ re-ranking   │
            │   (top-5)    │
            └──────┬───────┘
                   │
                   ▼
            ┌──────────────┐
            │  ChromaDB    │
            │  (21 chunks) │
            │  Persisted   │
            └──────────────┘
```

### Ingestion Pipeline

```
data/*.txt  ──►  Clause-Based  ──►  Ollama Embeddings  ──►  ChromaDB
(4 docs)         Chunker             (nomic-embed-text)      (21 chunks)
                 Split by §           768 dimensions          File-persisted
                 numbers              Local, free             vector_store/
```

---

## Agents & Responsibilities

| Agent | Role | Temperature | Prompt Strategy |
|-------|------|-------------|-----------------|
| **Orchestrator** | Query rewriting, classification, routing, follow-up generation | 0.0 | Deterministic routing; code-level guards for rewrite decisions |
| **Analysis Agent** | Factual Q&A with citations | 0.0 | Grounded in CONTEXT (primary), history (secondary); must cite sources |
| **Risk Agent** | Risk identification with severity levels | 0.2 | Categorizes HIGH/MEDIUM/LOW; flags cross-document conflicts |

The orchestrator is NOT a separate LLM agent -- it reuses the same model instance for lightweight tasks (classification, rewriting, follow-ups). This avoids loading multiple model instances.

---

## Design Choices & Justifications

### 1. Clause-Based Chunking

**Choice:** Split by numbered sections (1., 2., 3...) instead of fixed-size token windows.

**Why:** Legal contracts have natural semantic boundaries. Each section covers one concept. Fixed-size chunking splits clauses mid-sentence, causing the retriever to return fragments.

**Result:** 21 chunks across 4 documents. Each chunk is a complete, self-contained clause with metadata (document name, section number, title).

### 2. Embedding: nomic-embed-text via Ollama

**Choice:** 768-dim local embeddings via Ollama instead of sentence-transformers + PyTorch.

**Why:** sentence-transformers pulls PyTorch (~2GB). nomic-embed-text runs through Ollama (already required for the LLM), keeping the install lightweight (~274MB).

### 3. Retrieval: Two-Stage (Bi-Encoder + Cross-Encoder Re-Ranking)

**Choice:** Two-stage retriever with metadata filtering:
- **Stage 1 (bi-encoder):** ChromaDB cosine similarity retrieves top-10 candidates
  - **Single document mentioned** (e.g., "summarize the SLA") → metadata filter to that document
  - **Multiple or no documents** → semantic search across all
- **Stage 2 (cross-encoder):** `ms-marco-MiniLM-L-6-v2` re-ranks candidates, returns top-5

**Why:** The bi-encoder embeds query and chunks independently -- fast but approximate. The cross-encoder reads query + chunk jointly through cross-attention, catching nuances like paraphrase matching ("capped" vs. "shall not exceed") and negation handling. Casting a wider initial net (top-10) then re-ranking down (top-5) improves retrieval precision.

**Document detection** uses a keyword alias map (e.g., "sla" → "SLA", "non-disclosure" → "NDA"). Simple but sufficient for 4 known documents.

**Trade-off:** Adds `sentence-transformers` (pulls PyTorch ~2GB). Toggleable via `RERANKER_ENABLED=false` environment variable.

### 4. Multi-Turn Conversation: Hybrid Strategy

Three-level approach to handle follow-up questions:

| Component | Purpose | Example |
|-----------|---------|---------|
| **Rewrite Guard** (code) | Prevents unnecessary LLM rewrites | "can I engage subprocessors?" → skip rewriter |
| **Query Rewriter** (LLM) | Resolves vague references | "what about that?" → "What is the liability in the NDA?" |
| **Sliding Window** | Passes last N turns to agents | Agents can reference prior answers |

**Rewrite triggers:** Referential language ("that", "it", "tell me more") OR very short queries (<4 words, e.g., "yes", "1").

**Numeric selection:** When user types "1", "2", "3", directly indexes into stored follow-up suggestions (code-level, not LLM). More reliable than asking the 8B model to parse numbered selections.

### 5. Follow-Up Suggestions: Grounded + Mixed Types

After each answer, the system suggests 3 follow-up questions:
1. **Deepen** -- go deeper into the current topic/document
2. **Broaden** -- explore a different document or topic area
3. **Risk** -- risk perspective on what was just discussed

**Grounded** by passing the full document inventory (all documents + section titles) to the generator. Prevents hallucinated questions about information that doesn't exist in the corpus.

### 6. Prompt Design: Priority Hierarchy

Both agent prompts use an explicit information priority:
- **PRIMARY SOURCE:** Retrieved CONTEXT sections (actual document excerpts)
- **SECONDARY:** Conversation history (for reference only, NOT a factual source)

**Prompt ordering:** HISTORY → CONTEXT → QUESTION. Context is placed nearest to the question to mitigate the "lost in the middle" effect where LLMs attend less to information in the middle of the prompt.

### 7. Out-of-Scope Handling

The classifier detects out-of-scope queries (drafting requests, legal advice, unrelated questions). Response is a hardcoded polite refusal listing capabilities, not an LLM-generated response. This ensures consistency and avoids accidental legal advice.

---

## Project Structure

```
ril_eval/
├── data/                          # Legal documents (4 contracts)
│   ├── nda_acme_vendor.txt
│   ├── vendor_services_agreement.txt
│   ├── service_level_agreement.txt
│   └── data_processing_agreement.txt
├── src/
│   ├── config.py                  # All configuration in one place
│   ├── chunker.py                 # Clause-based document parser
│   ├── ingestion.py               # Embedding + ChromaDB storage
│   ├── retriever.py               # Semantic search + metadata filtering
│   ├── agents/
│   │   ├── orchestrator.py        # Query routing + conversation management
│   │   ├── analysis.py            # Factual Q&A agent
│   │   └── risk.py                # Risk assessment agent
│   └── prompts/
│       └── templates.py           # All prompt templates (centralized)
├── evaluation/
│   ├── test_cases.py              # Ground-truth dataset (16 queries)
│   └── evaluate.py                # 4-dimension evaluation runner
├── main.py                        # Interactive CLI entry point
├── requirements.txt               # Python dependencies
├── DESIGN_DECISIONS.md            # Detailed trade-off analysis
└── README.md                      # This file
```

---

## Setup & Installation

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai) installed and running

### Step 1: Install Ollama Models

```bash
ollama pull llama3.1:8b        # LLM for analysis (~4.7GB)
ollama pull nomic-embed-text   # Embedding model (~274MB)
ollama serve                   # Start Ollama server (keep running)
```

### Step 2: Python Environment

```bash
cd ril_eval
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 3: Run

```bash
# First run ingests documents into ChromaDB (takes ~30 seconds)
python main.py

# Force re-ingestion if documents change
python main.py --reingest
```

---

## Usage

```
You: What is the uptime commitment in the SLA?

┌─── Factual Analysis ──────────────────────────────────────┐
│ According to [Source: SLA, Section 1 - Service             │
│ Availability], the Vendor's uptime commitment is           │
│ 99.5% monthly uptime.                                     │
└───────────────────────────────────────────────────────────┘
┌─── Retrieved Sources ─────────────────────────────────────┐
│ • SLA → Service Availability (relevance: 0.52)             │
│ • SLA → Service Credits (relevance: 0.40)                  │
└───────────────────────────────────────────────────────────┘
┌─── You might also ask ────────────────────────────────────┐
│ 1. What are the service credits if uptime falls below?     │
│ 2. Is liability capped in the Vendor Services Agreement?   │
│ 3. Are there risks related to SLA exclusions?              │
└───────────────────────────────────────────────────────────┘

You: 2    ← selects suggestion #2 (broadening to a different document)
```

### Commands

| Command | Action |
|---------|--------|
| Type a question | Analyze contracts |
| `1`, `2`, `3` | Select a follow-up suggestion |
| `clear` | Reset conversation history |
| `quit` / `exit` | End session |

---

## Sample Queries Tested

| Query | Type | Tests |
|-------|------|-------|
| "What is the notice period for terminating the NDA?" | Analysis | Single-doc factual extraction |
| "What is the uptime commitment in the SLA?" | Analysis | Metadata-filtered retrieval |
| "Are there conflicting governing laws across agreements?" | Risk | Cross-document comparison |
| "Is there any unlimited liability in these agreements?" | Risk | Multi-doc risk identification |
| "Can you draft a better NDA for me?" | Out-of-scope | Boundary enforcement |
| "What legal strategy should Acme take?" | Out-of-scope | Legal advice refusal |
| "Summarize the SLA" | Analysis | Broad document-scoped query |
| "What about the vendor agreement?" (follow-up) | Analysis | Query rewriting from history |

---

## Evaluation

### How to Run

```bash
python -m evaluation.evaluate              # Run all 16 test cases
python -m evaluation.evaluate --verbose    # Show full answers
python -m evaluation.evaluate --csv        # Save results to evaluation/results.csv
```

### What We Evaluate (4 Dimensions)

| Dimension | What It Checks | How | Failure Example |
|-----------|---------------|-----|-----------------|
| **Classification Accuracy** | Query routed to correct agent? | Compare classifier output vs. expected category | "Is liability capped?" classified as analysis instead of risk |
| **Retrieval Recall** | Right documents/sections retrieved? | Check expected docs/sections appear in results | "Uptime in the SLA?" retrieves NDA instead of SLA |
| **Citation Faithfulness** | Citations grounded in retrieved context? | Parse `[Source: ...]` from answer, verify against retrieved chunks | Answer cites "NDA Section 7" but Section 7 doesn't exist |
| **Answer Correctness** | Answer contains expected key facts? | Case-insensitive phrase matching | "Uptime commitment?" answer missing "99.5%" |

### Why These Metrics Matter

- **Classification** errors cascade: wrong routing → wrong agent → wrong answer type
- **Retrieval** failures are unfixable downstream: if the right chunk isn't retrieved, the agent can't find the answer
- **Faithfulness** catches citation hallucination: the LLM inventing sources that weren't in its context
- **Correctness** verifies the end-to-end pipeline delivers the right facts

### Limitations of This Approach

1. **Keyword matching is brittle**: "30 days" matches but "a month" doesn't. Paraphrased answers may score lower than they deserve.
2. **Cannot detect subtle hallucinations**: If the LLM adds "24/7 support" while citing a real source, we miss it. A RAGAS-style LLM judge (GPT-4) would catch this by decomposing the answer into claims.
3. **Manual ground truth**: 16 test cases cover the assignment queries but don't cover novel user phrasings. In production, we'd add RAGAS for scalable evaluation without manual effort.
4. **No nuance evaluation**: Can't assess answer completeness, explanation quality, or tone -- only factual presence/absence.

---

## Known Limitations

1. **8B Model Constraints:** llama3.1:8b has limited instruction-following ability. The query rewriter occasionally misinterprets intent, mitigated by code-level guards.

2. **Small Corpus:** 4 documents, 21 chunks. The architecture is designed for scale but not stress-tested with hundreds of documents.

3. **Keyword-Based Document Detection:** The alias map covers known document names. Unusual references ("the confidentiality agreement") may not be caught.

4. **Context Window:** The 8B model's ~8K token context limits how much history + context can be passed simultaneously.

5. **No Hallucination Detection:** Answers are grounded via prompting, but there's no automated check that the answer actually matches the source text.

6. **Re-ranker adds PyTorch dependency:** The cross-encoder re-ranker pulls `sentence-transformers` + PyTorch (~2GB). Disable with `RERANKER_ENABLED=false` for a lighter install.

---

## Production Enhancements

If this were a production system, we would add:

| Enhancement | Impact | Effort |
|-------------|--------|--------|
| **Hybrid Retrieval (BM25 + Semantic)** | Handles keyword-heavy and meaning-heavy queries | Medium |
| **GPT-4o / GPT-4o-mini** | Much better instruction following, fewer code guards needed | Low |
| **Explored-Topics Tracking** | Smarter follow-ups biased toward unexplored areas | Medium |
| **Memory Summarization** | Handles 50+ turn conversations without context overflow | Medium |
| **Hallucination Detection Agent** | Validates answers against source text | High |
| **Hierarchical Chunking** | Clause-level + document-summary chunks for broad queries | Medium |
| **User Feedback Loop** | Thumbs up/down on answers to improve prompts over time | Medium |
| **Async Pipeline** | Parallel retrieval + classification for lower latency | Low |

---

## Configuration

All parameters are in `src/config.py` and can be overridden via environment variables:

```bash
# Use a different Ollama model
OLLAMA_MODEL=mistral python main.py

# Point to a remote Ollama server
OLLAMA_BASE_URL=http://192.168.1.100:11434 python main.py

# Switch to OpenAI (future support)
LLM_PROVIDER=openai OPENAI_API_KEY=sk-... python main.py
```
