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
            │ Single doc?  │
            │ → Metadata   │
            │   filter     │
            │              │
            │ Multi/none?  │
            │ → Semantic   │
            │   search     │
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

### 3. Retrieval: Semantic Search + Metadata Filtering

**Choice:** Two-strategy retriever:
- **Single document mentioned** (e.g., "summarize the SLA") → filter ChromaDB to that document
- **Multiple or no documents** → standard semantic search across all (top-5)

**Why:** Pure semantic search fails on broad document-scoped queries ("summarize the SLA") because "summarize" doesn't match any specific section. Metadata filtering ensures complete document coverage.

**Document detection** uses a keyword alias map (e.g., "sla" → "SLA", "non-disclosure" → "NDA"). Simple but sufficient for 4 known documents.

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
│   └── evaluate.py                # RAG evaluation framework
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

## Known Limitations

1. **8B Model Constraints:** llama3.1:8b has limited instruction-following ability. The query rewriter occasionally misinterprets intent, mitigated by code-level guards.

2. **Small Corpus:** 4 documents, 21 chunks. The architecture is designed for scale but not stress-tested with hundreds of documents.

3. **No Re-ranking:** Retrieved chunks are ranked by raw embedding similarity. A cross-encoder re-ranker would improve result ordering.

4. **Keyword-Based Document Detection:** The alias map covers known document names. Unusual references ("the confidentiality agreement") may not be caught.

5. **Context Window:** The 8B model's ~8K token context limits how much history + context can be passed simultaneously.

6. **No Hallucination Detection:** Answers are grounded via prompting, but there's no automated check that the answer actually matches the source text.

---

## Production Enhancements

If this were a production system, we would add:

| Enhancement | Impact | Effort |
|-------------|--------|--------|
| **Hybrid Retrieval (BM25 + Semantic)** | Handles keyword-heavy and meaning-heavy queries | Medium |
| **Cross-Encoder Re-ranking** | Better result ordering after initial retrieval | Low |
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
