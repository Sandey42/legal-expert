# Multi-Agent RAG System for Legal Contract Analysis

## Problem Overview

A console-based multi-agent RAG system that lets users query and analyze legal contracts using natural language. The system ingests 4 legal documents (NDA, Vendor Services Agreement, SLA, Data Processing Agreement), retrieves relevant clauses, and routes queries to specialized agents -- an Analysis Agent for factual Q&A with citations, and a Risk Agent for identifying legal/financial risks with severity levels. It supports multi-turn conversations with context-aware follow-up handling and out-of-scope query detection.

## Architecture

![Architecture Diagram](assets/architecture_diagram_fresh.png)

### System Flow

```
User Query
    │
    ▼
┌──────────────────────────────────────────────────┐
│          AGENT 1: ORCHESTRATOR (LLM, temp=0.0)   │
│                                                   │
│  1. Social phrase detection (code-level)          │
│  2. Rewrite guard / Query rewriter (LLM)         │
│  3. Query classifier → analysis | risk | oos     │
│  4. Follow-up suggestion generator (LLM)         │
└──────────┬───────────────┬──────────┬─────────────┘
           │               │          │
      analysis           risk     out_of_scope
           │               │          │
           ▼               ▼          ▼
    ┌─────────────────────────┐   Hardcoded
    │       RETRIEVER         │   refusal
    │                         │
    │  Stage 1: Bi-encoder    │
    │    (ChromaDB) → top 10  │
    │  Stage 2: Cross-encoder │
    │    (re-rank)  → top 5   │
    │  Stage 3: Threshold     │
    │    filter (min 3 kept)  │
    └────────┬────────────────┘
             │
     ┌───────┴────────┐
     ▼                ▼
┌───────────┐  ┌───────────┐
│  AGENT 2: │  │  AGENT 3: │
│ ANALYSIS  │  │   RISK    │
│ (LLM,     │  │ (LLM,     │
│  temp=0.0)│  │  temp=0.2)│
│           │  │           │
│ Factual   │  │ Identifies│
│ Q&A with  │  │ risks with│
│ citations │  │ H/M/L     │
└─────┬─────┘  └─────┬─────┘
      └───────┬───────┘
              ▼
     Response + Sources
     + Follow-up Suggestions
```

### Ingestion Pipeline

```
data/*.txt → Clause-based chunker → nomic-embed-text (768d) → ChromaDB (21 chunks)
```

### Component Summary

| Component | Role | Key Detail |
|-----------|------|------------|
| **Agent 1: Orchestrator** | Routes queries through gates, classification, and follow-ups | 3 LCEL chains (rewriter, classifier, follow-up) sharing 1 LLM |
| **Agent 2: Analysis** | Factual Q&A grounded strictly in retrieved context | Returns answers with source citations, temp=0.0 |
| **Agent 3: Risk** | Identifies legal/financial risks with severity ratings | Returns H/M/L risk levels with explanations, temp=0.2 |
| **Retriever** | Two-stage retrieval: bi-encoder search + cross-encoder re-ranking | Metadata filtering for single-document queries |
| **Chunker** | Splits legal documents by numbered clause sections | Produces 21 chunks with document-type and section metadata |
| **ChromaDB** | File-persisted vector store | In-process, no separate server required |

## Setup Instructions

**Prerequisites**

- Python 3.10+
- [Ollama](https://ollama.com) installed and running

**Start Ollama and pull models**

```bash
ollama serve                # start the Ollama server (if not already running)
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

**Install dependencies**

```bash
pip install -r requirements.txt
```

> Note: `sentence-transformers` pulls PyTorch as a transitive dependency (~2 GB). This is the trade-off for cross-encoder re-ranking quality.

## How to Run

```bash
python main.py              # Start interactive console
python main.py --reingest   # Force re-ingestion of documents
```

On first run, the system automatically ingests the 4 legal documents from `data/` into ChromaDB. Subsequent runs reuse the persisted vector store.

## Design Choices and Trade-offs

**1. Chunking Strategy**
- **Chose:** Clause-based splitting by numbered legal sections
- **Alternatives:** Fixed-size (512 tokens), recursive character splitting, sentence-level
- **Why:** Legal clauses are self-contained semantic units. Fixed-size chunks risk splitting a clause mid-sentence, losing meaning. Trade-off: uneven chunk sizes (some clauses are very short).

**2. Embedding Model**
- **Chose:** `nomic-embed-text` (768d, runs locally via Ollama)
- **Alternatives:** OpenAI `text-embedding-3-small`, `BGE-large`, `E5-mistral`
- **Why:** Runs fully local (no API key, no data leakage for legal docs), good quality for its size. Trade-off: lower benchmark scores than cloud embeddings.

**3. LLM Selection**
- **Chose:** `llama3.1:8b` via Ollama
- **Alternatives:** GPT-4o, Claude, Llama 70B, Mistral
- **Why:** Best quality-to-resource ratio for local inference. Runs on a single machine without GPU. Trade-off: less capable than 70B+ models on complex reasoning.

**4. Retrieval Strategy**
- **Chose:** Two-stage: bi-encoder (ChromaDB) → cross-encoder re-ranking
- **Alternatives:** Single-stage semantic search, BM25 + semantic hybrid, ColBERT
- **Why:** Bi-encoder is fast for candidate generation; cross-encoder captures token-level interactions for precise relevance. Trade-off: ~200ms extra latency, ~2GB PyTorch dependency.

**5. Vector Store**
- **Chose:** ChromaDB (file-persisted, in-process)
- **Alternatives:** Pinecone, Weaviate, Qdrant, FAISS
- **Why:** Zero infrastructure -- no server process, no cloud account. Persists to disk across runs. Trade-off: not suitable for concurrent users or million-scale corpora.

**6. Multi-Agent vs. Monolithic**
- **Chose:** 3 specialized agents (Orchestrator, Analysis, Risk)
- **Alternatives:** Single LLM with one mega-prompt, 2-agent (no orchestrator), tool-calling agent
- **Why:** Separation of concerns -- each agent has a focused prompt and temperature. Easier to test, debug, and extend individually. Trade-off: ~4 LLM calls per query instead of 1.

**7. LLM Instance Strategy**
- **Chose:** Separate `ChatOllama` instance per agent, injected via constructor (dependency injection)
- **Alternatives:** Single shared instance across all agents, hard-coded LLM inside each agent
- **Why:** Each agent owns its configuration (temperature, model). The Orchestrator reuses one LLM instance across its 3 internal LCEL chains (rewriter, classifier, follow-up generator) since they share the same temperature -- avoiding redundant connections while keeping agents independently configurable. Trade-off: slightly more wiring code, but makes agents testable and swappable.

**8. Multi-Turn Handling**
- **Chose:** Hybrid: code-level gates + LLM query rewriter
- **Alternatives:** Pure LLM rewriting for everything, stateless (no history)
- **Why:** Code gates handle trivial cases (greetings, numeric selection) without LLM cost. LLM rewriter handles complex referential queries. Trade-off: regex patterns need manual upkeep.

**9. Conversation History Retention**
- **Chose:** Sliding window of last 10 turns, stored in-memory, with CONTEXT prioritized over HISTORY in prompts
- **Alternatives:** Full history (unbounded), summarization-based compression, external memory store (Redis/DB)
- **Why:** Sliding window keeps prompt size bounded and predictable. Prioritizing retrieved context over history reduces hallucination from stale turns. Trade-off: earlier turns are silently dropped, which can lose important context in very long sessions.

**10. Follow-up Suggestion Strategy**
- **Chose:** LLM-generated suggestions (deepen, broaden, risk) with previous-suggestion deduplication and numeric shortcut selection
- **Alternatives:** Static predefined follow-ups, no follow-ups (let user type freely), retrieval-based suggestions
- **Why:** Dynamic suggestions guide the user toward high-value next queries while covering different exploration angles. Passing previous suggestions to the generator avoids repetition. Numeric selection ("1", "2", "3") reduces typing friction. Trade-off: adds one extra LLM call per turn.

**11. Temperature Strategy**
- **Chose:** 0.0 for Orchestrator/Analysis, 0.2 for Risk
- **Alternatives:** Uniform temperature, higher creativity for all
- **Why:** Factual Q&A needs deterministic output. Risk identification benefits from slight interpretive flexibility to surface non-obvious risks. Trade-off: risk agent occasionally over-interprets.

## Known Limitations

1. **8B model ceiling** -- `llama3.1:8b` occasionally struggles with complex multi-hop reasoning and can hallucinate when context is ambiguous. A 70B+ model or cloud API would improve quality significantly.

2. **Small corpus only** -- tested with 4 documents (21 chunks). Chunking strategy and retrieval thresholds are tuned for this scale; would need re-evaluation for 100+ documents.

3. **Single-user, in-process** -- conversation history is in-memory (lost on restart), ChromaDB runs in-process. Not suitable for concurrent users without adding a proper database and session management.

4. **Clause-based chunking assumes structure** -- the chunker expects numbered sections. Unstructured or differently formatted legal documents would need a more adaptive splitting strategy.

5. **Evaluation is deterministic only** -- ground-truth evaluation catches regressions but can't assess nuanced answer quality. LLM-as-judge (e.g., GPT-4) would be needed for production-grade evaluation.

6. **No streaming** -- responses arrive all at once. Token-by-token streaming would improve perceived latency for longer answers.
