# Multi-Agent RAG System for Legal Contract Analysis

## Problem Overview

A console-based multi-agent RAG system that lets users query and analyze legal contracts using natural language. The system ingests 4 legal documents (NDA, Vendor Services Agreement, SLA, Data Processing Agreement), retrieves relevant clauses, and routes queries to specialized agents -- an Analysis Agent for factual Q&A with citations, and a Risk Agent for identifying legal/financial risks with severity levels. It supports multi-turn conversations with context-aware follow-up handling and out-of-scope query detection.

## Architecture

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
