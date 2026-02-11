"""
Centralized configuration for the Multi-Agent RAG system.

All tunable parameters live here so they're easy to review and adjust.
This avoids magic numbers scattered across the codebase.
"""

import os
from pathlib import Path

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_PERSIST_DIR = PROJECT_ROOT / "vector_store"

# ──────────────────────────────────────────────
# LLM Configuration
# ──────────────────────────────────────────────
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # "ollama" or "openai"

# Ollama settings
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# OpenAI settings (for future use)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ──────────────────────────────────────────────
# Embedding Configuration
# ──────────────────────────────────────────────
# Using Ollama for embeddings too -- avoids heavy torch/sentence-transformers dependency.
# nomic-embed-text: 768 dims, strong performance, only ~274MB via Ollama.
EMBEDDING_MODEL = "nomic-embed-text"

# ──────────────────────────────────────────────
# Retrieval Configuration
# ──────────────────────────────────────────────
RETRIEVAL_TOP_K = 10         # Initial candidates from bi-encoder (wider net for re-ranking)
SIMILARITY_THRESHOLD = 0.3   # Minimum similarity score to include a chunk
RETRIEVAL_MIN_RESULTS = 3    # Always keep at least this many results after filtering

# ──────────────────────────────────────────────
# Re-Ranking Configuration
# ──────────────────────────────────────────────
# Two-stage retrieval: bi-encoder fetches RETRIEVAL_TOP_K candidates,
# cross-encoder re-ranks them, returns RERANKER_TOP_K best results.
# Toggle RERANKER_ENABLED to compare with/without re-ranking.
RERANKER_ENABLED = os.getenv("RERANKER_ENABLED", "true").lower() == "true"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # ~80MB, fast, accurate
RERANKER_TOP_K = 5           # Number of chunks to keep after re-ranking

# ──────────────────────────────────────────────
# Agent LLM Parameters
# ──────────────────────────────────────────────
# Temperature controls randomness:
#   0.0 = deterministic (best for factual extraction)
#   0.3 = slight creativity (useful for risk interpretation)
#   0.7+ = creative (we don't need this for legal analysis)

ORCHESTRATOR_TEMPERATURE = 0.0   # Deterministic routing decisions
ANALYSIS_TEMPERATURE = 0.0       # Factual, precise answers
RISK_TEMPERATURE = 0.2           # Slight flexibility for risk interpretation

# ──────────────────────────────────────────────
# Conversation Memory
# ──────────────────────────────────────────────
MAX_CONVERSATION_HISTORY = 10    # Number of past turns to keep in context

# ──────────────────────────────────────────────
# ChromaDB Collection
# ──────────────────────────────────────────────
COLLECTION_NAME = "legal_contracts"
