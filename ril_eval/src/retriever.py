"""
Retriever: finds the most relevant document chunks for a user query.

TWO-STAGE RETRIEVAL:
1. Bi-encoder stage: embed the query, find top-K candidates via cosine similarity
2. Cross-encoder stage (re-ranking): score each (query, chunk) pair with a
   cross-encoder model that reads both texts jointly for more accurate relevance

The bi-encoder is fast but approximate (encodes query and chunks independently).
The cross-encoder is slow but accurate (reads query + chunk together).
We use the bi-encoder to cast a wide net, then the cross-encoder to pick the best.

METADATA FILTERING:
When the user mentions a specific document (e.g., "SLA", "NDA"), we filter
ChromaDB to only search within that document's chunks. This solves the
"summarize the SLA" problem where broad queries fail with pure semantic search.

- Single document mentioned → filter to that document
- Multiple documents or none → normal semantic search across all
"""

import math
import re
from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config import (
    RETRIEVAL_TOP_K,
    SIMILARITY_THRESHOLD,
    RETRIEVAL_MIN_RESULTS,
    RERANKER_ENABLED,
    RERANKER_MODEL,
    RERANKER_TOP_K,
)
from src.ingestion import get_vector_store


def _sigmoid(x: float) -> float:
    """Convert a logit score to a 0-1 probability via sigmoid."""
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


# ──────────────────────────────────────────────
# Document name detection
# ──────────────────────────────────────────────
# Maps common ways users refer to each document → the metadata "document" value.
# This is intentionally simple: 4 documents with well-known names.
# For a production system with many documents, you'd use an LLM for this.

DOCUMENT_ALIASES: dict[str, str] = {
    # SLA
    "sla": "SLA",
    "service level agreement": "SLA",
    "service level": "SLA",
    # NDA
    "nda": "NDA",
    "non-disclosure agreement": "NDA",
    "non-disclosure": "NDA",
    "non disclosure": "NDA",
    "confidentiality agreement": "NDA",
    # DPA
    "dpa": "DPA",
    "data processing agreement": "DPA",
    "data processing": "DPA",
    # Vendor Services Agreement
    "vendor services agreement": "Vendor Services Agreement",
    "vendor agreement": "Vendor Services Agreement",
    "vendor services": "Vendor Services Agreement",
}


def _detect_document_references(query: str) -> list[str]:
    """
    Detect which documents are mentioned in the query.

    Scans for known document names/aliases (case-insensitive).
    Returns list of matching document metadata values.

    Examples:
        "summarize the SLA" → ["SLA"]
        "compare NDA and vendor agreement" → ["NDA", "Vendor Services Agreement"]
        "any unlimited liability?" → [] (no specific document)
    """
    query_lower = query.lower()
    found = set()

    # Check longest aliases first to avoid partial matches
    # (e.g., "vendor services agreement" before "vendor")
    sorted_aliases = sorted(DOCUMENT_ALIASES.keys(), key=len, reverse=True)

    for alias in sorted_aliases:
        # Use word boundary matching to avoid false positives
        # e.g., "sla" shouldn't match inside "translation"
        pattern = r'\b' + re.escape(alias) + r'\b'
        if re.search(pattern, query_lower):
            found.add(DOCUMENT_ALIASES[alias])

    return list(found)


class Retriever:
    """
    Retrieves relevant document chunks for a given query.

    Two-stage retrieval:
    1. Bi-encoder: ChromaDB cosine similarity (fast, approximate)
    2. Cross-encoder re-ranking: joint query+chunk scoring (slower, accurate)

    Also uses a two-strategy approach for initial retrieval:
    - If user mentions exactly one document → metadata filter (all chunks from that doc)
    - Otherwise → semantic search across all documents (top-K)
    """

    def __init__(self, vector_store: Chroma | None = None):
        self.vector_store = vector_store or get_vector_store()

        # Lazy-load the cross-encoder only when re-ranking is enabled.
        # Avoids loading the model (and PyTorch) if the feature is off.
        self._cross_encoder = None
        if RERANKER_ENABLED:
            self._load_cross_encoder()

    def _load_cross_encoder(self):
        """
        Load the cross-encoder model for re-ranking.

        Uses sentence-transformers' CrossEncoder which wraps a HuggingFace
        model that takes (query, document) pairs and outputs relevance scores.
        Model is downloaded on first use (~80MB) and cached locally.
        """
        from sentence_transformers import CrossEncoder
        print(f"Loading cross-encoder re-ranker: {RERANKER_MODEL}...")
        self._cross_encoder = CrossEncoder(RERANKER_MODEL)
        print("Re-ranker loaded.")

    def _rerank(self, query: str, results: list[dict]) -> list[dict]:
        """
        Re-rank retrieved chunks using the cross-encoder.

        The cross-encoder reads each (query, chunk) pair jointly and produces
        a relevance score. This is more accurate than bi-encoder cosine
        similarity because it models the interaction between query and chunk
        tokens through cross-attention.

        Args:
            query: The user's search query.
            results: List of dicts from bi-encoder retrieval (text, metadata, score).

        Returns:
            Re-ranked list, trimmed to RERANKER_TOP_K, with cross-encoder scores.
        """
        if not self._cross_encoder or not results:
            return results

        # Build (query, chunk_text) pairs for the cross-encoder
        pairs = [(query, r["text"]) for r in results]

        # Cross-encoder scores each pair: higher = more relevant
        scores = self._cross_encoder.predict(pairs)

        # Attach cross-encoder scores (normalized to 0-1 via sigmoid) and sort
        for result, ce_score in zip(results, scores):
            result["bi_encoder_score"] = result["score"]        # preserve original
            result["score"] = round(_sigmoid(float(ce_score)), 4)  # 0-1 relevance

        reranked = sorted(results, key=lambda r: r["score"], reverse=True)

        return reranked[:RERANKER_TOP_K]

    def retrieve(
        self,
        query: str,
        top_k: int = RETRIEVAL_TOP_K,
    ) -> list[dict]:
        """
        Retrieve the most relevant chunks for a query.

        Pipeline:
        1. Bi-encoder retrieval (ChromaDB cosine similarity) → top_k candidates
        2. Cross-encoder re-ranking (if enabled) → RERANKER_TOP_K best results

        Returns:
            List of dicts with: text, metadata, score
        """
        doc_refs = _detect_document_references(query)

        if len(doc_refs) == 1:
            # Single document: filter to it (solves "summarize the SLA" problem)
            results = self._retrieve_filtered(query, doc_refs[0], top_k)
        else:
            # No document or multiple: semantic search across everything
            results = self._retrieve_semantic(query, top_k)

        # Stage 2: Cross-encoder re-ranking
        if RERANKER_ENABLED and self._cross_encoder:
            results = self._rerank(query, results)

        # Stage 3: Filter out low-relevance chunks.
        # Prevents the agent from seeing irrelevant context that it might
        # hallucinate answers from. But always keeps at least RETRIEVAL_MIN_RESULTS
        # to support cross-document queries where individual chunks score low
        # but are needed together (e.g., "conflicting laws across agreements").
        results = self._filter_by_threshold(results)

        return results

    @staticmethod
    def _filter_by_threshold(results: list[dict]) -> list[dict]:
        """
        Remove chunks below the relevance threshold.

        Scores are 0-1 (higher = better) regardless of whether they came
        from the bi-encoder or cross-encoder path.

        Always keeps at least RETRIEVAL_MIN_RESULTS results. This prevents
        cross-document queries from being starved of context -- comparative
        queries like "conflicting laws across agreements" need chunks from
        multiple documents even if individual chunks score below threshold.
        """
        if not results:
            return results

        filtered = [r for r in results if r["score"] >= SIMILARITY_THRESHOLD]

        # Safety: always keep at least MIN_RESULTS (sorted by score, best first)
        if len(filtered) < RETRIEVAL_MIN_RESULTS:
            filtered = results[:RETRIEVAL_MIN_RESULTS]

        return filtered

    def _retrieve_semantic(self, query: str, top_k: int) -> list[dict]:
        """Standard semantic search across all documents."""
        results_with_scores = self.vector_store.similarity_search_with_score(
            query, k=top_k
        )

        retrieved = []
        for doc, distance in results_with_scores:
            # ChromaDB returns cosine distance (0-2, lower = closer).
            # Convert to 0-1 relevance (higher = better) for consistent scoring.
            relevance = max(0.0, 1.0 - distance)
            retrieved.append({
                "text": doc.page_content,
                "metadata": doc.metadata,
                "score": round(relevance, 4),
            })
        return retrieved

    def _retrieve_filtered(
        self, query: str, document_name: str, top_k: int
    ) -> list[dict]:
        """
        Retrieve chunks filtered to a specific document.

        Uses ChromaDB's metadata filter: WHERE document = "SLA"
        Then ranks by semantic similarity within that document.
        """
        results_with_scores = self.vector_store.similarity_search_with_score(
            query,
            k=top_k,
            filter={"document": document_name},
        )

        retrieved = []
        for doc, distance in results_with_scores:
            # ChromaDB returns cosine distance (0-2, lower = closer).
            # Convert to 0-1 relevance (higher = better) for consistent scoring.
            relevance = max(0.0, 1.0 - distance)
            retrieved.append({
                "text": doc.page_content,
                "metadata": doc.metadata,
                "score": round(relevance, 4),
            })
        return retrieved

    @staticmethod
    def format_results(results: list[dict]) -> str:
        """
        Format already-retrieved chunks as a string for LLM context.

        Each chunk is clearly labeled with its source and relevance score
        for citation. The score helps the agent prioritize high-relevance
        sources over low-relevance ones.
        """
        if not results:
            return "No relevant document sections found."

        formatted_chunks = []
        for result in results:
            meta = result["metadata"]
            doc_name = meta.get("document", "Unknown")
            section_num = meta.get("section_number", "?")
            section_title = meta.get("section_title", "Unknown")
            source_file = meta.get("source_file", "unknown")
            score = result.get("score", 0)

            header = f"[Source: {doc_name}, Section {section_num} - {section_title} (file: {source_file}) | relevance: {score:.2f}]"
            formatted_chunks.append(f"{header}\n{result['text']}")

        return "\n\n---\n\n".join(formatted_chunks)

    def retrieve_formatted(self, query: str, top_k: int = RETRIEVAL_TOP_K) -> str:
        """
        Retrieve chunks and format them as a string for LLM context.

        Convenience method that calls retrieve() then format_results().
        If you already have results from retrieve(), use format_results() directly
        to avoid a redundant embedding + search call.
        """
        return self.format_results(self.retrieve(query, top_k))


# ──────────────────────────────────────────────
# Quick test: run `python -m src.retriever`
# ──────────────────────────────────────────────
if __name__ == "__main__":
    retriever = Retriever()

    test_queries = [
        ("What is the uptime commitment in the SLA?", "Single doc, specific query"),
        ("Summarize the SLA", "Single doc, broad query (was broken before)"),
        ("Is liability capped for breach of confidentiality?", "No doc mentioned"),
        ("Compare NDA and vendor agreement governing laws", "Two docs mentioned"),
    ]

    for query, description in test_queries:
        print(f"\n{'=' * 60}")
        print(f"QUERY: {query}")
        print(f"TYPE:  {description}")
        refs = _detect_document_references(query)
        print(f"DETECTED DOCS: {refs if refs else 'None (semantic search)'}")
        print("-" * 60)
        results = retriever.retrieve(query, top_k=5)
        for r in results:
            doc = r["metadata"].get("document", "?")
            section = r["metadata"].get("section_title", "?")
            print(f"  • {doc} → {section} (score: {r['score']})")
