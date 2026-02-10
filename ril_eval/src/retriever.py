"""
Retriever: finds the most relevant document chunks for a user query.

HOW RETRIEVAL WORKS:
1. User asks: "What is the uptime commitment in the SLA?"
2. We embed that question into a vector (same model used during ingestion)
3. ChromaDB finds the chunks whose vectors are closest (cosine similarity)
4. We return the top-K most similar chunks

METADATA FILTERING:
When the user mentions a specific document (e.g., "SLA", "NDA"), we filter
ChromaDB to only search within that document's chunks. This solves the
"summarize the SLA" problem where broad queries fail with pure semantic search.

- Single document mentioned → filter to that document
- Multiple documents or none → normal semantic search across all
"""

import re
from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config import RETRIEVAL_TOP_K, SIMILARITY_THRESHOLD
from src.ingestion import get_vector_store


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

    Uses a two-strategy approach:
    - If user mentions exactly one document → metadata filter (all chunks from that doc)
    - Otherwise → semantic search across all documents (top-K)
    """

    def __init__(self, vector_store: Chroma | None = None):
        self.vector_store = vector_store or get_vector_store()

    def retrieve(
        self,
        query: str,
        top_k: int = RETRIEVAL_TOP_K,
    ) -> list[dict]:
        """
        Retrieve the most relevant chunks for a query.

        Strategy:
        - Single document mentioned → filter to that document's chunks
        - Multiple or no documents → semantic search across all

        Returns:
            List of dicts with: text, metadata, score
        """
        doc_refs = _detect_document_references(query)

        if len(doc_refs) == 1:
            # Single document: filter to it (solves "summarize the SLA" problem)
            return self._retrieve_filtered(query, doc_refs[0], top_k)
        else:
            # No document or multiple: semantic search across everything
            return self._retrieve_semantic(query, top_k)

    def _retrieve_semantic(self, query: str, top_k: int) -> list[dict]:
        """Standard semantic search across all documents."""
        results_with_scores = self.vector_store.similarity_search_with_score(
            query, k=top_k
        )

        retrieved = []
        for doc, score in results_with_scores:
            retrieved.append({
                "text": doc.page_content,
                "metadata": doc.metadata,
                "score": round(score, 4),
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
        for doc, score in results_with_scores:
            retrieved.append({
                "text": doc.page_content,
                "metadata": doc.metadata,
                "score": round(score, 4),
            })
        return retrieved

    def retrieve_formatted(self, query: str, top_k: int = RETRIEVAL_TOP_K) -> str:
        """
        Retrieve chunks and format them as a string for LLM context.

        Each chunk is clearly labeled with its source for citation.
        """
        results = self.retrieve(query, top_k)

        if not results:
            return "No relevant document sections found."

        formatted_chunks = []
        for i, result in enumerate(results, 1):
            meta = result["metadata"]
            doc_name = meta.get("document", "Unknown")
            section_num = meta.get("section_number", "?")
            section_title = meta.get("section_title", "Unknown")
            source_file = meta.get("source_file", "unknown")

            header = f"[Source: {doc_name}, Section {section_num} - {section_title} (file: {source_file})]"
            formatted_chunks.append(f"{header}\n{result['text']}")

        return "\n\n---\n\n".join(formatted_chunks)


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
