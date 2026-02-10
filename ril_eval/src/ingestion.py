"""
Ingestion pipeline: embed document chunks and store in ChromaDB.

HOW THIS WORKS:
1. Takes the clause-based chunks from chunker.py
2. Sends each chunk's text to Ollama's nomic-embed-text model
3. Gets back a 768-dimensional vector for each chunk
4. Stores text + vector + metadata in ChromaDB (a lightweight vector database)
5. ChromaDB persists to disk, so we only need to ingest once

WHY ChromaDB?
- Lightweight: runs in-process, no separate server needed
- File-based persistence: just a folder on disk
- Good LangChain integration
- For 4 documents (~20 chunks), anything heavier (Pinecone, Weaviate) is overkill

WHY nomic-embed-text VIA OLLAMA?
- Runs locally (no API costs, no network dependency)
- 768 dimensions -- good quality for semantic similarity
- Already using Ollama for LLM, so no extra infrastructure
- Avoids pulling torch (~2GB) that sentence-transformers requires
"""

from pathlib import Path

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config import (
    OLLAMA_BASE_URL,
    EMBEDDING_MODEL,
    CHROMA_PERSIST_DIR,
    COLLECTION_NAME,
)
from src.chunker import chunk_all_documents, DocumentChunk


def _chunks_to_langchain_docs(chunks: list[DocumentChunk]) -> list[Document]:
    """
    Convert our DocumentChunk objects to LangChain Document objects.

    LangChain's vector stores expect Document(page_content=..., metadata=...).
    This is just a format conversion -- no logic change.
    """
    return [
        Document(page_content=chunk.text, metadata=chunk.metadata)
        for chunk in chunks
    ]


def get_embedding_function() -> OllamaEmbeddings:
    """
    Create the embedding function using Ollama.

    This is the function that converts text -> vector.
    We use it both during ingestion (embed chunks) and during
    retrieval (embed the user's query).
    """
    return OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )


def get_vector_store() -> Chroma:
    """
    Get a handle to the ChromaDB vector store.

    If the store already exists on disk (from a previous ingestion),
    it loads the existing data. Otherwise, it creates an empty store.
    """
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embedding_function(),
        persist_directory=str(CHROMA_PERSIST_DIR),
    )


def ingest_documents(force: bool = False) -> Chroma:
    """
    Main ingestion pipeline: chunk documents -> embed -> store in ChromaDB.

    Args:
        force: If True, re-ingest even if the vector store already exists.
               If False (default), skip ingestion if data already exists.

    Returns:
        The ChromaDB vector store, ready for retrieval.
    """
    vector_store = get_vector_store()

    # Check if we already have data (avoid re-ingesting on every run)
    existing_count = vector_store._collection.count()
    if existing_count > 0 and not force:
        print(f"Vector store already contains {existing_count} chunks. Skipping ingestion.")
        print("  (Use force=True to re-ingest)")
        return vector_store

    # If forcing re-ingestion, clear existing data
    if force and existing_count > 0:
        print(f"Clearing {existing_count} existing chunks...")
        # Delete and recreate
        vector_store.delete_collection()
        vector_store = get_vector_store()

    print("Starting document ingestion...")
    print("-" * 40)

    # Step 1: Chunk all documents
    print("\n[Step 1/2] Chunking documents by clause...")
    chunks = chunk_all_documents()

    # Step 2: Embed and store
    print("\n[Step 2/2] Embedding chunks and storing in ChromaDB...")
    langchain_docs = _chunks_to_langchain_docs(chunks)
    vector_store.add_documents(langchain_docs)

    final_count = vector_store._collection.count()
    print(f"\nIngestion complete! {final_count} chunks stored in ChromaDB.")
    print(f"Persisted to: {CHROMA_PERSIST_DIR}")

    return vector_store


# ──────────────────────────────────────────────
# Quick test: run `python -m src.ingestion` to ingest documents
# ──────────────────────────────────────────────
if __name__ == "__main__":
    store = ingest_documents(force=True)
    print("\n" + "=" * 60)
    print("Testing retrieval with a sample query...")
    results = store.similarity_search("What is the uptime commitment?", k=3)
    for i, doc in enumerate(results, 1):
        print(f"\n--- Result {i} ---")
        print(f"  Document: {doc.metadata.get('document')}")
        print(f"  Section:  {doc.metadata.get('section_title')}")
        print(f"  Text:     {doc.page_content[:150]}...")
