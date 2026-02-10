"""
Quick end-to-end test of the RAG pipeline.

This tests the full flow:
  User question -> Retrieve chunks -> LLM generates answer

Run: python test_rag.py
"""

from src.retriever import Retriever
from src.agents.analysis import AnalysisAgent


def main():
    print("Loading retriever and analysis agent...")
    retriever = Retriever()
    agent = AnalysisAgent()

    test_queries = [
        "What is the uptime commitment in the SLA?",
        "What is the notice period for terminating the NDA?",
        "Which law governs the Vendor Services Agreement?",
    ]

    for query in test_queries:
        print(f"\n{'=' * 60}")
        print(f"QUESTION: {query}")
        print("-" * 60)

        # Step 1: Retrieve relevant chunks
        context = retriever.retrieve_formatted(query, top_k=5)

        # Step 2: Send to LLM with context
        answer = agent.run(question=query, context=context)

        print(f"\nANSWER:\n{answer}")
        print("=" * 60)


if __name__ == "__main__":
    main()
