"""
Evaluation framework for the Multi-Agent RAG system.

Runs all test cases through the pipeline and evaluates 4 dimensions:
1. Classification Accuracy -- did the query get routed correctly?
2. Retrieval Recall -- were the expected documents/sections retrieved?
3. Citation Faithfulness -- do cited sources exist in retrieved context?
4. Answer Correctness -- does the answer contain expected key phrases?

DESIGN CHOICES:
- Deterministic evaluation (no LLM-as-judge) -- reproducible, fast, explainable.
- Ground-truth based: each test case has manually verified expected results.
- All checks are case-insensitive to tolerate LLM phrasing variation.
- Results printed to console AND saved to CSV for easy sharing.

LIMITATIONS:
- Keyword matching is brittle to paraphrasing ("30 days" vs "a month").
- Cannot detect subtle hallucinations where the LLM adds details not in context
  while citing a real source. Would need RAGAS-style LLM judge for that.
- Ground truth requires manual effort -- doesn't scale to thousands of queries.
- Doesn't evaluate answer completeness, tone, or explanation quality.

Usage:
    python -m evaluation.evaluate              # Run all test cases
    python -m evaluation.evaluate --verbose    # Show full answers
    python -m evaluation.evaluate --csv        # Save results to CSV

Architecture:
    test_cases.py (ground truth) → evaluate.py (runner) → console report + CSV
"""

import re
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field

from evaluation.test_cases import TEST_CASES
from src.ingestion import ingest_documents
from src.retriever import Retriever
from src.agents.orchestrator import Orchestrator


# ──────────────────────────────────────────────
# Result data structures
# ──────────────────────────────────────────────

@dataclass
class TestResult:
    """Result of evaluating a single test case."""
    query: str
    description: str

    # Classification
    expected_category: str
    actual_category: str
    classification_correct: bool

    # Retrieval
    expected_documents: list[str]
    expected_sections: list[str]
    retrieved_documents: list[str]
    retrieved_sections: list[str]
    retrieval_doc_hits: int
    retrieval_doc_total: int
    retrieval_section_hits: int
    retrieval_section_total: int

    # Faithfulness
    citations_found: int
    citations_valid: int

    # Answer correctness
    expected_phrases: list[str]
    phrase_hits: int
    phrase_total: int

    # Full answer (for verbose mode)
    answer: str = ""

    # Timing
    latency_seconds: float = 0.0


# ──────────────────────────────────────────────
# Evaluation logic
# ──────────────────────────────────────────────

def _check_classification(result: dict, expected_category: str) -> tuple[str, bool]:
    """Check if the query was classified correctly."""
    actual = result["query_type"]
    return actual, actual == expected_category


def _check_retrieval(
    result: dict, expected_documents: list[str], expected_sections: list[str]
) -> tuple[list[str], list[str], int, int, int, int]:
    """
    Check if expected documents and sections appear in retrieved sources.

    This is a recall check: did we find what we needed?
    Case-insensitive matching to handle variations.
    """
    sources = result.get("sources", [])
    retrieved_docs = [s["document"] for s in sources]
    retrieved_sections = [s["section"] for s in sources]

    # Case-insensitive matching
    retrieved_docs_lower = [d.lower() for d in retrieved_docs]
    retrieved_sections_lower = [s.lower() for s in retrieved_sections]

    doc_hits = sum(
        1 for doc in expected_documents
        if doc.lower() in retrieved_docs_lower
    )
    section_hits = sum(
        1 for section in expected_sections
        if any(section.lower() in rs for rs in retrieved_sections_lower)
    )

    return (
        retrieved_docs,
        retrieved_sections,
        doc_hits,
        len(expected_documents),
        section_hits,
        len(expected_sections),
    )


def _check_faithfulness(result: dict) -> tuple[int, int]:
    """
    Check if citations in the answer correspond to actually retrieved sources.

    Parses [Source: DocName, Section X - Title] from the answer and verifies
    each cited document was in the retrieved context.

    This catches "citation hallucination" -- when the LLM invents a source
    that wasn't in its context.
    """
    answer = result.get("answer", "")
    sources = result.get("sources", [])

    # Parse citations from the answer
    citation_pattern = r'\[Source:\s*([^,\]]+)'
    citations = re.findall(citation_pattern, answer)

    if not citations:
        return 0, 0

    # Check each citation against retrieved sources
    retrieved_docs_lower = [s["document"].lower() for s in sources]
    valid = sum(
        1 for cite in citations
        if cite.strip().lower() in retrieved_docs_lower
    )

    return len(citations), valid


def _check_answer_correctness(
    result: dict, expected_phrases: list[str]
) -> tuple[int, int]:
    """
    Check if expected key phrases appear in the answer.

    Case-insensitive substring matching. Each phrase in expected_phrases
    must appear somewhere in the answer text.
    """
    answer = result.get("answer", "").lower()

    if not expected_phrases:
        return 0, 0

    hits = sum(1 for phrase in expected_phrases if phrase.lower() in answer)
    return hits, len(expected_phrases)


# ──────────────────────────────────────────────
# Main evaluation runner
# ──────────────────────────────────────────────

def run_evaluation(verbose: bool = False, save_csv: bool = False) -> list[TestResult]:
    """
    Run all test cases through the pipeline and evaluate.

    Returns list of TestResult objects for further analysis.
    """
    print("\n" + "=" * 70)
    print("  MULTI-AGENT RAG EVALUATION")
    print("=" * 70)

    # Initialize the system
    print("\nInitializing system...")
    vector_store = ingest_documents()
    retriever = Retriever(vector_store)
    orchestrator = Orchestrator(retriever)

    results: list[TestResult] = []

    print(f"\nRunning {len(TEST_CASES)} test cases...\n")
    print("-" * 70)

    for i, test_case in enumerate(TEST_CASES, 1):
        query = test_case["query"]
        print(f"\n[{i}/{len(TEST_CASES)}] {query}")

        # Clear conversation history between test cases
        # (each test case should be independent)
        orchestrator.conversation_history.clear()
        orchestrator._last_follow_ups.clear()

        # Run the query through the full pipeline
        start_time = time.time()
        pipeline_result = orchestrator.process_query(query)
        latency = time.time() - start_time

        # ── Evaluate all 4 dimensions ──

        # 1. Classification
        actual_category, classification_correct = _check_classification(
            pipeline_result, test_case["expected_category"]
        )

        # 2. Retrieval (skip for out_of_scope -- no retrieval expected)
        if test_case["expected_category"] == "out_of_scope":
            retrieved_docs, retrieved_sections = [], []
            doc_hits, doc_total, section_hits, section_total = 0, 0, 0, 0
        else:
            (retrieved_docs, retrieved_sections,
             doc_hits, doc_total,
             section_hits, section_total) = _check_retrieval(
                pipeline_result,
                test_case["expected_documents"],
                test_case["expected_sections"],
            )

        # 3. Faithfulness (skip for out_of_scope -- hardcoded response)
        if test_case["expected_category"] == "out_of_scope":
            citations_found, citations_valid = 0, 0
        else:
            citations_found, citations_valid = _check_faithfulness(pipeline_result)

        # 4. Answer correctness
        phrase_hits, phrase_total = _check_answer_correctness(
            pipeline_result, test_case["expected_phrases"]
        )

        # ── Build result ──

        test_result = TestResult(
            query=query,
            description=test_case["description"],
            expected_category=test_case["expected_category"],
            actual_category=actual_category,
            classification_correct=classification_correct,
            expected_documents=test_case["expected_documents"],
            expected_sections=test_case["expected_sections"],
            retrieved_documents=retrieved_docs,
            retrieved_sections=retrieved_sections,
            retrieval_doc_hits=doc_hits,
            retrieval_doc_total=doc_total,
            retrieval_section_hits=section_hits,
            retrieval_section_total=section_total,
            citations_found=citations_found,
            citations_valid=citations_valid,
            expected_phrases=test_case["expected_phrases"],
            phrase_hits=phrase_hits,
            phrase_total=phrase_total,
            answer=pipeline_result.get("answer", ""),
            latency_seconds=round(latency, 2),
        )
        results.append(test_result)

        # ── Print per-query results ──

        _print_query_result(test_result, verbose)

    # ── Print aggregate metrics ──
    _print_aggregate(results)

    # ── Save to CSV if requested ──
    if save_csv:
        _save_csv(results)

    return results


def _print_query_result(r: TestResult, verbose: bool = False):
    """Print evaluation results for a single query."""
    check = "✓"
    cross = "✗"

    # Classification
    cls_icon = check if r.classification_correct else cross
    print(f"  Classification:  {cls_icon} {r.actual_category} (expected: {r.expected_category})")

    # Retrieval
    if r.expected_category == "out_of_scope":
        print(f"  Retrieval:       — skipped (out of scope)")
    elif r.retrieval_doc_total == 0 and r.retrieval_section_total == 0:
        print(f"  Retrieval:       — broad query (any docs acceptable)")
    else:
        doc_icon = check if r.retrieval_doc_hits == r.retrieval_doc_total else cross
        sec_icon = check if r.retrieval_section_hits == r.retrieval_section_total else cross
        print(f"  Retrieval docs:  {doc_icon} {r.retrieval_doc_hits}/{r.retrieval_doc_total} found")
        print(f"  Retrieval secs:  {sec_icon} {r.retrieval_section_hits}/{r.retrieval_section_total} found")

    # Faithfulness
    if r.expected_category == "out_of_scope":
        print(f"  Faithfulness:    — skipped (hardcoded response)")
    elif r.citations_found == 0:
        print(f"  Faithfulness:    — no citations found in answer")
    else:
        faith_icon = check if r.citations_valid == r.citations_found else cross
        print(f"  Faithfulness:    {faith_icon} {r.citations_valid}/{r.citations_found} citations verified")

    # Answer correctness
    if r.phrase_total == 0:
        print(f"  Answer phrases:  — no expected phrases (broad query)")
    else:
        phrase_icon = check if r.phrase_hits == r.phrase_total else cross
        print(f"  Answer phrases:  {phrase_icon} {r.phrase_hits}/{r.phrase_total} expected phrases found")

    print(f"  Latency:         {r.latency_seconds}s")

    if verbose:
        print(f"\n  Answer (first 300 chars):")
        print(f"  {r.answer[:300]}...")

    print("-" * 70)


def _print_aggregate(results: list[TestResult]):
    """Print aggregate metrics across all test cases."""
    print("\n" + "=" * 70)
    print("  AGGREGATE METRICS")
    print("=" * 70)

    # Classification accuracy
    cls_correct = sum(1 for r in results if r.classification_correct)
    cls_total = len(results)
    print(f"\n  Classification Accuracy:  {cls_correct}/{cls_total} ({100 * cls_correct / cls_total:.1f}%)")

    # Retrieval recall (only for in-scope queries with expected docs/sections)
    retrieval_results = [
        r for r in results
        if r.expected_category != "out_of_scope"
        and (r.retrieval_doc_total > 0 or r.retrieval_section_total > 0)
    ]
    if retrieval_results:
        doc_hits = sum(r.retrieval_doc_hits for r in retrieval_results)
        doc_total = sum(r.retrieval_doc_total for r in retrieval_results)
        sec_hits = sum(r.retrieval_section_hits for r in retrieval_results)
        sec_total = sum(r.retrieval_section_total for r in retrieval_results)
        if doc_total > 0:
            print(f"  Retrieval Doc Recall:     {doc_hits}/{doc_total} ({100 * doc_hits / doc_total:.1f}%)")
        if sec_total > 0:
            print(f"  Retrieval Section Recall: {sec_hits}/{sec_total} ({100 * sec_hits / sec_total:.1f}%)")

    # Faithfulness
    faith_results = [r for r in results if r.citations_found > 0]
    if faith_results:
        cite_valid = sum(r.citations_valid for r in faith_results)
        cite_total = sum(r.citations_found for r in faith_results)
        print(f"  Citation Accuracy:        {cite_valid}/{cite_total} ({100 * cite_valid / cite_total:.1f}%)")

    # Answer correctness (only for queries with expected phrases)
    phrase_results = [r for r in results if r.phrase_total > 0]
    if phrase_results:
        p_hits = sum(r.phrase_hits for r in phrase_results)
        p_total = sum(r.phrase_total for r in phrase_results)
        print(f"  Answer Correctness:       {p_hits}/{p_total} ({100 * p_hits / p_total:.1f}%)")

    # Average latency
    avg_latency = sum(r.latency_seconds for r in results) / len(results)
    print(f"\n  Average Latency:          {avg_latency:.1f}s per query")
    print(f"  Total Test Cases:         {len(results)}")

    print("\n" + "=" * 70)


def _save_csv(results: list[TestResult]):
    """Save evaluation results to a CSV file."""
    import pandas as pd

    output_path = Path(__file__).parent / "results.csv"

    rows = []
    for r in results:
        rows.append({
            "query": r.query,
            "description": r.description,
            "expected_category": r.expected_category,
            "actual_category": r.actual_category,
            "classification_correct": r.classification_correct,
            "retrieval_doc_recall": (
                f"{r.retrieval_doc_hits}/{r.retrieval_doc_total}"
                if r.retrieval_doc_total > 0 else "n/a"
            ),
            "retrieval_section_recall": (
                f"{r.retrieval_section_hits}/{r.retrieval_section_total}"
                if r.retrieval_section_total > 0 else "n/a"
            ),
            "citations_valid": (
                f"{r.citations_valid}/{r.citations_found}"
                if r.citations_found > 0 else "n/a"
            ),
            "phrase_hits": (
                f"{r.phrase_hits}/{r.phrase_total}"
                if r.phrase_total > 0 else "n/a"
            ),
            "latency_seconds": r.latency_seconds,
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"\n  Results saved to: {output_path}")


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    verbose = "--verbose" in sys.argv
    save_csv = "--csv" in sys.argv
    run_evaluation(verbose=verbose, save_csv=save_csv)
