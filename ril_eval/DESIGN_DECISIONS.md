# Design Decisions & Trade-offs

Tracking all architectural choices, why they were made, and what we'd do differently in production. This feeds into the README and interview presentation.

---

## 1. Chunking Strategy: Clause-Based

**Choice:** Split documents by numbered sections (1., 2., 3...) rather than fixed-size token windows.

**Why:** Legal documents have natural semantic boundaries. Each section covers one concept (e.g., "Liability", "Termination"). Fixed-size chunking splits clauses mid-sentence, losing meaning and hurting retrieval precision.

**Trade-off:** Clause-level chunks are small (~50-150 tokens). Broad queries need multiple chunks. Handled by retrieving top-K (5) results.

**Production enhancement:** Add hierarchical chunking -- clause-level for precision + document-level summary chunks for broad queries like "summarize the SLA."

---

## 2. Embedding Model: nomic-embed-text via Ollama

**Choice:** Ollama-hosted nomic-embed-text (768 dims) instead of sentence-transformers (all-MiniLM-L6-v2).

**Why:** sentence-transformers pulls PyTorch (~2GB). nomic-embed-text runs through Ollama (already installed), avoiding the heavy dependency. Quality is comparable for this corpus size.

**Trade-off:** Ollama embeddings require the Ollama server running. Slightly slower than in-process sentence-transformers.

**Production enhancement:** Benchmark against OpenAI text-embedding-3-small for quality comparison. Consider in-process embeddings for latency-critical deployments.

---

## 3. Vector Store: ChromaDB (File-Based)

**Choice:** ChromaDB with file persistence, no separate server.

**Why:** 4 documents, ~21 chunks. Anything heavier (Pinecone, Weaviate, pgvector) is overkill. ChromaDB runs in-process and persists to a folder.

**Trade-off:** Not suitable for large-scale production (millions of documents). No built-in access control.

**Production enhancement:** Migrate to pgvector or Qdrant for scale, concurrent access, and filtering capabilities.

---

## 4. LLM: llama3.1:8b via Ollama

**Choice:** Local 8B parameter model via Ollama. Temperature 0.0 for factual analysis, 0.2 for risk assessment.

**Why:** Runs locally (no API costs), good quality for structured extraction tasks. Different temperatures per agent control determinism vs. interpretive flexibility.

**Trade-off:** 8B models have limited instruction-following ability (we saw this with the query rewriter). Context window (~8K tokens) limits how much history/context we can pass.

**Production enhancement:** Use GPT-4o-mini or GPT-4o for better instruction following. Configurable via environment variable (already supported in config.py).

---

## 5. Multi-Turn Strategy: Hybrid (Query Rewriting + Sliding Window)

**Choice:** Four-component approach:
1. **Numeric selection** (code-level): "1", "2", "3" directly maps to stored follow-up suggestions
2. **Rewrite guard** (code-level): checks for referential language OR short queries (<4 words) before invoking rewriter
3. **Query rewriter** (LLM): resolves vague follow-ups using conversation history
4. **Sliding window**: last N turns of raw history passed to agents

**Why:** Pure LLM rewriting was too aggressive -- the 8B model injected document names from history into self-contained queries. The code-level guard prevents unnecessary rewrites while letting the LLM handle genuinely ambiguous follow-ups. Numeric selection bypasses the LLM entirely because the 8B model couldn't reliably map "3" → 3rd item in a list.

**Trade-off:** The referential pattern list is manually maintained. Could miss edge cases. Short-query threshold (<4 words) may occasionally trigger rewriting for brief but self-contained queries.

**Production enhancement:**
- Use a larger model (GPT-4o) that follows rewrite instructions more precisely, reducing need for code guards
- Track explored topics to generate smarter follow-ups
- Memory summarization for very long sessions (50+ turns)

---

## 6. Retrieval: Semantic Search + Metadata Filtering

**Choice:** Two-strategy retrieval:
- If user mentions exactly one document → filter ChromaDB to that document
- Otherwise → semantic search across all documents (top-K)

**Why:** Pure semantic search fails on broad, document-scoped queries ("summarize the SLA") because "summarize" doesn't match any specific section well. Metadata filtering ensures all sections of the named document are retrieved.

**Trade-off:** Document detection uses keyword matching (hardcoded aliases). Won't catch unusual references like "the confidentiality agreement" (though the query rewriter would typically resolve these).

**Production enhancement:**
- Hybrid retrieval (BM25 + semantic) for best of keyword and meaning-based search
- Union filter for multi-document queries
- Re-ranking with a cross-encoder model for better result ordering

---

## 7. Multi-Agent Architecture: 3 Agents + Orchestrator

**Choice:** Orchestrator + Analysis Agent + Risk Agent. No more, no less.

**Why:** Each agent has a distinct responsibility and prompt strategy:
- Orchestrator: classifies, routes, manages conversation (temp=0.0)
- Analysis: factual extraction with citations (temp=0.0)
- Risk: interpretive risk assessment with levels (temp=0.2)

Over-agentification (e.g., separate "citation agent", "formatting agent") adds complexity without clear value.

**Trade-off:** The orchestrator handles multiple concerns (rewriting, classification, follow-ups). Could be split further if any one concern becomes complex.

**Production enhancement:** Add a "comparison agent" for cross-document analysis queries. Add a "validation agent" that checks answers against source text for hallucination detection.

---

## 8. Prompt Design: Priority Hierarchy + Conversational Guidance

**Choice:**
- CONTEXT is labeled "PRIMARY SOURCE", history is "for reference only"
- Prompt ordering: HISTORY → CONTEXT → QUESTION (context nearest to question)
- Agents ask clarifying questions when queries are ambiguous
- Follow-up suggestions guide users toward unexplored areas

**Why:** Early testing showed the LLM over-weighting conversation history vs. retrieved context ("lost in the middle" effect). Explicit priority labels and strategic ordering fixed this.

**Trade-off:** Longer system prompts consume context window tokens.

**Production enhancement:** A/B test prompt variations. Implement prompt versioning for systematic optimization.

---

## 9. Follow-Up Suggestions: LLM-Generated, Grounded, Mixed Types

**Choice:** After each answer, generate 3 contextual follow-up suggestions using the LLM:
1. **Deepen** -- go deeper on the current topic/document
2. **Broaden** -- explore a different document the user hasn't asked about
3. **Risk** -- risk angle on what was just discussed

Grounded by passing the full document inventory (all documents + section titles) to the generator. Constrained to: max 10 words, no section number references, no external references (Schedule A/B), no questions about info already stated as "not found." Parsed with question-mark filtering to remove LLM preambles.

**Why:** Encourages multi-turn exploration. Guides users who don't know what to ask. Prevents tunnel vision on one topic. Grounding prevents hallucinated questions about information not in the corpus.

**Trade-off:** Extra LLM call per turn (~2-3 seconds). Follow-up quality depends on the LLM. The 8B model occasionally still generates verbose or section-referencing questions despite prompting.

**Production enhancement:** Track which documents/sections the user has explored, bias suggestions toward unexplored areas. Cache common follow-up patterns. Use a larger model for more reliable instruction following on constraints.

---

## 10. Numeric Selection for Follow-Ups: Code-Level Mapping

**Choice:** When user types "1", "2", "3", directly index into stored suggestions (code-level). Don't use the LLM rewriter.

**Why:** The 8B model couldn't reliably map "3" → 3rd item in a numbered list. It would grab item #1 regardless. Code-level indexing is 100% reliable and instant.

**Trade-off:** Only handles numeric input. "yes" or "the first one" still goes through the LLM rewriter.

**Production enhancement:** NLU-based intent detection for richer selection ("the second one", "both 1 and 3", "none of these").

---

## 11. Out-of-Scope Handling: Boundary Enforcement

**Choice:** Classifier detects out-of-scope queries ("draft a better NDA", "what legal strategy"). Returns a polite refusal with clear capability statement. No LLM call for the response (hardcoded).

**Why:** The sample queries include deliberate traps. A system that hallucinates legal advice is worse than one that says "I can't do that." Hardcoded response ensures consistency.

**Trade-off:** Binary classification (in-scope vs out-of-scope). Some borderline queries might be wrongly rejected.

**Production enhancement:** Softer handling -- "I can't draft an NDA, but I can highlight clauses in the current NDA that might need improvement."

---

## 12. Social Phrase Detection: Code-Level Gate 0

**Choice:** Before any LLM call (rewriting, classification, retrieval), check if the entire input matches a social phrase (greetings, farewells, thanks). If yes, return a friendly canned response immediately. Also exempt social phrases from the short-query rewrite trigger so they don't leak through as follow-ups.

**Why:** "good night!" is 2 words, which triggered the short-query rewrite guard. The rewriter then hallucinated a contract question from conversation history. The classifier saw a valid-looking question and routed it to analysis. The entire pipeline ran on a farewell. Two layers of defense:
1. **Gate 0 in `process_query`**: catches social phrases before any processing (primary defense)
2. **Exemption in `_needs_rewriting`**: prevents social phrases from triggering the short-query rewrite (safety net)

**Trade-off:** Finite pattern list -- creative phrasing ("catch you later!", "peace out") may slip through. However, the worst case is the classifier routing it to `out_of_scope`, which is acceptable.

**Production enhancement:** Use a lightweight intent classifier (fasttext or similar) to detect chitchat vs. task intent, covering novel phrasings without maintaining a pattern list.
