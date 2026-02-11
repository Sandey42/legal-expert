"""
Ground-truth test cases for evaluating the multi-agent RAG system.

Each test case specifies:
- query: the user's natural language question
- expected_category: what the classifier should output (analysis / risk / out_of_scope)
- expected_documents: which document(s) should appear in retrieved results
- expected_sections: which section(s) should appear in retrieved results
- expected_phrases: phrases that MUST appear in the answer (case-insensitive)
- description: what this test case is designed to stress-test

These cover: the 16 sample queries from the assignment instructions, broad/summary
queries, and cross-document comparison queries (23 total). Ground truth was
manually derived from the actual document content.

WHY GROUND TRUTH?
- Deterministic and reproducible (no LLM-as-judge variability)
- Works fully local (no GPT-4 API needed)
- Each failure is immediately explainable ("expected section X, got section Y")
- Fast regression testing: change a prompt, re-run, see what broke
"""

TEST_CASES = [
    # ──────────────────────────────────────────────
    # ANALYSIS QUERIES: Factual extraction
    # ──────────────────────────────────────────────
    {
        "query": "What is the notice period for terminating the NDA?",
        "expected_category": "analysis",
        "expected_documents": ["NDA"],
        "expected_sections": ["Term and Termination"],
        "expected_phrases": ["thirty", "30", "days"],
        "description": "Single-doc factual extraction (NDA termination notice)",
    },
    {
        "query": "What is the uptime commitment in the SLA?",
        "expected_category": "analysis",
        "expected_documents": ["SLA"],
        "expected_sections": ["Service Availability"],
        "expected_phrases": ["99.5%"],
        "description": "Single-doc factual extraction (SLA uptime)",
    },
    {
        "query": "Which law governs the Vendor Services Agreement?",
        "expected_category": "analysis",
        "expected_documents": ["Vendor Services Agreement"],
        "expected_sections": ["Governing Law"],
        "expected_phrases": ["england", "wales"],
        "description": "Single-doc factual extraction (governing law)",
    },
    {
        "query": "Do confidentiality obligations survive termination of the NDA?",
        "expected_category": "analysis",
        "expected_documents": ["NDA"],
        "expected_sections": ["Term and Termination"],
        "expected_phrases": ["five", "5", "years", "survive"],
        "description": "Single-doc factual extraction (survival clause)",
    },
    {
        "query": "What remedies are available if the SLA uptime is not met?",
        "expected_category": "analysis",
        "expected_documents": ["SLA"],
        "expected_sections": ["Service Credits"],
        "expected_phrases": ["service credits"],
        "description": "Single-doc factual extraction (SLA remedies)",
    },
    {
        "query": "Which agreement governs data breach notification timelines?",
        "expected_category": "analysis",
        "expected_documents": ["DPA"],
        "expected_sections": ["Data Breach Notification"],
        "expected_phrases": ["72 hours"],
        "description": "Cross-doc identification (which agreement covers breach notification)",
    },
    {
        "query": "Can Vendor XYZ share Acme's confidential data with subcontractors?",
        "expected_category": "analysis",
        "expected_documents": ["DPA"],
        "expected_sections": ["Subprocessors"],
        "expected_phrases": ["written authorization"],
        "description": "Cross-doc factual extraction (subprocessor permissions)",
    },
    {
        "query": "What happens if Vendor delays breach notification beyond 72 hours?",
        "expected_category": "risk",
        "expected_documents": ["DPA"],
        "expected_sections": ["Data Breach Notification"],
        "expected_phrases": ["72 hours"],
        "description": "Risk identification + implication query (breach notification delay)",
    },

    # ──────────────────────────────────────────────
    # RISK QUERIES: Risk identification
    # ──────────────────────────────────────────────
    {
        "query": "Is liability capped for breach of confidentiality?",
        "expected_category": "risk",
        "expected_documents": ["NDA"],
        "expected_sections": ["Liability"],
        "expected_phrases": ["no explicit limitation", "not limited"],
        "description": "Risk identification (NDA unlimited liability)",
    },
    {
        "query": "Is Vendor XYZ's liability capped for data breaches?",
        "expected_category": "risk",
        "expected_documents": ["DPA"],
        "expected_sections": ["Liability"],
        "expected_phrases": ["vendor services agreement"],
        "description": "Risk identification (DPA liability defers to VSA)",
    },
    {
        "query": "Are there conflicting governing laws across agreements?",
        "expected_category": "risk",
        "expected_documents": ["NDA", "Vendor Services Agreement", "DPA"],
        "expected_sections": ["Governing Law"],
        "expected_phrases": ["california", "england", "european union"],
        "description": "Cross-document risk (conflicting governing laws across 3+ docs)",
    },
    {
        "query": "Are there any legal risks related to liability exposure?",
        "expected_category": "risk",
        "expected_documents": ["NDA"],
        "expected_sections": ["Liability"],
        "expected_phrases": ["no explicit limitation", "unlimited", "not limited"],
        "description": "Broad risk identification (liability exposure)",
    },
    {
        "query": "Identify any clauses that could pose financial risk to Acme Corp.",
        "expected_category": "risk",
        "expected_documents": ["NDA", "Vendor Services Agreement"],
        "expected_sections": ["Liability", "Limitation of Liability"],
        "expected_phrases": [],  # broad risk query, hard to pin exact phrases
        "description": "Broad risk identification (financial risk to Acme)",
    },
    {
        "query": "Is there any unlimited liability in these agreements?",
        "expected_category": "risk",
        "expected_documents": ["NDA"],
        "expected_sections": ["Liability"],
        "expected_phrases": ["no explicit limitation", "unlimited", "not limited"],
        "description": "Multi-doc risk (unlimited liability scan)",
    },
    {
        "query": "Summarize all risks for Acme Corp in one paragraph.",
        "expected_category": "risk",
        "expected_documents": [],  # broad query, any docs acceptable
        "expected_sections": [],   # broad query, any sections acceptable
        "expected_phrases": [],    # broad summary, hard to pin exact phrases
        "description": "Broad risk summary (tests risk agent with open-ended question)",
    },

    # ──────────────────────────────────────────────
    # BROAD / SUMMARY QUERIES: Document-scoped
    # ──────────────────────────────────────────────
    {
        "query": "Summarize the SLA",
        "expected_category": "analysis",
        "expected_documents": ["SLA"],
        "expected_sections": ["Service Availability", "Service Credits"],
        "expected_phrases": ["99.5%", "service credits"],
        "description": "Broad summary (tests metadata-filtered retrieval for single doc)",
    },
    {
        "query": "What are the key features of the NDA?",
        "expected_category": "analysis",
        "expected_documents": ["NDA"],
        "expected_sections": ["Confidentiality Obligations", "Term and Termination"],
        "expected_phrases": ["confidential"],
        "description": "Broad summary (tests NDA feature extraction)",
    },
    {
        "query": "What does the Data Processing Agreement cover?",
        "expected_category": "analysis",
        "expected_documents": ["DPA"],
        "expected_sections": ["Scope and Purpose", "Data Security"],
        "expected_phrases": ["personal data"],
        "description": "Broad summary (tests DPA scope overview)",
    },

    # ──────────────────────────────────────────────
    # CROSS-DOCUMENT QUERIES: Comparison / synthesis
    # ──────────────────────────────────────────────
    {
        "query": "Compare the termination clauses across all agreements.",
        "expected_category": "analysis",
        "expected_documents": ["NDA", "Vendor Services Agreement"],
        "expected_sections": ["Term and Termination", "Termination"],
        "expected_phrases": ["30", "60"],
        "description": "Cross-doc comparison (NDA 30 days vs Vendor 60 days termination)",
    },
    {
        "query": "Which agreements have governing law clauses and what do they say?",
        "expected_category": "analysis",
        "expected_documents": ["NDA", "Vendor Services Agreement", "DPA"],
        "expected_sections": ["Governing Law"],
        "expected_phrases": ["california", "england"],
        "description": "Cross-doc synthesis (governing law across all docs)",
    },
    {
        "query": "How do liability terms differ between the NDA and Vendor Services Agreement?",
        "expected_category": "analysis",
        "expected_documents": ["NDA", "Vendor Services Agreement"],
        "expected_sections": ["Liability", "Limitation of Liability"],
        "expected_phrases": ["no explicit limitation", "12 months"],
        "description": "Cross-doc comparison (NDA unlimited vs Vendor capped liability)",
    },

    # ──────────────────────────────────────────────
    # OUT-OF-SCOPE QUERIES: Boundary enforcement
    # ──────────────────────────────────────────────
    {
        "query": "Can you draft a better NDA for me?",
        "expected_category": "out_of_scope",
        "expected_documents": [],
        "expected_sections": [],
        "expected_phrases": ["cannot", "draft"],
        "description": "Out-of-scope: drafting request (should be refused)",
    },
    {
        "query": "What legal strategy should Acme take against Vendor XYZ?",
        "expected_category": "out_of_scope",
        "expected_documents": [],
        "expected_sections": [],
        "expected_phrases": ["cannot", "advice", "strateg"],
        "description": "Out-of-scope: legal advice request (should be refused)",
    },
]
