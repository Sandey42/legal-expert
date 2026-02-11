"""
Orchestrator Agent: the central coordinator of the multi-agent system.

RESPONSIBILITIES:
1. Rewrite follow-up questions into self-contained queries (query rewriting)
2. Classify the user's query (analysis / risk / out_of_scope)
3. Retrieve relevant document chunks
4. Route to the appropriate specialist agent
5. Maintain conversation history across turns (sliding window)
6. Generate follow-up suggestions for conversational engagement

MULTI-TURN STRATEGY: Hybrid (Query Rewriting + Sliding Window)
- Query Rewriting: Before classification/retrieval, the LLM rewrites vague follow-ups
  into self-contained queries. Guarded by referential language detection to prevent
  the 8B model from over-eagerly injecting document names from history.
- Sliding Window: Last N turns of raw history passed to specialist agents.
- Follow-up Suggestions: After each answer, LLM generates 2-3 contextual suggestions
  to guide the user and encourage multi-turn conversation.
"""

import re

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.config import (
    OLLAMA_MODEL,
    OLLAMA_BASE_URL,
    ORCHESTRATOR_TEMPERATURE,
    ANALYSIS_TEMPERATURE,
    RISK_TEMPERATURE,
    MAX_CONVERSATION_HISTORY,
)
from src.prompts.templates import (
    ORCHESTRATOR_SYSTEM_PROMPT,
    ORCHESTRATOR_USER_PROMPT,
    REWRITER_SYSTEM_PROMPT,
    REWRITER_USER_PROMPT,
    FOLLOWUP_SYSTEM_PROMPT,
    FOLLOWUP_USER_PROMPT,
    OUT_OF_SCOPE_RESPONSE,
    SOCIAL_RESPONSE,
)
from src.retriever import Retriever
from src.agents.analysis import AnalysisAgent
from src.agents.risk import RiskAgent


# ──────────────────────────────────────────────
# Rewrite Guard: patterns that indicate a follow-up referencing prior conversation.
# Only when these are detected do we invoke the LLM rewriter.
# Without these, the query is treated as self-contained.
# ──────────────────────────────────────────────
REFERENTIAL_PATTERNS = [
    r"\bthat\b",           # "what about that?"
    r"\bthis\b",           # "explain this further"
    r"\bthose\b",          # "what about those clauses?"
    r"\bthese\b",          # "are these risky?"
    r"\bit\b",             # "is it capped?"
    r"\bthem\b",           # "summarize them"
    r"\bthe same\b",       # "does the same apply?"
    r"\bthe one\b",        # "the one we discussed"
    r"\babove\b",          # "the above clause"
    r"\bprevious\b",       # "the previous answer"
    r"\bearlier\b",        # "you mentioned earlier"
    r"\bbefore\b",         # "what you said before"
    r"\byou said\b",       # "you said the notice period..."
    r"\byou told\b",       # "what you told me"
    r"\byou mentioned\b",  # "you mentioned liability"
    r"\btell me more\b",   # "tell me more"
    r"\bmore about\b",     # "more about that"
    r"\bwhat about\b",     # "what about the NDA?"
    r"\bhow about\b",      # "how about the other one?"
    r"\bfollow.?up\b",     # "follow up on that"
    r"\bexpand\b",         # "expand on that"
    r"\belaborate\b",      # "elaborate on the liability"
    r"\bcontinue\b",       # "continue"
    r"\bgo on\b",          # "go on"
    r"\band the\b",        # "and the vendor agreement?"
    # Conversational continuations -- user continues a topic from prior turn
    r"\blets\s+talk\b",    # "lets talk about section 1"
    r"\blet'?s\s+discuss\b",  # "let's discuss the liability"
    r"\btalk\s+about\b",   # "talk about section 2"
    r"\bdiscuss\b",        # "discuss the indemnification"
    r"\bsection\s+\d\b",   # "section 1" -- bare section ref needs context
    r"\bclause\b",         # "what does this clause say?"
]

# Short queries (< this many words) with conversation history are treated
# as follow-ups. Handles "yes", "sure", "the first one", "option 2", etc.
SHORT_QUERY_WORD_THRESHOLD = 4

# ──────────────────────────────────────────────
# Social Phrase Detection (Gate 0)
# ──────────────────────────────────────────────
# Common greetings, farewells, and pleasantries that should NOT enter the
# RAG pipeline. Matched against the full (stripped, lowered) query.
# These are whole-query patterns -- the ENTIRE input must match.
SOCIAL_PATTERNS = [
    # Greetings
    r"^(hi|hello|hey|howdy|yo)[\s!.]*$",
    r"^good\s+(morning|afternoon|evening|day)[\s!.]*$",
    r"^(what'?s\s+up|sup|wassup)[\s!?.]*$",
    r"^how\s+are\s+you[\s!?.]*$",
    r"^how'?s\s+it\s+going[\s!?.]*$",
    # Farewells
    r"^(bye|goodbye|good\s*bye|see\s+you|see\s+ya|later|ciao)[\s!.]*$",
    r"^good\s*night[\s!.]*$",
    r"^take\s+care[\s!.]*$",
    r"^(have\s+a\s+)?(good|great|nice)\s+(day|night|evening|one)[\s!.]*$",
    r"^(gotta\s+go|i'?m\s+off|signing\s+off|that'?s\s+all)[\s!.]*$",
    # Thanks / acknowledgements (not follow-ups)
    r"^(thanks|thank\s+you|thx|ty|cheers|much\s+appreciated)[\s!.]*$",
    r"^(ok|okay|alright|great|cool|perfect|awesome|nice|got\s+it)[\s,]*"
    r"(thanks|thank\s+you|thx|bye|goodbye|good\s*night)?[\s!.]*$",
]


def _is_social_phrase(text: str) -> bool:
    """
    Check if the entire input is a social/conversational phrase.

    Returns True for greetings ("hi"), farewells ("good night!"),
    and pleasantries ("thanks!") that should not enter the RAG pipeline.
    """
    cleaned = text.lower().strip()
    for pattern in SOCIAL_PATTERNS:
        if re.match(pattern, cleaned):
            return True
    return False


class Orchestrator:
    """
    Central coordinator for the multi-agent RAG system.

    Manages the full lifecycle of a query:
    rewrite -> classify -> retrieve -> route to agent -> suggest follow-ups
    """

    def __init__(self, retriever: Retriever | None = None):
        # ── LLM Wiring (centralized) ──────────────────────
        # All model selection and temperature config lives here.
        # Agents receive their LLM via dependency injection -- they don't
        # know or care whether it's Ollama, OpenAI, or a mock for testing.
        self._orchestrator_llm = ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=ORCHESTRATOR_TEMPERATURE,  # 0.0 = deterministic routing
        )
        analysis_llm = ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=ANALYSIS_TEMPERATURE,  # 0.0 = factual precision
        )
        risk_llm = ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=RISK_TEMPERATURE,  # 0.2 = interpretive flexibility
        )

        # ── Orchestrator Chains ───────────────────────────
        # Query rewriter chain
        self.rewriter_prompt = ChatPromptTemplate.from_messages([
            ("system", REWRITER_SYSTEM_PROMPT),
            ("human", REWRITER_USER_PROMPT),
        ])
        self.rewriter_chain = (
            self.rewriter_prompt | self._orchestrator_llm | StrOutputParser()
        )

        # Query classifier chain
        self.classifier_prompt = ChatPromptTemplate.from_messages([
            ("system", ORCHESTRATOR_SYSTEM_PROMPT),
            ("human", ORCHESTRATOR_USER_PROMPT),
        ])
        self.classifier_chain = (
            self.classifier_prompt | self._orchestrator_llm | StrOutputParser()
        )

        # Follow-up suggestion chain
        self.followup_prompt = ChatPromptTemplate.from_messages([
            ("system", FOLLOWUP_SYSTEM_PROMPT),
            ("human", FOLLOWUP_USER_PROMPT),
        ])
        self.followup_chain = (
            self.followup_prompt | self._orchestrator_llm | StrOutputParser()
        )

        # ── Specialist Agents (LLMs injected) ────────────
        self.retriever = retriever or Retriever()
        self.analysis_agent = AnalysisAgent(llm=analysis_llm)
        self.risk_agent = RiskAgent(llm=risk_llm)

        # Conversation memory: list of (question, answer) tuples
        # We store the ORIGINAL question (not rewritten) for natural history
        self.conversation_history: list[tuple[str, str]] = []

        # Store last follow-up suggestions for direct numeric selection
        self._last_follow_ups: list[str] = []

    # ──────────────────────────────────────────────
    # Query Rewriting
    # ──────────────────────────────────────────────

    def _needs_rewriting(self, question: str) -> bool:
        """
        Determine if a query needs rewriting using conversation history.

        Two triggers:
        1. Referential language detected ("that", "it", "tell me more", etc.)
        2. Very short query (< 4 words) -- likely a follow-up like "yes", "option 1"

        If neither trigger fires, the query is treated as self-contained
        and the rewriter LLM is NOT invoked.
        """
        question_lower = question.lower().strip()

        # Trigger 1: Referential language
        for pattern in REFERENTIAL_PATTERNS:
            if re.search(pattern, question_lower):
                return True

        # Trigger 2: Very short query (likely a follow-up or confirmation)
        # BUT exempt social phrases -- "good night!" is short but NOT a follow-up
        word_count = len(question_lower.split())
        if word_count < SHORT_QUERY_WORD_THRESHOLD and not _is_social_phrase(question):
            return True

        return False

    def _try_numeric_selection(self, question: str) -> str | None:
        """
        Check if the user typed a number to select a follow-up suggestion.

        Direct code-level mapping is more reliable than asking the 8B model
        to parse "3" → 3rd item in a numbered list (it often grabs #1).

        Returns the selected suggestion text, or None if not a numeric selection.
        """
        stripped = question.strip()

        # Match "1", "2", "3", or "option 1", "number 2", etc.
        match = re.match(r'^(?:option\s*|number\s*)?(\d)$', stripped, re.IGNORECASE)
        if not match:
            return None

        index = int(match.group(1)) - 1  # Convert to 0-based

        if 0 <= index < len(self._last_follow_ups):
            return self._last_follow_ups[index]

        return None

    def _rewrite_query(self, question: str) -> str:
        """
        Rewrite a follow-up question into a self-contained query.

        Uses a four-level gate:
        1. No conversation history → skip (nothing to reference)
        2. Numeric selection ("1", "2", "3") → direct map to stored suggestion
        3. No referential language and not short → skip (self-contained)
        4. Has triggers → invoke LLM rewriter
        """
        # Gate 1: No history = no rewriting needed
        if not self.conversation_history:
            return question

        # Gate 2: Numeric selection = direct map (no LLM needed)
        selected = self._try_numeric_selection(question)
        if selected:
            return selected

        # Gate 3: No triggers = query is self-contained
        if not self._needs_rewriting(question):
            return question

        # Gate 4: Has triggers → LLM rewrites using history
        chat_history = self._format_chat_history()
        rewritten = self.rewriter_chain.invoke({
            "question": question,
            "chat_history": chat_history,
        })
        return rewritten.strip().strip('"\'')

    # ──────────────────────────────────────────────
    # Query Classification
    # ──────────────────────────────────────────────

    def _classify_query(self, question: str) -> str:
        """
        Classify the query into: analysis, risk, or out_of_scope.

        Includes a document-name safety net: if the classifier says
        "out_of_scope" but the query mentions a known document,
        override to "analysis" -- UNLESS the query contains explicit
        out-of-scope verbs like "draft" or "write".
        """
        result = self.classifier_chain.invoke({"question": question})
        category = result.strip().lower().strip('"\'')

        valid_categories = {"analysis", "risk", "out_of_scope"}
        if category not in valid_categories:
            for valid in valid_categories:
                if valid in category:
                    category = valid
                    break
            else:
                category = "analysis"  # safe default

        # Safety net: override out_of_scope if query mentions a known document
        # but does NOT contain explicit out-of-scope verbs
        if category == "out_of_scope":
            from src.retriever import _detect_document_references
            doc_refs = _detect_document_references(question)

            if doc_refs and not self._has_out_of_scope_intent(question):
                category = "analysis"

        return category

    @staticmethod
    def _has_out_of_scope_intent(question: str) -> bool:
        """
        Check if the query has explicit out-of-scope intent even though
        it mentions a document name.

        "Draft a better NDA" → mentions NDA but intent is drafting (out of scope)
        "Lets discuss the NDA" → mentions NDA and intent is analysis (not out of scope)
        """
        out_of_scope_verbs = [
            r"\bdraft\b",
            r"\bwrite\b",
            r"\bcreate\b",
            r"\bgenerate\b",
            r"\brewrite\b",
            r"\bredraft\b",
            r"\bstrategy\b",
            r"\badvise\b",
            r"\brecommend\b",
            r"\bsuggest\s+changes\b",
        ]
        question_lower = question.lower()
        for pattern in out_of_scope_verbs:
            if re.search(pattern, question_lower):
                return True
        return False

    # ──────────────────────────────────────────────
    # Follow-Up Suggestions
    # ──────────────────────────────────────────────

    def _get_document_inventory(self) -> str:
        """
        Build a plain-text inventory of all documents and their sections.

        This is passed to the follow-up generator so it knows what's
        actually available in the corpus, preventing it from suggesting
        questions about information that doesn't exist.
        """
        all_docs = self.retriever.vector_store.get()
        if not all_docs or not all_docs.get("metadatas"):
            return "NDA, Vendor Services Agreement, SLA, DPA"

        # Build a doc → sections mapping
        doc_sections: dict[str, list[str]] = {}
        for meta in all_docs["metadatas"]:
            doc_name = meta.get("document", "Unknown")
            section = f"Section {meta.get('section_number', '?')} - {meta.get('section_title', '?')}"
            if doc_name not in doc_sections:
                doc_sections[doc_name] = []
            if section not in doc_sections[doc_name]:
                doc_sections[doc_name].append(section)

        # Format as readable text
        lines = []
        for doc, sections in sorted(doc_sections.items()):
            lines.append(f"• {doc}: {', '.join(sections)}")

        return "\n".join(lines)

    def _generate_followups(
        self, question: str, answer: str, sources: list[dict]
    ) -> list[str]:
        """
        Generate 3 contextual follow-up suggestions: deepen + broaden + risk.

        Grounded in the actual document inventory so every suggestion
        is answerable from the corpus. Prevents hallucinated questions
        about information that doesn't exist.
        """
        # Summarize the answer (first 300 chars) to save context window
        answer_summary = answer[:300] + "..." if len(answer) > 300 else answer

        # List unique documents referenced
        doc_names = list({s["document"] for s in sources})
        docs_str = ", ".join(doc_names) if doc_names else "multiple documents"

        # Get full document inventory for grounding
        inventory = self._get_document_inventory()

        # Format previous suggestions so the model knows what NOT to repeat
        if self._last_follow_ups:
            prev_str = "\n".join(f"- {s}" for s in self._last_follow_ups)
        else:
            prev_str = "None (this is the first question)."

        try:
            result = self.followup_chain.invoke({
                "question": question,
                "answer_summary": answer_summary,
                "documents_referenced": docs_str,
                "document_inventory": inventory,
                "previous_suggestions": prev_str,
            })

            # Parse numbered list (e.g., "1. Question?\n2. Question?")
            suggestions = []
            for line in result.strip().split("\n"):
                line = line.strip()
                # Remove numbering like "1.", "2.", "- "
                cleaned = re.sub(r'^[\d]+[\.\)]\s*', '', line)
                cleaned = re.sub(r'^[-•]\s*', '', cleaned)
                # Only keep actual questions (must contain '?' and be non-trivial)
                if cleaned and len(cleaned) > 10 and "?" in cleaned:
                    # Strip section references the 8B model keeps adding
                    cleaned = self._clean_followup(cleaned)
                    suggestions.append(cleaned)

            return suggestions[:3]  # Cap at 3 suggestions

        except Exception:
            # If follow-up generation fails, don't break the main flow
            return []

    @staticmethod
    def _clean_followup(text: str) -> str:
        """
        Clean up LLM-generated follow-up suggestions.

        The 8B model often ignores prompt instructions and adds section
        references, category labels, verbose phrases, etc. We fix these
        in code since prompt-only fixes are unreliable with smaller models.
        """
        # Remove category labels the 8B model leaks from the prompt instructions
        # e.g., "**DEEPEN**: What...", "BROADEN: How...", "**RISK**: Can..."
        text = re.sub(r'^\*{0,2}(DEEPEN|BROADEN|RISK|DEEPENING|BROADENING)\*{0,2}[:\s-]+', '', text, flags=re.IGNORECASE)
        # Remove section references: "under Section 1", "in Section 3 - Title", etc.
        text = re.sub(r'\s*(under|in|of|from|per|as stated in|as outlined in|outlined in)\s+Section\s+\d+(\s*-\s*[^,?.!]+)?', '', text, flags=re.IGNORECASE)
        # Remove standalone "Section X" references
        text = re.sub(r'\bSection\s+\d+(\s*-\s*[^,?.!]+)?', '', text, flags=re.IGNORECASE)
        # Remove "(file: ...)" references
        text = re.sub(r'\(file:\s*[^)]+\)', '', text)
        # Clean up double spaces and trailing/leading whitespace
        text = re.sub(r'\s{2,}', ' ', text).strip()
        # Remove trailing punctuation before ? (e.g., ",?" or " ?")
        text = re.sub(r'[\s,]+\?', '?', text)
        return text

    # ──────────────────────────────────────────────
    # Conversation History
    # ──────────────────────────────────────────────

    def _format_chat_history(self) -> str:
        """
        Format recent conversation history as a string.

        Uses a sliding window of the last N turns to prevent
        context window overflow with the local 8B model.
        """
        if not self.conversation_history:
            return "No previous conversation."

        recent = self.conversation_history[-MAX_CONVERSATION_HISTORY:]
        formatted = []
        for q, a in recent:
            formatted.append(f"User: {q}")
            truncated_answer = a[:500] + "..." if len(a) > 500 else a
            formatted.append(f"Assistant: {truncated_answer}")

        return "\n".join(formatted)

    # ──────────────────────────────────────────────
    # Main Pipeline
    # ──────────────────────────────────────────────

    def process_query(self, question: str) -> dict:
        """
        Process a user query through the full multi-agent pipeline.

        Pipeline: social check -> rewrite -> classify -> retrieve -> agent -> suggest follow-ups

        Returns:
            dict with keys:
              - answer: the final response text
              - query_type: "analysis", "risk", or "out_of_scope"
              - rewritten_query: the self-contained version (if rewritten)
              - sources: list of source metadata (for display)
              - follow_ups: list of suggested follow-up questions
        """
        # Gate 0: Social phrase detection (greetings, farewells, thanks)
        # Catches "good night!", "hi", "thanks!" etc. BEFORE any LLM call.
        # Deterministic, zero cost, prevents the rewriter from hallucinating.
        if _is_social_phrase(question):
            self.conversation_history.append((question, SOCIAL_RESPONSE))
            self._last_follow_ups = []
            return {
                "answer": SOCIAL_RESPONSE,
                "query_type": "social",
                "rewritten_query": None,
                "sources": [],
                "follow_ups": [],
            }

        # Step 1: Rewrite the query to be self-contained (if follow-up)
        rewritten_query = self._rewrite_query(question)
        was_rewritten = rewritten_query.lower() != question.lower()

        # Step 2: Classify using the rewritten query
        query_type = self._classify_query(rewritten_query)

        # Step 3: Handle out-of-scope immediately (no retrieval needed)
        if query_type == "out_of_scope":
            self.conversation_history.append((question, OUT_OF_SCOPE_RESPONSE))
            self._last_follow_ups = []
            return {
                "answer": OUT_OF_SCOPE_RESPONSE,
                "query_type": query_type,
                "rewritten_query": rewritten_query if was_rewritten else None,
                "sources": [],
                "follow_ups": [],
            }

        # Step 4: Retrieve relevant chunks using the REWRITTEN query
        # Single retrieval call; format the same results for the LLM context
        retrieved = self.retriever.retrieve(rewritten_query)
        context = self.retriever.format_results(retrieved)
        chat_history = self._format_chat_history()

        # Step 5: Route to the appropriate agent
        if query_type == "risk":
            answer = self.risk_agent.run(
                question=rewritten_query,
                context=context,
                chat_history=chat_history,
            )
        else:  # "analysis"
            answer = self.analysis_agent.run(
                question=rewritten_query,
                context=context,
                chat_history=chat_history,
            )

        # Step 6: Extract source metadata for display
        sources = [
            {
                "document": r["metadata"].get("document", "Unknown"),
                "section": r["metadata"].get("section_title", "Unknown"),
                "score": r["score"],
            }
            for r in retrieved
        ]

        # Step 7: Generate follow-up suggestions
        follow_ups = self._generate_followups(rewritten_query, answer, sources)

        # Step 8: Store follow-ups for direct numeric selection on next turn
        self._last_follow_ups = follow_ups

        # Step 9: Store in conversation history
        # Include follow-ups in history so "yes" / "option 1" can reference them
        history_answer = answer
        if follow_ups:
            history_answer += "\n\nSuggested follow-ups:\n"
            for i, suggestion in enumerate(follow_ups, 1):
                history_answer += f"{i}. {suggestion}\n"

        self.conversation_history.append((question, history_answer))

        return {
            "answer": answer,
            "query_type": query_type,
            "rewritten_query": rewritten_query if was_rewritten else None,
            "sources": sources,
            "follow_ups": follow_ups,
        }
