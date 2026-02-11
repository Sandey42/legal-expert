"""
Analysis Agent: answers factual questions about legal contracts.

THIS IS THE CORE Q&A AGENT. It:
1. Receives retrieved context (relevant document chunks)
2. Reads the chunks carefully
3. Generates a precise, cited answer

KEY DESIGN DECISIONS:
- Temperature = 0.0 (deterministic -- we want factual precision, not creativity)
- System prompt enforces grounding (must cite sources, must not hallucinate)
- Uses {context} + {question} prompt pattern standard in RAG systems
- LLM is injected via constructor (dependency injection) so the agent doesn't
  own its infrastructure -- the orchestrator controls wiring.
"""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.prompts.templates import ANALYSIS_SYSTEM_PROMPT, ANALYSIS_USER_PROMPT


class AnalysisAgent:
    """
    Factual Q&A agent for legal contract analysis.

    Given retrieved document context and a user question,
    generates a grounded, cited answer.

    The LLM is provided externally (dependency injection), making this agent
    testable (pass a mock) and independently reconfigurable without code changes.
    """

    def __init__(self, llm: BaseChatModel):
        self.llm = llm

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", ANALYSIS_SYSTEM_PROMPT),
            ("human", ANALYSIS_USER_PROMPT),
        ])

        # Chain: prompt -> LLM -> parse output as string
        # This is LangChain's "LCEL" (LangChain Expression Language) pattern
        self.chain = self.prompt | self.llm | StrOutputParser()

    def run(self, question: str, context: str, chat_history: str = "") -> str:
        """
        Answer a factual question using the provided context.

        Args:
            question: The user's natural language question.
            context: Formatted retrieved chunks from the retriever.
            chat_history: Previous conversation turns for context.

        Returns:
            A grounded, cited answer string.
        """
        response = self.chain.invoke({
            "question": question,
            "context": context,
            "chat_history": chat_history if chat_history else "No previous conversation.",
        })
        return response
