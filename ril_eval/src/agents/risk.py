"""
Risk Assessment Agent: identifies legal, financial, and compliance risks.

HOW THIS DIFFERS FROM THE ANALYSIS AGENT:
- Analysis Agent: extracts facts ("The uptime is 99.5%")
- Risk Agent: interprets implications ("No liability cap = HIGH RISK for unlimited exposure")

The Risk Agent has a slightly higher temperature (0.2) because risk interpretation
requires some reasoning flexibility -- it's not pure fact extraction.

It categorizes risks as HIGH / MEDIUM / LOW and explains why each matters.

LLM is injected via constructor (dependency injection) -- the orchestrator
controls model selection and temperature.
"""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.prompts.templates import RISK_SYSTEM_PROMPT, RISK_USER_PROMPT


class RiskAgent:
    """
    Risk assessment agent for legal contracts.

    Given retrieved document context and a risk-related question,
    identifies and categorizes potential risks with citations.

    The LLM is provided externally (dependency injection), making this agent
    testable (pass a mock) and independently reconfigurable without code changes.
    """

    def __init__(self, llm: BaseChatModel):
        self.llm = llm

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", RISK_SYSTEM_PROMPT),
            ("human", RISK_USER_PROMPT),
        ])

        self.chain = self.prompt | self.llm | StrOutputParser()

    def run(self, question: str, context: str, chat_history: str = "") -> str:
        """
        Assess risks related to the question using the provided context.

        Args:
            question: The user's risk-related question.
            context: Formatted retrieved chunks from the retriever.
            chat_history: Previous conversation turns for context.

        Returns:
            Risk assessment with HIGH/MEDIUM/LOW ratings and citations.
        """
        response = self.chain.invoke({
            "question": question,
            "context": context,
            "chat_history": chat_history if chat_history else "No previous conversation.",
        })
        return response
