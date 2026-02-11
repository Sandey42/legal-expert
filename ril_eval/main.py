"""
Multi-Agent RAG System for Legal Contract Analysis
===================================================

Interactive console application that allows users to query and analyze
legal contracts using a multi-agent Retrieval-Augmented Generation system.

Usage:
    python main.py              # Start interactive console
    python main.py --reingest   # Force re-ingestion of documents

Architecture:
    User Query → Orchestrator (classifies) → Retriever (finds chunks)
    → Analysis Agent OR Risk Agent → Formatted Response with Citations
"""

import sys

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.text import Text

from src.ingestion import ingest_documents
from src.retriever import Retriever
from src.agents.orchestrator import Orchestrator

console = Console()


def display_welcome():
    """Show welcome banner with system capabilities."""
    welcome_text = """
# Legal Contract Analysis System

**Multi-Agent RAG System** for analyzing legal contracts.

**Loaded Documents:** NDA, Vendor Services Agreement, SLA, Data Processing Agreement

**What I can do:**
- Answer factual questions about contract terms and clauses
- Identify legal, financial, and compliance risks
- Compare terms across different agreements
- Maintain conversation context across multiple questions

**Commands:**
- Type your question and press Enter
- Type `quit` or `exit` to end the session
- Type `clear` to reset conversation history
    """
    console.print(Panel(Markdown(welcome_text), title="Welcome", border_style="blue"))


def display_response(result: dict):
    """Format and display the agent's response."""
    query_type = result["query_type"]
    answer = result["answer"]
    sources = result["sources"]
    rewritten = result.get("rewritten_query")

    # Show rewritten query if the question was a follow-up
    if rewritten:
        console.print(f"\n[dim italic]  Interpreted as: \"{rewritten}\"[/dim italic]")

    # Color-code by query type
    type_colors = {
        "analysis": "green",
        "risk": "red",
        "out_of_scope": "yellow",
        "social": "blue",
    }
    type_labels = {
        "analysis": "Factual Analysis",
        "risk": "Risk Assessment",
        "out_of_scope": "Out of Scope",
        "social": "Chat",
    }
    color = type_colors.get(query_type, "white")
    label = type_labels.get(query_type, "Response")

    # Display the answer
    console.print()
    console.print(Panel(
        Markdown(answer),
        title=f"[bold {color}]{label}[/bold {color}]",
        border_style=color,
    ))

    # Display sources (if any)
    if sources:
        source_lines = []
        for s in sources:
            # Score is already 0-1 relevance (higher = better),
            # normalized in the retriever for both bi-encoder and cross-encoder paths.
            score_display = f"(relevance: {s['score']:.2f})" if s['score'] else ""
            source_lines.append(f"  • {s['document']} → {s['section']} {score_display}")

        console.print(
            Panel(
                "\n".join(source_lines),
                title="[dim]Retrieved Sources[/dim]",
                border_style="dim",
            )
        )

    # Display follow-up suggestions (if any)
    follow_ups = result.get("follow_ups", [])
    if follow_ups:
        suggestion_lines = []
        for i, suggestion in enumerate(follow_ups, 1):
            suggestion_lines.append(f"  [cyan]{i}.[/cyan] {suggestion}")

        console.print(
            Panel(
                "\n".join(suggestion_lines),
                title="[bold blue]You might also ask[/bold blue]",
                border_style="blue",
            )
        )


def main():
    """Main interactive loop."""
    force_reingest = "--reingest" in sys.argv

    # Step 1: Ensure documents are ingested
    console.print("\n[bold]Initializing system...[/bold]")
    with console.status("[bold blue]Loading vector store..."):
        vector_store = ingest_documents(force=force_reingest)
        retriever = Retriever(vector_store)
        orchestrator = Orchestrator(retriever)

    console.print("[green]System ready![/green]\n")

    # Step 2: Welcome screen
    display_welcome()

    # Step 3: Interactive loop
    while True:
        try:
            console.print()
            question = console.input("[bold cyan]You:[/bold cyan] ").strip()

            if not question:
                continue

            if question.lower() in ("quit", "exit", "q"):
                console.print("\n[dim]Goodbye![/dim]\n")
                break

            if question.lower() == "clear":
                orchestrator.conversation_history.clear()
                console.print("[dim]Conversation history cleared.[/dim]")
                continue

            # Process the query through the multi-agent pipeline
            with console.status("[bold blue]Thinking...[/bold blue]"):
                result = orchestrator.process_query(question)

            display_response(result)

        except KeyboardInterrupt:
            console.print("\n\n[dim]Interrupted. Goodbye![/dim]\n")
            break


if __name__ == "__main__":
    main()
