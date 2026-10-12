#!/usr/bin/env python3
"""
Google Workspace Assistant - Entry Point

Usage:
    python main.py --interactive
    python main.py "What meetings do I have today?"
"""

import argparse
import asyncio
from google.adk.runners import InMemoryRunner
from google.genai import types as genai_types
from rich.console import Console
from rich.prompt import Prompt
from rich.markdown import Markdown

from agent import create_agent

console = Console()

USER_ID = "user"
SESSION_ID = "session"


def run_query(runner, query: str) -> str:
    """Execute a query using the InMemoryRunner and return the response."""
    try:
        user_message = genai_types.Content(
            role="user",
            parts=[genai_types.Part(text=query)],
        )
        response_text = ""
        for event in runner.run(
            user_id=USER_ID,
            session_id=SESSION_ID,
            new_message=user_message,
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        response_text += part.text
        return response_text or "(No response)"
    except Exception as e:
        return f"Error: {str(e)}"


def interactive_mode(runner):
    """Run in interactive mode."""
    console.print("[bold green]Google Workspace Assistant[/bold green]")
    console.print("Type 'quit' to exit.\n")

    while True:
        query = Prompt.ask("[cyan]You[/cyan]")
        if query.lower() in ('quit', 'exit', 'q'):
            break
        if not query.strip():
            continue

        with console.status("Thinking..."):
            response = run_query(runner, query)

        console.print("\n[green]Assistant[/green]")
        console.print(Markdown(response))
        console.print()


def main():
    parser = argparse.ArgumentParser(description="Google Workspace Assistant")
    parser.add_argument("query", nargs="?", help="Query to send")
    parser.add_argument("--interactive", "-i", action="store_true")

    args = parser.parse_args()
    agent = create_agent()
    runner = InMemoryRunner(agent=agent)
    # Newer google-adk versions no longer create the session implicitly.
    runner.auto_create_session = True

    if args.interactive:
        interactive_mode(runner)
    elif args.query:
        response = run_query(runner, args.query)
        console.print(Markdown(response))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
