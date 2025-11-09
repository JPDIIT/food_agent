import os
import re
import sys
from io import StringIO
import time
from pathlib import Path
from typing import Optional, Dict, List, Any, Optional, Callable, Type, Union
from pydantic import BaseModel
import inspect

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm

from openai import OpenAI

from analytics_assistant.agent import DataAnalyticsAgent
from analytics_assistant.cli import DataAnalyticsCLI

console = Console()

def setup_llm_client(provider: str = "openai") -> tuple:
    """
    Set up LLM client with API key validation.

    Returns:
        (provider_name, client)
    """
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            console.print("[red]Error: OPENAI_API_KEY environment variable not set[/red]")
            console.print("Set it with: export OPENAI_API_KEY='your-key-here'")
            sys.exit(1)
        return "OpenAI GPT-4", OpenAI(api_key=api_key)

    else:
        console.print(f"[red]Unknown provider: {provider}[/red]")
        sys.exit(1)


def run_full_demo():
    console.print("[yellow]Full demo not implemented yet[/yellow]")
    sys.exit(1)

def run_interactive(provider: str = "openai"):
    """Run interactive CLI mode."""

    cli = DataAnalyticsCLI(llm_provider=provider)
    cli.run()

def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Workshop 07 Data Analytics Agent Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo.py                    # Interactive mode
  python demo.py --scenario quick   # Quick 5-minute demo
  python demo.py --scenario full    # Full 15-minute demo
  python demo.py --provider anthropic --scenario quick
        """
    )

    parser.add_argument(
        "--scenario",
        choices=["demo", "interactive"],
        help="Demo scenario to run (default: interactive)"
    )

    parser.add_argument(
        "--provider",
        choices=["openai"],
        default="openai",
        help="LLM provider (default: openai)"
    )

    args = parser.parse_args()

    # Set default provider for demo scenarios
    if args.scenario in ["demo"]:
        os.environ.setdefault("LLM_PROVIDER", args.provider)

    try:
        if args.scenario == "demo":
            run_full_demo()
        else:
            # Default to interactive
            run_interactive(args.provider)

    except KeyboardInterrupt:
        console.print("\n[yellow]Demo interrupted[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[red]Error: {str(e)}[/red]")
        import traceback
        console.print("[dim]" + traceback.format_exc() + "[/dim]")
        sys.exit(1)


if __name__ == "__main__":
    main()