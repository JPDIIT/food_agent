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

def check_data_files() -> bool:
    """Check if demo data files exist."""
    data_dir = Path(__file__).parent / "data"
    foods_file = data_dir / "foods3.csv"
    daily_values_file = data_dir / "daily_values.csv"

    if not foods_file.exists() or not daily_values_file.exists():
        console.print("[red]Error: Demo data files not found![/red]")
        console.print(f"Expected files:")
        console.print(f"  - {foods_file}")
        console.print(f"  - {daily_values_file}")
        console.print("\nGenerate them with: python generate_demo_data.py")
        return False

    return True

def run_query_with_feedback(agent: DataAnalyticsAgent, query: str, show_code: bool = False):
    """
    Run a query and display results with nice formatting.

    Args:
        agent: DataAnalyticsAgent instance
        query: Natural language query
        show_code: Whether to display generated code
    """
    console.print(f"\n[bold cyan]Query:[/bold cyan] {query}")

    with console.status("[bold green]Processing...", spinner="dots"):
        result = agent.run(query)

    if result["status"] == "success":
        # Show answer
        console.print(Panel(
            Markdown(result["answer"]),
            title="[green]Answer[/green]",
            border_style="green"
        ))

        # Show code if requested
        if show_code and result.get("conversation_history"):
            for turn in result["conversation_history"]:
                if "generated_code" in turn.get("result", {}):
                    from rich.syntax import Syntax
                    code = turn["result"]["generated_code"]
                    console.print("\n[bold]Generated Code:[/bold]")
                    console.print(Syntax(code, "python", theme="monokai", line_numbers=True))

        # Show iterations
        console.print(f"[dim]Completed in {result.get('iterations', 0)} iterations[/dim]")

        return True
    else:
        console.print(Panel(
            f"[red]{result['error_message']}[/red]",
            title="[red]Error[/red]",
            border_style="red"
        ))
        return False

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