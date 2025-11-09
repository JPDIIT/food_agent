"""Interactive CLI interface for the Data Analytics Agent.

This module provides a command-line interface for interacting with the
DataAnalyticsAgent using the Rich library for beautiful terminal output.
"""

import os
import sys
import base64
import tempfile
from typing import Optional
from pathlib import Path
import io

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.table import Table
from rich import print as rprint
from rich.prompt import Confirm

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
from prompt_toolkit.formatted_text import HTML

from openai import OpenAI

from .agent import DataAnalyticsAgent


class DataAnalyticsCLI:
    """Interactive command-line interface for the data analytics agent."""

    def __init__(self, llm_provider: str = "openai", api_key: Optional[str] = None):
        """
        Initialize the CLI.

        Args:
            llm_provider: "openai"
            api_key: API key for the LLM provider (uses env var if not provided)
        """
        self.console = Console()
        self.llm_provider = llm_provider

        # Initialize LLM client
        if llm_provider == "openai":
            api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
            self.llm_client = OpenAI(api_key=api_key)
        else:
            raise ValueError(f"Unknown LLM provider: {llm_provider}. Use 'openai'.")

        # Initialize agent
        self.agent = DataAnalyticsAgent(
            llm_client=self.llm_client,
            max_iterations=20,
            verbose=False  # CLI handles its own output
        )

        self.running = True

        # Initialize prompt session with history
        history_file = Path.home() / ".claude_agent_history"
        self.session = PromptSession(
            history=FileHistory(str(history_file)),
            style=Style.from_dict({
                'prompt': '#00aaaa bold',  # Cyan color for "You:"
            })
        )

    def run(self):
        """Start the interactive CLI loop."""
        self._print_welcome()

        while self.running:
            try:
                # Get user input with history support
                query = self.session.prompt(HTML('\n<prompt>You: </prompt>'))

                if not query.strip():
                    continue

                # Check for commands
                if query.startswith('/'):
                    self._handle_command(query)
                    continue

                # Process query with agent
                self._process_query(query)

            except KeyboardInterrupt:
                self.console.print("\n[yellow]Use /exit to quit[/yellow]")
                continue  # Continue loop instead of breaking
            except EOFError:
                self.running = False
            except Exception as e:
                self.console.print(f"[red]Unexpected error: {str(e)}[/red]")

        self._print_goodbye()

    def _print_welcome(self):
        """Print welcome message."""
        welcome_text = """
# Data Analytics Agent

Ask questions about your CSV data in natural language!

**Available Commands:**
- `/load <path> <alias>` - Load a CSV file
- `/list` - List loaded datasets
- `/clear` - Clear conversation memory
- `/help` - Show this help message
- `/exit` - Exit the CLI

**Example Queries:**
- "What columns are in the foods data?"
- "Show me the top 5 foods by calories"
- "Create a bar chart of foods with higher than average sodium"
- "Calculate average amounts of each nutrient"

**Tips:**
- Start by loading a dataset with `/load`
- Ask natural language questions about your data
- Request visualizations and charts
- Use specific column names for better results
"""

        self.console.print(Panel(
            Markdown(welcome_text),
            title=f"[bold green]Welcome![/bold green]",
            subtitle=f"Powered by {self.llm_provider.upper()}",
            border_style="green"
        ))

    def _handle_command(self, command: str):
        """Handle CLI commands."""
        parts = command.split(maxsplit=2)
        cmd = parts[0].lower()

        if cmd == "/exit" or cmd == "/quit":
            self.running = False

        elif cmd == "/help":
            self._print_welcome()

        elif cmd == "/clear":
            self.agent.clear_memory()
            self.console.print("[green]✓ Memory cleared![/green]")

        elif cmd == "/list":
            self._list_datasets()

        elif cmd == "/load":
            if len(parts) < 3:
                self.console.print("[red]Usage: /load <path> <alias>[/red]")
                self.console.print("[yellow]Example: /load data/foods3.csv foods[/yellow]")
                return

            file_path = parts[1]
            alias = parts[2]

            # Validate file exists
            if not Path(file_path).exists():
                self.console.print(f"[red]Error: File not found: {file_path}[/red]")
                return

            # Call load_csv tool directly (bypass agent for deterministic operation)
            from .tools.dataset import LoadCSVTool
            load_tool = LoadCSVTool()

            with self.console.status("[bold green]Loading dataset..."):
                result = load_tool.execute(file_path=file_path, alias=alias)

            if result["status"] == "success":
                self.console.print(f"[green]✓ {result['result']}[/green]")
            else:
                error_msg = result.get("error_message", "Unknown error")
                self.console.print(f"[red]✗ {error_msg}[/red]")

        else:
            self.console.print(f"[red]Unknown command: {cmd}[/red]")
            self.console.print("[yellow]Type /help for available commands[/yellow]")

    def _list_datasets(self):
        """List loaded datasets."""
        datasets = self.agent.get_loaded_datasets()

        if not datasets:
            self.console.print("[yellow]No datasets loaded[/yellow]")
            self.console.print("[dim]Use /load <path> <alias> to load a CSV file[/dim]")
            return

        table = Table(title="Loaded Datasets", show_header=True, header_style="bold magenta")
        table.add_column("Alias", style="cyan", no_wrap=True)
        table.add_column("Rows", justify="right", style="green")
        table.add_column("Columns", justify="right", style="green")
        table.add_column("Memory (MB)", justify="right", style="yellow")
        table.add_column("Column Names", style="dim")

        for ds in datasets:
            # Get column names (first 5)
            columns = ds.get("column_names", [])
            col_display = ", ".join(columns[:5])
            if len(columns) > 5:
                col_display += f" ... (+{len(columns) - 5} more)"

            table.add_row(
                ds["alias"],
                str(ds["rows"]),
                str(ds["columns"]),
                f"{ds['memory_mb']:.2f}",
                col_display
            )

        self.console.print(table)

    def _process_query(self, query: str):
        """Process user query with the agent."""
        # Show processing message
        with self.console.status("[bold green]Thinking...", spinner="dots"):
            result = self.agent.run(query)

        # Display result
        if result["status"] == "success":
            self._display_success(result)
        else:
            self._display_error(result)

    def _display_success(self, result: dict):
        """Display successful result."""
        # Show answer
        answer_panel = Panel(
            Markdown(result["answer"]),
            title="[bold green]Answer[/bold green]",
            border_style="green",
            padding=(1, 2)
        )
        self.console.print(answer_panel)

        # Display visualizations
        if result.get("visualizations"):
            self._display_visualizations(result["visualizations"])

        # Show conversation history option if there were multiple steps
        if result.get("conversation_history") and len(result["conversation_history"]) > 1:
            self._show_history_option(result["conversation_history"])

        # Show stats
        iterations = result.get("iterations", 0)
        total_time = result.get("total_time", 0)

        stats = f"[dim]Completed in {iterations} iteration"
        if iterations != 1:
            stats += "s"
        if total_time > 0:
            stats += f" ({total_time:.2f}s)"
        stats += "[/dim]"

        self.console.print(stats)

    def _display_error(self, result: dict):
        """Display error result."""
        error_msg = result.get("error_message", "Unknown error occurred")

        # Create detailed error message
        error_text = f"**Error:** {error_msg}\n\n"

        # Add context if available
        if result.get("conversation_history"):
            last_turn = result["conversation_history"][-1]
            if "error" in last_turn.get("result", {}):
                error_text += f"**Details:** {last_turn['result']['error']}\n\n"

        error_text += "**Suggestions:**\n"
        error_text += "- Check if the dataset is loaded (`/list`)\n"
        error_text += "- Verify column names in your query\n"
        error_text += "- Try rephrasing your question\n"
        error_text += "- Use `/clear` to reset and try again"

        error_panel = Panel(
            Markdown(error_text),
            title="[bold red]Error[/bold red]",
            border_style="red",
            padding=(1, 2)
        )
        self.console.print(error_panel)

    def _display_visualizations(self, visualizations: list):
        """Display visualization file paths."""
        if not visualizations:
            return

        self.console.print(f"\n[bold cyan]Generated {len(visualizations)} visualization(s):[/bold cyan]")

        for i, viz in enumerate(visualizations, 1):
            # viz is now a dict with filepath, chart_type, query
            filepath = viz.get("filepath", "")
            chart_type = viz.get("chart_type", "chart")

            self.console.print(f"  {i}. [{chart_type}] {filepath}")

    def _show_history_option(self, history: list):
        """Optionally show conversation history."""
        if len(history) <= 1:
            return

        show = Confirm.ask(
            "[dim]Show conversation history?[/dim]",
            default=False
        )

        if show:
            self._display_history(history)

    def _display_history(self, history: list):
        """Display conversation history."""
        for i, turn in enumerate(history, 1):
            # Create panel for each turn
            turn_text = ""

            # Thought
            if turn.get("thought"):
                turn_text += f"**Thought:** {turn['thought']}\n\n"

            # Action
            if turn.get("action"):
                turn_text += f"**Action:** {turn['action']}\n\n"

            # Code
            if "generated_code" in turn.get("result", {}):
                code = turn["result"]["generated_code"]
                turn_text += "**Generated Code:**\n"

                panel = Panel(
                    Markdown(turn_text),
                    title=f"[bold]Turn {i}[/bold]",
                    border_style="blue",
                    padding=(1, 2)
                )
                self.console.print(panel)

                # Display code with syntax highlighting
                syntax = Syntax(code, "python", theme="monokai", line_numbers=True)
                self.console.print(syntax)

                # Result or error
                if turn["result"].get("success"):
                    self.console.print("[green]✓ Execution successful[/green]")
                    if "output" in turn["result"]:
                        self.console.print(f"[dim]Output: {turn['result']['output']}[/dim]")
                else:
                    error = turn["result"].get("error", "Unknown error")
                    self.console.print(f"[red]✗ Execution failed: {error}[/red]")
            else:
                panel = Panel(
                    Markdown(turn_text),
                    title=f"[bold]Turn {i}[/bold]",
                    border_style="blue",
                    padding=(1, 2)
                )
                self.console.print(panel)

            self.console.print()  # Blank line between turns

    def _print_goodbye(self):
        """Print goodbye message."""
        goodbye_text = """
# Thanks for using the Data Analytics Agent!

Your data exploration session has ended.

**Next Steps:**
- Try the agent on different datasets
- Explore more complex queries
- Check out the documentation for advanced features

Happy analyzing!
"""

        self.console.print(Panel(
            Markdown(goodbye_text),
            title="[bold green]Goodbye![/bold green]",
            border_style="green",
            padding=(1, 2)
        ))


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Data Analytics Agent CLI - Ask questions about your CSV data in natural language",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           # Use OpenAI (default)
  %(prog)s --api-key sk-...          # Provide API key directly

Environment Variables:
  OPENAI_API_KEY      API key for OpenAI
"""
    )

    parser.add_argument(
        "--provider",
        choices=["openai"],
        default="openai",
        help="LLM provider to use (default: openai)"
    )
    parser.add_argument(
        "--api-key",
        help="API key for the LLM provider (uses environment variable if not provided)"
    )
    parser.add_argument(
        "--version",
        action="version",
        version="Data Analytics Agent CLI v1.0.0"
    )

    args = parser.parse_args()

    # Print header
    console = Console()
    console.print("[bold blue]Data Analytics Agent CLI[/bold blue]")
    console.print("[dim]Starting...[/dim]\n")

    try:
        cli = DataAnalyticsCLI(
            llm_provider=args.provider,
            api_key=args.api_key
        )
        cli.run()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[red]Error initializing CLI: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()