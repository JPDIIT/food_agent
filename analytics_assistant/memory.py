"""
Conversation memory management for the ReAct agent.

This module provides memory capabilities for tracking conversation history,
loaded datasets, and created variables across multi-turn interactions.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ConversationTurn:
    """Represents a single turn in the conversation using ReAct pattern."""

    query: str
    thought: str
    action: str
    action_params: Dict[str, Any]
    result: str
    timestamp: datetime = field(default_factory=datetime.now)

    def format(self) -> str:
        """Format the turn for display in conversation context."""
        params_str = ", ".join(f"{k}={repr(v)}" for k, v in self.action_params.items())
        return (
            f"Query: \"{self.query}\"\n"
            f"Thought: \"{self.thought}\"\n"
            f"Action: {self.action}({params_str})\n"
            f"Result: {self.result}"
        )


@dataclass
class DatasetInfo:
    """Information about a loaded dataset."""

    alias: str
    rows: int
    columns: int
    memory_mb: float
    loaded_at: datetime = field(default_factory=datetime.now)

    def format(self) -> str:
        """Format dataset info for display."""
        return f"{self.alias}: {self.rows} rows, {self.columns} cols, {self.memory_mb:.2f} MB"


class ConversationMemory:
    """
    Manages conversation state and context for the ReAct agent.

    Tracks:
    - Conversation history using ReAct pattern (query, thought, action, result)
    - Loaded datasets with metadata
    - Variables created during code execution

    This enables the agent to maintain context across multiple turns and
    reference previous actions, datasets, and variables.
    """

    def __init__(self):
        """Initialize empty conversation memory."""
        self.turns: List[ConversationTurn] = []
        self.loaded_datasets: Dict[str, DatasetInfo] = {}
        self.variables: Dict[str, str] = {}

    def add_turn(
        self,
        query: str,
        thought: str,
        action: str,
        action_params: Dict[str, Any],
        result: str
    ) -> None:
        """
        Add a new conversation turn to memory.

        Args:
            query: The user's natural language query
            thought: The agent's reasoning about what to do
            action: The action taken (e.g., "load_data", "analyze")
            action_params: Parameters passed to the action
            result: The result of executing the action
        """
        turn = ConversationTurn(
            query=query,
            thought=thought,
            action=action,
            action_params=action_params,
            result=result
        )
        self.turns.append(turn)

    def get_recent_context(self, last_n: int = 5) -> str:
        """
        Format recent conversation turns for prompt injection.

        Args:
            last_n: Number of recent turns to include (default: 5)

        Returns:
            Formatted string containing recent conversation history
        """
        if not self.turns:
            return "=== Recent Conversation ===\n\nNo previous conversation."

        recent_turns = self.turns[-last_n:]
        context_lines = ["=== Recent Conversation ===\n"]

        for i, turn in enumerate(recent_turns, 1):
            context_lines.append(f"Turn {i}:")
            context_lines.append(turn.format())
            context_lines.append("")  # Blank line between turns

        return "\n".join(context_lines)

    def add_dataset(
        self,
        alias: str,
        rows: int,
        columns: int,
        memory_mb: float
    ) -> None:
        """
        Track a newly loaded dataset.

        Args:
            alias: Dataset alias (variable name)
            rows: Number of rows in the dataset
            columns: Number of columns in the dataset
            memory_mb: Memory usage in megabytes
        """
        dataset_info = DatasetInfo(
            alias=alias,
            rows=rows,
            columns=columns,
            memory_mb=memory_mb
        )
        self.loaded_datasets[alias] = dataset_info

    def remove_dataset(self, alias: str) -> bool:
        """
        Remove a dataset from tracking.

        Args:
            alias: Dataset alias to remove

        Returns:
            True if dataset was removed, False if it didn't exist
        """
        if alias in self.loaded_datasets:
            del self.loaded_datasets[alias]
            return True
        return False

    def get_loaded_datasets(self) -> List[str]:
        """
        Get list of currently loaded dataset aliases.

        Returns:
            List of dataset aliases (variable names)
        """
        return list(self.loaded_datasets.keys())

    def get_datasets_summary(self) -> str:
        """
        Get a formatted summary of all loaded datasets.

        Returns:
            Formatted string describing loaded datasets
        """
        if not self.loaded_datasets:
            return "No datasets currently loaded."

        summary_lines = ["Loaded Datasets:"]
        for dataset in self.loaded_datasets.values():
            summary_lines.append(f"  - {dataset.format()}")

        return "\n".join(summary_lines)

    def track_variable(self, var_name: str, var_type: str) -> None:
        """
        Track a variable created during code execution.

        Args:
            var_name: Name of the variable
            var_type: Type of the variable (e.g., "DataFrame", "Series", "int")
        """
        self.variables[var_name] = var_type

    def get_variables(self) -> Dict[str, str]:
        """
        Get all tracked variables.

        Returns:
            Dictionary mapping variable names to their types
        """
        return self.variables.copy()

    def get_variables_summary(self) -> str:
        """
        Get a formatted summary of all tracked variables.

        Returns:
            Formatted string describing tracked variables
        """
        if not self.variables:
            return "No variables currently tracked."

        summary_lines = ["Tracked Variables:"]
        for var_name, var_type in self.variables.items():
            summary_lines.append(f"  - {var_name}: {var_type}")

        return "\n".join(summary_lines)

    def get_full_context(self, last_n_turns: int = 5) -> str:
        """
        Get complete context including conversation, datasets, and variables.

        Args:
            last_n_turns: Number of recent turns to include

        Returns:
            Comprehensive context string for prompt injection
        """
        sections = []

        # Recent conversation
        sections.append(self.get_recent_context(last_n_turns))

        # Loaded datasets
        if self.loaded_datasets:
            sections.append(f"\n{self.get_datasets_summary()}")

        # Tracked variables
        if self.variables:
            sections.append(f"\n{self.get_variables_summary()}")

        return "\n".join(sections)

    def clear(self) -> None:
        """Reset all memory (conversation, datasets, variables)."""
        self.turns.clear()
        self.loaded_datasets.clear()
        self.variables.clear()

    def get_turn_count(self) -> int:
        """Get the total number of conversation turns."""
        return len(self.turns)

    def get_last_turn(self) -> Optional[ConversationTurn]:
        """
        Get the most recent conversation turn.

        Returns:
            The last ConversationTurn, or None if no turns exist
        """
        return self.turns[-1] if self.turns else None