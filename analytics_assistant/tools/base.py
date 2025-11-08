"""
Base Tool Abstract Class

This module defines the abstract base class for all tools in the agent system.
Tools are discrete capabilities that the agent can invoke to perform specific tasks.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseTool(ABC):
    """
    Abstract base class for all agent tools.

    All tools must inherit from this class and implement the required abstract
    properties and methods. This ensures a consistent interface for tool execution
    and metadata access across the agent system.

    Tools should be stateless where possible, with all necessary context passed
    through the execute method's kwargs parameter.

    Example:
        ```python
        class LoadCSVTool(BaseTool):
            @property
            def name(self) -> str:
                return "load_csv"

            @property
            def description(self) -> str:
                return "Load a CSV file into memory"

            @property
            def parameters_schema(self) -> Dict[str, Any]:
                return {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                        "alias": {"type": "string"}
                    },
                    "required": ["file_path", "alias"]
                }

            def execute(self, **kwargs) -> Dict[str, Any]:
                file_path = kwargs.get("file_path")
                alias = kwargs.get("alias")
                # Implementation here
                return {
                    "status": "success",
                    "result": f"Loaded {file_path} as {alias}"
                }
        ```
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        The unique identifier for this tool.

        Should be lowercase with underscores (snake_case) and descriptive
        of the tool's primary function.

        Returns:
            str: The tool's unique name identifier
        """
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """
        A clear, concise description of what this tool does.

        This description is used by the agent to determine when to use this tool.
        It should explain the tool's purpose and when it should be invoked.

        Returns:
            str: Human-readable description of the tool's functionality
        """
        pass

    @property
    @abstractmethod
    def parameters_schema(self) -> Dict[str, Any]:
        """
        JSON Schema defining the parameters this tool accepts.

        The schema should follow JSON Schema specification (draft-07 or later).
        It defines the structure, types, and validation rules for tool parameters.

        Returns:
            Dict[str, Any]: JSON Schema object with properties and requirements

        Example:
            ```python
            {
                "type": "object",
                "properties": {
                    "param1": {"type": "string", "description": "First parameter"},
                    "param2": {"type": "integer", "description": "Second parameter"}
                },
                "required": ["param1"]
            }
            ```
        """
        pass

    @abstractmethod
    def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the tool's primary function with the given parameters.

        This method contains the core logic of the tool. It should handle all
        parameter validation, error cases, and return a standardized response.

        Args:
            **kwargs: Tool-specific parameters as defined in parameters_schema

        Returns:
            Dict[str, Any]: Execution result with the following structure:
                - status (str): Either "success" or "error"
                - result (Any): The actual result on success (type varies by tool)
                - error_message (str, optional): Error description if status is "error"

        Example:
            Success response:
            ```python
            {
                "status": "success",
                "result": {"data": "some value"}
            }
            ```

            Error response:
            ```python
            {
                "status": "error",
                "result": None,
                "error_message": "Failed to process: invalid input"
            }
            ```
        """
        pass

    def __repr__(self) -> str:
        """
        String representation of the tool for debugging.

        Returns:
            str: Formatted string with tool name and description
        """
        return f"{self.__class__.__name__}(name='{self.name}')"