"""
Analytics Tools - Code Generation and Visualization

This module provides tools for generating and executing pandas analysis code
and creating data visualizations using matplotlib/seaborn. It includes:
- AnalyzeTool: Generates and executes pandas code for data analysis
- VisualizeTool: Creates visualizations using matplotlib/seaborn
"""

import logging
import time
import io
import base64
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt

from .base import BaseTool
from .dataset import _DATASET_STORE, get_dataset
from ..executor import SafeCodeExecutor
from ..prompts import (
    build_code_generation_prompt,
    build_visualization_prompt,
    build_error_retry_prompt,
    extract_code_from_response,
    format_dataset_schemas
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnalyzeTool(BaseTool):
    """
    Tool for performing data analysis by generating and executing pandas code.

    This tool accepts natural language queries about datasets, generates
    appropriate pandas code using an LLM, executes the code safely, and
    returns formatted results. It includes automatic error retry logic
    to recover from common code generation mistakes.

    Key features:
    - Natural language to pandas code translation
    - Safe code execution with timeout enforcement
    - Automatic error detection and retry
    - Support for multiple datasets and joins
    - Comprehensive result formatting

    Example:
        >>> tool = AnalyzeTool(llm_client, executor)
        >>> result = tool.execute(
        ...     query="What's the average amount of calories?",
        ...     datasets=["foods"]
        ... )
        >>> print(result["result"])
    """

    def __init__(self, llm_client, executor: SafeCodeExecutor):
        """
        Initialize the AnalyzeTool.

        Args:
            llm_client: OpenAI or Anthropic client for LLM API calls
            executor: SafeCodeExecutor instance for safe code execution
        """
        self.llm_client = llm_client
        self.executor = executor
        logger.info("AnalyzeTool initialized")

    @property
    def name(self) -> str:
        """Return the tool's unique identifier."""
        return "analyze"

    @property
    def description(self) -> str:
        """Return description of the tool's functionality."""
        return (
            "Execute a data analysis query using generated pandas code. "
            "Use this tool when the user asks analytical questions like "
            "'What is the average...', 'How many...', 'Show me the top...', etc. "
            "This tool generates and runs pandas code to answer the query."
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        """Define the JSON Schema for tool parameters."""
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Natural language query describing the analysis to perform. "
                        "Examples: 'average amount of calories', 'foods that are low in sodium', "
                        "'foods that are high in protein and fiber'"
                    )
                },
                "datasets": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "List of dataset aliases to use for the analysis. "
                        "Include all datasets needed for the query (e.g., for joins)."
                    )
                }
            },
            "required": ["query", "datasets"]
        }

    def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute a data analysis query by generating and running pandas code.

        This method orchestrates the full analysis pipeline:
        1. Validate requested datasets exist
        2. Format dataset schemas for the LLM
        3. Generate pandas code using the LLM
        4. Extract and clean the code
        5. Update executor with current datasets
        6. Execute the code safely
        7. Handle errors with automatic retry
        8. Format and return results

        Args:
            query (str): Natural language query describing the analysis
            datasets (list): List of dataset aliases to use

        Returns:
            Dict with execution results:
                On success:
                {
                    "status": "success",
                    "query": "the original query",
                    "generated_code": "the pandas code",
                    "result": "formatted output",
                    "execution_time_ms": 145,
                    "retry_attempted": False
                }

                On error:
                {
                    "status": "error",
                    "error_message": "detailed error description",
                    "generated_code": "the failed code if available",
                    "retry_attempted": True
                }
        """
        query = kwargs.get("query")
        dataset_aliases = kwargs.get("datasets", [])

        # Validate inputs
        if not query:
            return {
                "status": "error",
                "error_message": "query parameter is required"
            }

        if not dataset_aliases:
            return {
                "status": "error",
                "error_message": "datasets parameter is required and must contain at least one dataset alias"
            }

        # Step 1: Validate all requested datasets exist
        available_datasets = list(_DATASET_STORE.keys())
        missing_datasets = [alias for alias in dataset_aliases if alias not in _DATASET_STORE]

        if missing_datasets:
            available_str = ", ".join(available_datasets) if available_datasets else "none"
            missing_str = ", ".join(missing_datasets)
            return {
                "status": "error",
                "error_message": (
                    f"Dataset(s) not found: {missing_str}. "
                    f"Available datasets: {available_str}. "
                    f"Use load_csv to load datasets first."
                )
            }

        # Get the requested datasets
        datasets = {alias: get_dataset(alias) for alias in dataset_aliases}

        # Step 2: Format dataset schemas for the prompt
        dataset_schemas = format_dataset_schemas(datasets)

        # Step 3: Build code generation prompt
        prompt = build_code_generation_prompt(query, datasets)

        # Start timing
        start_time = time.time()

        try:
            # Step 4: Call LLM to generate code
            logger.info(f"Generating code for query: {query}")
            code_response = self._call_llm(prompt)

            # Step 5: Extract clean code from response
            generated_code = extract_code_from_response(code_response)

            if not generated_code:
                return {
                    "status": "error",
                    "error_message": "LLM did not generate valid code",
                    "retry_attempted": False
                }

            logger.info(f"Generated code:\n{generated_code}")

            # Step 6: Update executor with current datasets
            self.executor.update_datasets(datasets)

            # Step 7: Execute the code
            success, output, result = self.executor.execute(generated_code)

            # Calculate execution time
            execution_time_ms = int((time.time() - start_time) * 1000)

            # Step 8: Handle execution results
            if success:
                # Format result for display
                formatted_result = output if output else str(result) if result is not None else "No output"

                # Truncate long results
                if len(formatted_result) > 1000:
                    formatted_result = formatted_result[:1000] + "\n... (truncated)"

                logger.info("Code executed successfully")
                return {
                    "status": "success",
                    "query": query,
                    "generated_code": generated_code,
                    "result": formatted_result,
                    "execution_time_ms": execution_time_ms,
                    "retry_attempted": False
                }
            else:
                # Execution failed - attempt retry with error context
                error_message = output

                logger.warning(f"Code execution failed: {error_message}")
                logger.info("Attempting error recovery with retry...")

                # Build error retry prompt
                retry_prompt = build_error_retry_prompt(query, generated_code, error_message)

                # Call LLM to fix the code
                retry_code_response = self._call_llm(retry_prompt)
                retry_code = extract_code_from_response(retry_code_response)

                if not retry_code:
                    return {
                        "status": "error",
                        "error_message": f"Initial execution failed: {error_message}. Retry also failed to generate valid code.",
                        "generated_code": generated_code,
                        "retry_attempted": True
                    }

                logger.info(f"Retry code generated:\n{retry_code}")

                # Try executing the fixed code
                success_retry, output_retry, result_retry = self.executor.execute(retry_code)

                # Calculate total execution time including retry
                execution_time_ms = int((time.time() - start_time) * 1000)

                if success_retry:
                    # Retry succeeded
                    formatted_result = output_retry if output_retry else str(result_retry) if result_retry is not None else "No output"

                    # Truncate long results
                    if len(formatted_result) > 1000:
                        formatted_result = formatted_result[:1000] + "\n... (truncated)"

                    logger.info("Code executed successfully after retry")
                    return {
                        "status": "success",
                        "query": query,
                        "generated_code": retry_code,
                        "result": formatted_result,
                        "execution_time_ms": execution_time_ms,
                        "retry_attempted": True
                    }
                else:
                    # Both attempts failed
                    logger.error(f"Retry execution also failed: {output_retry}")
                    return {
                        "status": "error",
                        "error_message": (
                            f"Initial execution failed: {error_message}. "
                            f"Retry also failed: {output_retry}"
                        ),
                        "generated_code": retry_code,
                        "retry_attempted": True
                    }

        except Exception as e:
            logger.error(f"Unexpected error during analysis: {str(e)}")
            return {
                "status": "error",
                "error_message": f"Unexpected error: {str(e)}",
                "retry_attempted": False
            }

    def _call_llm(self, prompt: str) -> str:
        """
        Call LLM API to generate code.

        Supports both OpenAI and Anthropic clients with automatic detection.
        Uses temperature=0 for deterministic code generation.

        Args:
            prompt: The prompt to send to the LLM

        Returns:
            The LLM's response text

        Raises:
            RuntimeError: If the LLM API call fails
        """
        try:
            # Check if client is OpenAI
            if hasattr(self.llm_client, 'chat') and hasattr(self.llm_client.chat, 'completions'):
                logger.info("Using OpenAI client")
                response = self.llm_client.chat.completions.create(
                    model="o4-mini-2025-04-16",  # Use gpt-4o for better performance and JSON support
                    # model="gpt-4o-mini",  # Use gpt-4o for better performance and JSON support
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a Python pandas expert. Generate clean, executable pandas code."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    # temperature=0.01  # Deterministic for code generation
                )
                return response.choices[0].message.content

            # Check if client is Anthropic
            elif hasattr(self.llm_client, 'messages'):
                logger.info("Using Anthropic client")
                response = self.llm_client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=2000,
                    temperature=0,  # Deterministic for code generation
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )
                return response.content[0].text

            else:
                raise RuntimeError(
                    "Unsupported LLM client. Expected OpenAI or Anthropic client."
                )

        except Exception as e:
            logger.error(f"LLM API call failed: {str(e)}")
            raise RuntimeError(f"LLM API call failed: {str(e)}")


class VisualizeTool(BaseTool):
    """
    Tool for creating data visualizations by generating matplotlib/seaborn code.

    This tool accepts natural language descriptions of desired charts, generates
    appropriate visualization code using an LLM, executes the code safely, and
    returns the figure as a base64-encoded PNG image. It supports various chart
    types including line, bar, scatter, histogram, box, and pie charts.

    Key features:
    - Natural language to visualization code translation
    - Automatic chart type selection or user-specified type
    - Safe code execution with timeout enforcement
    - Automatic error detection and retry
    - Figure capture as base64-encoded PNG
    - Matplotlib state cleanup to prevent memory leaks

    Example:
        >>> tool = VisualizeTool(llm_client, executor)
        >>> result = tool.execute(
        ...     query="Show foods that are high in protein",
        ...     datasets=["foods"],
        ...     chart_type="bar"
        ... )
        >>> print(result["image_base64"])
    """

    def __init__(self, llm_client, executor: SafeCodeExecutor):
        """
        Initialize the VisualizeTool.

        Args:
            llm_client: OpenAI or Anthropic client for LLM API calls
            executor: SafeCodeExecutor instance for safe code execution
        """
        self.llm_client = llm_client
        self.executor = executor
        logger.info("VisualizeTool initialized")

    @property
    def name(self) -> str:
        """Return the tool's unique identifier."""
        return "visualize"

    @property
    def description(self) -> str:
        """Return description of the tool's functionality."""
        return (
            "Create a data visualization using generated matplotlib/seaborn code. "
            "Use this tool when the user wants to see a chart, graph, plot, or "
            "any visual representation of data. Supports line charts, bar charts, "
            "scatter plots, histograms, box plots, and more. "
            "Returns the visualization as a base64-encoded PNG image."
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        """Define the JSON Schema for tool parameters."""
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Natural language description of the desired visualization. "
                        "Examples: 'compare calcium content of foods that are high in vitamin D', 'bar chart of top 5 foods that are high in iron', "
                        "'scatter plot of calories vs protein'"
                    )
                },
                "datasets": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "List of dataset aliases to use for the visualization. "
                        "Include all datasets needed for the chart."
                    )
                },
                "chart_type": {
                    "type": "string",
                    "description": (
                        "Chart type hint: 'auto', 'line', 'bar', 'scatter', 'histogram', "
                        "'box', 'pie'. Use 'auto' to let the LLM choose. Default: 'auto'"
                    ),
                    "default": "auto"
                }
            },
            "required": ["query", "datasets"]
        }

    def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Generate and execute visualization code to create a chart.

        This method orchestrates the full visualization pipeline:
        1. Validate requested datasets exist
        2. Format dataset schemas for the LLM
        3. Generate visualization code using the LLM
        4. Extract and clean the code
        5. Update executor with current datasets
        6. Execute the code safely
        7. Handle errors with automatic retry
        8. Capture the figure and convert to base64 PNG
        9. Clean up matplotlib state
        10. Format and return results

        Args:
            query (str): Natural language description of desired chart
            datasets (list): List of dataset aliases to use
            chart_type (str, optional): Chart type hint (default: "auto")

        Returns:
            Dict with execution results:
                On success:
                {
                    "status": "success",
                    "query": "the original query",
                    "generated_code": "the matplotlib code",
                    "image_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
                    "chart_type": "line",
                    "execution_time_ms": 234,
                    "retry_attempted": False
                }

                On error:
                {
                    "status": "error",
                    "error_message": "detailed error description",
                    "generated_code": "the failed code if available",
                    "retry_attempted": True
                }
        """
        query = kwargs.get("query")
        dataset_aliases = kwargs.get("datasets", [])
        chart_type = kwargs.get("chart_type", "auto")

        # Validate inputs
        if not query:
            return {
                "status": "error",
                "error_message": "query parameter is required"
            }

        if not dataset_aliases:
            return {
                "status": "error",
                "error_message": "datasets parameter is required and must contain at least one dataset alias"
            }

        # Validate chart_type
        valid_chart_types = ["auto", "line", "bar", "scatter", "histogram", "box", "pie"]
        if chart_type not in valid_chart_types:
            return {
                "status": "error",
                "error_message": f"Invalid chart_type: {chart_type}. Must be one of {', '.join(valid_chart_types)}"
            }

        # Step 1: Validate all requested datasets exist
        available_datasets = list(_DATASET_STORE.keys())
        missing_datasets = [alias for alias in dataset_aliases if alias not in _DATASET_STORE]

        if missing_datasets:
            available_str = ", ".join(available_datasets) if available_datasets else "none"
            missing_str = ", ".join(missing_datasets)
            return {
                "status": "error",
                "error_message": (
                    f"Dataset(s) not found: {missing_str}. "
                    f"Available datasets: {available_str}. "
                    f"Use load_csv to load datasets first."
                )
            }

        # Get the requested datasets
        datasets = {alias: get_dataset(alias) for alias in dataset_aliases}

        # Step 2: Format dataset schemas for the prompt
        dataset_schemas = format_dataset_schemas(datasets)

        # Step 3: Build visualization prompt
        prompt = build_visualization_prompt(query, datasets, chart_type)

        # Start timing
        start_time = time.time()

        try:
            # Step 4: Call LLM to generate visualization code
            logger.info(f"Generating visualization code for query: {query}")
            code_response = self._call_llm(prompt)

            # Step 5: Extract clean code from response
            generated_code = extract_code_from_response(code_response)

            if not generated_code:
                return {
                    "status": "error",
                    "error_message": "LLM did not generate valid visualization code",
                    "retry_attempted": False
                }

            logger.info(f"Generated visualization code:\n{generated_code}")

            # Step 6: Update executor with current datasets
            self.executor.update_datasets(datasets)

            # Step 7: Execute the code
            success, output, result = self.executor.execute(generated_code)

            # Calculate execution time
            execution_time_ms = int((time.time() - start_time) * 1000)

            # Step 8: Handle execution results
            if not success:
                # Execution failed - attempt retry with error context
                error_message = output

                logger.warning(f"Visualization code execution failed: {error_message}")
                logger.info("Attempting error recovery with retry...")

                # Build error retry prompt
                retry_prompt = build_error_retry_prompt(query, generated_code, error_message)

                # Call LLM to fix the code
                retry_code_response = self._call_llm(retry_prompt)
                retry_code = extract_code_from_response(retry_code_response)

                if not retry_code:
                    plt.close('all')  # Clean up
                    return {
                        "status": "error",
                        "error_message": f"Initial execution failed: {error_message}. Retry also failed to generate valid code.",
                        "generated_code": generated_code,
                        "retry_attempted": True
                    }

                logger.info(f"Retry visualization code generated:\n{retry_code}")

                # Try executing the fixed code
                success, output, result = self.executor.execute(retry_code)

                # Calculate total execution time including retry
                execution_time_ms = int((time.time() - start_time) * 1000)

                if not success:
                    # Both attempts failed
                    plt.close('all')  # Clean up
                    logger.error(f"Retry execution also failed: {output}")
                    return {
                        "status": "error",
                        "error_message": (
                            f"Initial execution failed: {error_message}. "
                            f"Retry also failed: {output}"
                        ),
                        "generated_code": retry_code,
                        "retry_attempted": True
                    }

                # Retry succeeded, update generated_code to show the working version
                generated_code = retry_code
                retry_attempted = True
            else:
                retry_attempted = False

            # Step 9: Get the figure from executor namespace
            fig = self.executor.namespace.get('fig')

            if fig is None:
                plt.close('all')  # Clean up
                return {
                    "status": "error",
                    "error_message": (
                        "No figure was created. The visualization code must store "
                        "the matplotlib figure in a variable called 'fig'. "
                        "Example: fig, ax = plt.subplots()"
                    ),
                    "generated_code": generated_code,
                    "retry_attempted": retry_attempted
                }

            # Step 10: Detect chart type if auto was specified (before saving)
            detected_chart_type = chart_type
            if chart_type == "auto":
                detected_chart_type = self._detect_chart_type(generated_code)

            # Step 11: Save figure to file
            try:
                filepath = self._save_figure(fig, detected_chart_type)
                logger.info(f"Figure saved to: {filepath}")
            except Exception as e:
                logger.error(f"Failed to save figure to file: {str(e)}")
                plt.close('all')
                return {
                    "status": "error",
                    "error_message": f"Failed to save figure to file: {str(e)}",
                    "generated_code": generated_code,
                    "retry_attempted": retry_attempted
                }

            # Step 12: Convert figure to base64 PNG
            try:
                image_base64 = self._figure_to_base64(fig)
                logger.info("Figure successfully converted to base64 PNG")
            except Exception as e:
                logger.error(f"Failed to convert figure to image: {str(e)}")
                return {
                    "status": "error",
                    "error_message": f"Failed to convert figure to image: {str(e)}",
                    "generated_code": generated_code,
                    "retry_attempted": retry_attempted
                }
            finally:
                # Step 13: Clean up matplotlib state
                plt.close('all')

            logger.info(f"Visualization created successfully (type: {detected_chart_type})")

            # Create human-readable result message with filepath
            result_msg = f"Created {detected_chart_type} chart for query: '{query}'\n"
            result_msg += f"Saved to: {filepath}\n"
            result_msg += f"Execution time: {execution_time_ms}ms"
            if retry_attempted:
                result_msg += " (recovered from initial error)"

            return {
                "status": "success",
                "result": result_msg,  # ✅ Added for BaseTool contract
                "data": {
                    "query": query,
                    "generated_code": generated_code,
                    "image_base64": image_base64,
                    "filepath": filepath,  # ✅ Added filepath
                    "chart_type": detected_chart_type,
                    "execution_time_ms": execution_time_ms,
                    "retry_attempted": retry_attempted
                }
            }

        except Exception as e:
            # Always clean up matplotlib state
            plt.close('all')
            logger.error(f"Unexpected error during visualization: {str(e)}")
            return {
                "status": "error",
                "error_message": f"Unexpected error: {str(e)}",
                "retry_attempted": False
            }

    def _call_llm(self, prompt: str) -> str:
        """
        Call LLM API to generate visualization code.

        Supports both OpenAI and Anthropic clients with automatic detection.
        Uses temperature=0 for deterministic code generation.

        Args:
            prompt: The prompt to send to the LLM

        Returns:
            The LLM's response text

        Raises:
            RuntimeError: If the LLM API call fails
        """
        try:
            # Check if client is OpenAI
            if hasattr(self.llm_client, 'chat') and hasattr(self.llm_client.chat, 'completions'):
                logger.info("Using OpenAI client for visualization")
                response = self.llm_client.chat.completions.create(
                    model="gpt-4o",  # Use gpt-4o for better performance and JSON support
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a Python data visualization expert specializing in matplotlib and seaborn. Generate clean, executable visualization code."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0  # Deterministic for code generation
                )
                return response.choices[0].message.content

            # Check if client is Anthropic
            elif hasattr(self.llm_client, 'messages'):
                logger.info("Using Anthropic client for visualization")
                response = self.llm_client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=2000,
                    temperature=0,  # Deterministic for code generation
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )
                return response.content[0].text

            else:
                raise RuntimeError(
                    "Unsupported LLM client. Expected OpenAI or Anthropic client."
                )

        except Exception as e:
            logger.error(f"LLM API call failed: {str(e)}")
            raise RuntimeError(f"LLM API call failed: {str(e)}")

    def _figure_to_base64(self, fig) -> str:
        """
        Convert matplotlib figure to base64 PNG string.

        This method saves the figure to an in-memory buffer, reads the bytes,
        encodes them as base64, and returns the string. The figure is saved
        with tight bounding box and 100 DPI for good quality.

        Args:
            fig: Matplotlib figure object

        Returns:
            Base64-encoded PNG image string

        Raises:
            Exception: If figure conversion fails
        """
        buf = io.BytesIO()
        try:
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
            buf.seek(0)
            img_bytes = buf.read()
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')
            return img_base64
        finally:
            buf.close()

    def _save_figure(self, fig, chart_type: str = "chart") -> str:
        """
        Save matplotlib figure to file in outputs directory.

        Creates outputs/ directory if it doesn't exist and saves the figure
        with a timestamped filename for easy identification.

        Args:
            fig: Matplotlib figure object
            chart_type: Type of chart for filename (e.g., "bar", "scatter")

        Returns:
            Absolute path to saved file

        Raises:
            Exception: If figure save fails
        """
        # Create outputs directory if it doesn't exist
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)

        # Generate timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{chart_type}_{timestamp}.png"
        filepath = output_dir / filename

        # Save figure
        fig.savefig(filepath, format='png', bbox_inches='tight', dpi=100)

        # Return absolute path for clarity
        return str(filepath.absolute())

    def _detect_chart_type(self, code: str) -> str:
        """
        Detect the chart type from generated visualization code.

        Analyzes the code to identify which matplotlib/seaborn plotting
        function was used, in order to classify the chart type. This is
        useful when chart_type="auto" was specified.

        Args:
            code: Generated visualization code

        Returns:
            Detected chart type as a string (e.g., "line", "bar", "scatter")
        """
        code_lower = code.lower()

        # Check for different plot types in order of specificity
        # More specific patterns first to avoid false positives
        if 'scatter' in code_lower or 'ax.scatter' in code_lower or 'plt.scatter' in code_lower:
            return 'scatter'
        elif 'hist' in code_lower or '.histogram' in code_lower or 'plt.hist' in code_lower:
            return 'histogram'
        elif 'boxplot' in code_lower or 'box_plot' in code_lower or '.box(' in code_lower:
            return 'box'
        elif 'pie' in code_lower or 'plt.pie' in code_lower:
            return 'pie'
        elif 'barh' in code_lower or "kind='barh'" in code_lower:
            return 'horizontal_bar'
        elif 'bar' in code_lower or "kind='bar'" in code_lower or 'plt.bar' in code_lower:
            return 'bar'
        elif 'plot' in code_lower or 'line' in code_lower or 'ax.plot' in code_lower or 'plt.plot' in code_lower:
            return 'line'
        elif 'heatmap' in code_lower or 'sns.heatmap' in code_lower:
            return 'heatmap'
        else:
            return 'unknown'