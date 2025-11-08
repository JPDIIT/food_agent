"""
Dataset Management Tools

This module provides tools for loading, listing, and inspecting datasets.
All datasets are stored in a global in-memory store for the agent to access.
"""

from typing import Any, Dict
import pandas as pd
from pathlib import Path

from .base import BaseTool


# Global storage for loaded datasets
_DATASET_STORE: Dict[str, pd.DataFrame] = {}


class LoadCSVTool(BaseTool):
    """
    Tool for loading CSV files into memory with a user-friendly alias.

    This tool reads a CSV file, stores it in the global dataset store,
    and returns detailed information about the loaded dataset including
    shape, columns, memory usage, and a sample of the data.
    """

    @property
    def name(self) -> str:
        """Return the tool's unique identifier."""
        return "load_csv"

    @property
    def description(self) -> str:
        """Return description of the tool's functionality."""
        return (
            "Load a CSV file into memory with a friendly alias. "
            "Use this tool when the user wants to load or import a dataset. "
            "Returns dataset information including shape, columns, and a data sample."
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        """Define the JSON Schema for tool parameters."""
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the CSV file to load (absolute or relative)"
                },
                "alias": {
                    "type": "string",
                    "description": "Short, memorable name to reference this dataset (e.g., 'foods', 'daily value')"
                }
            },
            "required": ["file_path", "alias"]
        }

    def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Load a CSV file into the dataset store.

        Args:
            file_path (str): Path to the CSV file
            alias (str): Alias name for the dataset

        Returns:
            Dict with status, dataset info on success, or error message on failure
        """
        file_path = kwargs.get("file_path")
        alias = kwargs.get("alias")

        # Validate inputs
        if not file_path:
            return {
                "status": "error",
                "result": None,
                "error_message": "file_path parameter is required"
            }

        if not alias:
            return {
                "status": "error",
                "result": None,
                "error_message": "alias parameter is required"
            }

        # Check if alias already exists
        if alias in _DATASET_STORE:
            return {
                "status": "error",
                "result": None,
                "error_message": (
                    f"Alias '{alias}' is already in use. "
                    f"Please choose a different alias or use list_datasets to see existing datasets."
                )
            }

        try:
            # Check if file exists
            path = Path(file_path)
            if not path.exists():
                return {
                    "status": "error",
                    "result": None,
                    "error_message": f"File not found: {file_path}"
                }

            # Load the CSV file
            df = pd.read_csv(file_path)

            # Check if CSV is empty
            if df.empty:
                return {
                    "status": "error",
                    "result": None,
                    "error_message": f"CSV file is empty: {file_path}"
                }

            # Calculate memory usage in MB
            memory_bytes = df.memory_usage(deep=True).sum()
            memory_mb = round(memory_bytes / (1024 * 1024), 2)

            # Get sample rows (first 3)
            sample = df.head(3).to_string(index=True)

            # Store in global dataset store
            _DATASET_STORE[alias] = df

            # Format result message
            result_msg = f"Successfully loaded '{alias}' from {file_path}\n"
            result_msg += f"- {len(df)} rows, {len(df.columns)} columns\n"
            result_msg += f"- Columns: {', '.join(df.columns.tolist())}\n"
            result_msg += f"- Memory: {memory_mb:.2f} MB\n\n"
            result_msg += f"Sample (first 3 rows):\n{sample}"

            # Return success response with detailed info
            return {
                "status": "success",
                "result": result_msg,
                "data": {
                    "alias": alias,
                    "rows": len(df),
                    "columns": len(df.columns),
                    "column_names": df.columns.tolist(),
                    "memory_mb": memory_mb
                }
            }

        except pd.errors.EmptyDataError:
            return {
                "status": "error",
                "result": None,
                "error_message": f"CSV file is empty or invalid: {file_path}"
            }
        except pd.errors.ParserError as e:
            return {
                "status": "error",
                "result": None,
                "error_message": f"Invalid CSV format: {str(e)}"
            }
        except Exception as e:
            return {
                "status": "error",
                "result": None,
                "error_message": f"Failed to load CSV: {str(e)}"
            }


class ListDatasetsTool(BaseTool):
    """
    Tool for listing all currently loaded datasets.

    This tool provides an overview of all datasets in the global store,
    including their aliases, shapes, and memory usage.
    """

    @property
    def name(self) -> str:
        """Return the tool's unique identifier."""
        return "list_datasets"

    @property
    def description(self) -> str:
        """Return description of the tool's functionality."""
        return (
            "List all currently loaded datasets. "
            "Use this tool when the user wants to see what datasets are available "
            "or asks 'what data do I have' or 'show me the datasets'."
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        """Define the JSON Schema for tool parameters."""
        return {
            "type": "object",
            "properties": {},
            "required": []
        }

    def execute(self, **kwargs) -> Dict[str, Any]:
        """
        List all loaded datasets.

        Returns:
            Dict with status and list of datasets with their metadata
        """
        try:
            # Check if any datasets are loaded
            if not _DATASET_STORE:
                return {
                    "status": "success",
                    "result": "No datasets currently loaded. Use load_csv to load a dataset.",
                    "data": {
                        "datasets": [],
                        "total_memory_mb": 0.0,
                        "count": 0
                    }
                }

            # Collect information about each dataset
            datasets = []
            total_memory = 0.0

            for alias, df in _DATASET_STORE.items():
                # Calculate memory usage
                memory_bytes = df.memory_usage(deep=True).sum()
                memory_mb = round(memory_bytes / (1024 * 1024), 2)
                total_memory += memory_mb

                datasets.append({
                    "alias": alias,
                    "rows": len(df),
                    "columns": len(df.columns),
                    "memory_mb": memory_mb
                })

            # Format result message
            result_msg = f"Loaded datasets ({len(datasets)} total, {round(total_memory, 2)} MB):\n"
            for ds in datasets:
                result_msg += f"- '{ds['alias']}': {ds['rows']} rows, {ds['columns']} columns ({ds['memory_mb']} MB)\n"

            return {
                "status": "success",
                "result": result_msg.strip(),
                "data": {
                    "datasets": datasets,
                    "total_memory_mb": round(total_memory, 2),
                    "count": len(datasets)
                }
            }

        except Exception as e:
            return {
                "status": "error",
                "result": None,
                "error_message": f"Failed to list datasets: {str(e)}"
            }


class InspectDatasetTool(BaseTool):
    """
    Tool for inspecting detailed information about a specific dataset.

    This tool provides comprehensive information about a dataset including
    schema (column names, types, null counts), sample rows, and optional
    statistical summaries for numeric columns.
    """

    @property
    def name(self) -> str:
        """Return the tool's unique identifier."""
        return "inspect_dataset"

    @property
    def description(self) -> str:
        """Return description of the tool's functionality."""
        return (
            "Get detailed information about a specific dataset including schema, "
            "data types, null counts, sample rows, and optional statistics. "
            "Use this when the user wants to explore or understand a dataset's structure."
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        """Define the JSON Schema for tool parameters."""
        return {
            "type": "object",
            "properties": {
                "alias": {
                    "type": "string",
                    "description": "The alias of the dataset to inspect"
                },
                "include_stats": {
                    "type": "boolean",
                    "description": "Whether to include statistical summaries for numeric columns (default: True)"
                }
            },
            "required": ["alias"]
        }

    def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Inspect a dataset and return detailed information.

        Args:
            alias (str): Dataset alias to inspect
            include_stats (bool): Whether to include column statistics (default: True)

        Returns:
            Dict with status and detailed dataset information, or error message
        """
        alias = kwargs.get("alias")
        include_stats = kwargs.get("include_stats", True)

        # Validate alias parameter
        if not alias:
            return {
                "status": "error",
                "result": None,
                "error_message": "alias parameter is required"
            }

        # Check if dataset exists
        if alias not in _DATASET_STORE:
            available = list(_DATASET_STORE.keys())
            available_str = ", ".join(available) if available else "none"
            return {
                "status": "error",
                "result": None,
                "error_message": (
                    f"Dataset '{alias}' not found. "
                    f"Available datasets: {available_str}"
                )
            }

        try:
            df = _DATASET_STORE[alias]

            # Build schema information
            schema = []
            null_counts = df.isnull().sum()
            total_rows = len(df)

            for col in df.columns:
                null_count = int(null_counts[col])
                null_percentage = round((null_count / total_rows) * 100, 2) if total_rows > 0 else 0.0

                schema.append({
                    "name": col,
                    "dtype": str(df[col].dtype),
                    "null_count": null_count,
                    "null_percentage": null_percentage
                })

            # Get sample rows (first 5)
            sample_rows = df.head(5).to_string(index=True)

            # Build detailed inspection data
            inspection_data = {
                "alias": alias,
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": list(df.columns),
                "schema": schema,
                "sample_rows": sample_rows
            }

            # Add statistics if requested
            if include_stats:
                stats = {}
                numeric_cols = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns

                if len(numeric_cols) > 0:
                    desc = df[numeric_cols].describe()

                    for col in numeric_cols:
                        # Extract statistics and round to 2 decimal places
                        col_stats = {
                            "count": int(desc.loc['count', col]),
                            "mean": round(desc.loc['mean', col], 2),
                            "std": round(desc.loc['std', col], 2),
                            "min": round(desc.loc['min', col], 2),
                            "25%": round(desc.loc['25%', col], 2),
                            "median": round(desc.loc['50%', col], 2),
                            "75%": round(desc.loc['75%', col], 2),
                            "max": round(desc.loc['max', col], 2)
                        }
                        stats[col] = col_stats

                inspection_data["stats"] = stats if stats else None

            # Format as human-readable string for the "result" field
            result_str = f"Dataset '{alias}' inspection:\n"
            result_str += f"- {len(df)} rows, {len(df.columns)} columns\n"
            result_str += f"- Columns: {', '.join(df.columns)}\n\n"
            result_str += "Schema:\n"
            for col_info in schema:
                result_str += f"  • {col_info['name']} ({col_info['dtype']})"
                if col_info['null_count'] > 0:
                    result_str += f" - {col_info['null_count']} nulls ({col_info['null_percentage']}%)"
                result_str += "\n"
            result_str += f"\nSample rows:\n{sample_rows}"

            # Return in BaseTool standard format
            return {
                "status": "success",
                "result": result_str,
                "data": inspection_data  # Full structured data for programmatic access
            }

        except Exception as e:
            return {
                "status": "error",
                "result": None,
                "error_message": f"Failed to inspect dataset: {str(e)}"
            }


def clear_datasets() -> None:
    """
    Clear all datasets from the global store.

    This is a utility function for testing or resetting the dataset store.
    Not exposed as a tool to the agent by default.
    """
    global _DATASET_STORE
    _DATASET_STORE.clear()


def get_dataset(alias: str) -> pd.DataFrame:
    """
    Get a dataset from the store by alias.

    Args:
        alias: The dataset alias

    Returns:
        The pandas DataFrame

    Raises:
        KeyError: If the alias doesn't exist in the store
    """
    if alias not in _DATASET_STORE:
        raise KeyError(f"Dataset '{alias}' not found in store")
    return _DATASET_STORE[alias]