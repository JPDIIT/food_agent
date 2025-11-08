"""
Safe code execution engine for data analysis agent.

This module provides a secure sandbox for executing dynamically generated Python code,
specifically designed for data analysis with pandas, numpy, matplotlib, and seaborn.

Security Features:
- AST validation to block dangerous operations (file I/O, network, subprocess, etc.)
- Timeout enforcement (5 seconds maximum execution time)
- Restricted namespace with whitelisted libraries only
- Isolated but persistent namespace for variable continuity
- Safe builtins subset (no open, exec, eval, __import__)

Example:
    >>> import pandas as pd
    >>> executor = SafeCodeExecutor({"foods": pd.DataFrame({"calcium": [100, 200]})})
    >>> success, output, result = executor.execute("print(foods['calcium'].mean())")
    >>> print(output)
    150.0
"""

import ast
import io
import logging
import signal
import sys
from contextlib import contextmanager, redirect_stdout
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from RestrictedPython import compile_restricted_exec
from RestrictedPython.Eval import default_guarded_getiter
from RestrictedPython.Guards import (
    guarded_iter_unpack_sequence,
    guarded_unpack_sequence,
    safe_builtins,
    safe_globals,
)
from RestrictedPython.PrintCollector import PrintCollector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TimeoutException(Exception):
    """Raised when code execution exceeds the timeout limit."""
    pass


class SafeCodeExecutor:
    """
    Secure code execution engine for data analysis operations.

    This class provides a sandboxed environment for executing dynamically generated
    Python code with strict security controls and timeout enforcement.

    Attributes:
        datasets: Dictionary mapping dataset aliases to pandas DataFrames
        namespace: Persistent execution namespace containing datasets and libraries
        timeout_seconds: Maximum execution time (default: 5 seconds)
    """

    # Dangerous operations to block
    BLOCKED_OPERATIONS = {
        'open', 'file', 'input', 'raw_input', 'execfile',
        'compile', 'reload', '__import__', 'eval', 'exec',
        'system', 'popen', 'spawn', 'subprocess'
    }

    # Dangerous modules to block
    BLOCKED_MODULES = {
        'os', 'sys', 'subprocess', 'socket', 'urllib', 'requests',
        'http', 'ftplib', 'smtplib', 'telnetlib', 'pickle',
        'shelve', 'dbm', 'gdbm', 'importlib', 'builtins', '__builtin__'
    }

    # Dangerous attributes to block
    BLOCKED_ATTRIBUTES = {
        '__code__', '__globals__', '__builtins__', '__loader__',
        '__spec__', '__package__', '__file__', '__name__',
        '__dict__', '__class__', '__bases__', '__subclasses__'
    }

    def __init__(self, datasets: dict[str, pd.DataFrame], timeout_seconds: int = 5):
        """
        Initialize the safe code executor.

        Args:
            datasets: Dictionary mapping dataset aliases to DataFrames
            timeout_seconds: Maximum execution time in seconds (default: 5)
        """
        self.datasets = datasets
        self.timeout_seconds = timeout_seconds
        self.namespace = self._create_safe_namespace()
        logger.info(f"SafeCodeExecutor initialized with {len(datasets)} dataset(s)")

    def execute(self, code: str) -> tuple[bool, str, Any]:
        """
        Execute Python code in a secure, isolated environment.

        This method:
        1. Validates code for dangerous operations
        2. Compiles with RestrictedPython
        3. Executes with timeout enforcement
        4. Captures stdout and results
        5. Formats output appropriately

        Args:
            code: Python code string to execute

        Returns:
            Tuple of (success: bool, output: str, result: Any)
            - success: True if execution succeeded without errors
            - output: Captured stdout or error message
            - result: Last expression result (if any)

        Example:
            >>> success, output, result = executor.execute("df.head()")
            >>> if success:
            ...     print(output)
        """
        # Step 1: Validate code for dangerous operations
        is_valid, error_msg = self._validate_code(code)
        if not is_valid:
            logger.warning(f"Code validation failed: {error_msg}")
            return False, f"Security Error: {error_msg}", None

        # Step 2: Compile with RestrictedPython
        try:
            byte_code = compile_restricted_exec(code, filename='<user_code>')

            if byte_code.errors:
                error_msg = "\n".join(byte_code.errors)
                logger.warning(f"Compilation errors: {error_msg}")
                return False, f"Compilation Error: {error_msg}", None

        except SyntaxError as e:
            logger.warning(f"Syntax error: {e}")
            return False, f"Syntax Error: {str(e)}", None

        # Step 3: Execute with timeout and print capture
        # Reset print collector for each execution
        # Clear any previous _print instance from the namespace
        self.namespace.pop('_print', None)
        # Set up print collector (RestrictedPython expects the class, not an instance)
        self.namespace['_print_'] = PrintCollector

        try:
            with self._timeout(self.timeout_seconds):
                # Execute the code
                exec(byte_code.code, self.namespace)

            # Get captured output from print statements
            # After execution, _print contains the PrintCollector instance
            print_collector = self.namespace.get('_print')
            output = print_collector() if print_collector else ""

            # CRITICAL FIX: Extract 'result' variable from namespace
            # The generated code stores its result in a variable called 'result'
            # We must extract it from the namespace, not try to eval the code
            result = self.namespace.get('result')

            # Format the result if it exists and nothing was printed
            if result is not None and not output:
                output = self._format_result(result)
            elif result is not None and output:
                # If both print output and result exist, include both
                formatted_result = self._format_result(result)
                output = f"{output}\n\nResult:\n{formatted_result}"

            logger.info(f"Code executed successfully. Result type: {type(result).__name__}")
            return True, output.strip() if output else "", result

        except TimeoutException:
            error_msg = f"Execution timed out (>{self.timeout_seconds}s)"
            logger.error(error_msg)
            return False, error_msg, None

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            logger.error(f"Execution error: {error_msg}")
            return False, f"Runtime Error: {error_msg}", None

    def update_datasets(self, datasets: dict[str, pd.DataFrame]) -> None:
        """
        Update the available datasets in the namespace.

        This method allows adding or replacing datasets while preserving
        other namespace variables and computed results.

        Args:
            datasets: Dictionary mapping dataset aliases to DataFrames
        """
        self.datasets = datasets
        # Update only the dataset entries in the namespace
        for alias, df in datasets.items():
            self.namespace[alias] = df
        logger.info(f"Updated {len(datasets)} dataset(s) in namespace")

    def _create_safe_namespace(self) -> dict:
        """
        Create a restricted namespace with whitelisted libraries and safe builtins.

        The namespace includes:
        - All datasets from self.datasets
        - pandas (as pd), numpy (as np), matplotlib.pyplot (as plt), seaborn (as sns)
        - Safe subset of builtins (no file I/O, no imports, no exec/eval)

        Returns:
            Dictionary containing the safe execution namespace
        """
        # Start with safe builtins from RestrictedPython
        namespace = {
            '__builtins__': self._get_safe_builtins(),
            '_getattr_': self._safe_getattr,
            '_getitem_': self._safe_getitem,
            '_write_': self._safe_write,
            '_getiter_': default_guarded_getiter,
            '_unpack_sequence_': guarded_unpack_sequence,
            '_iter_unpack_sequence_': guarded_iter_unpack_sequence,
            # _print_ is set per execution in execute()
        }

        # Add whitelisted libraries
        namespace.update({
            'pd': pd,
            'np': np,
            'plt': plt,
            'sns': sns,
        })

        # Add datasets
        namespace.update(self.datasets)

        return namespace

    def _get_safe_builtins(self) -> dict:
        """
        Create a restricted set of safe builtins.

        Removes dangerous functions like open, exec, eval, __import__, etc.

        Returns:
            Dictionary of safe builtin functions
        """
        # Start with RestrictedPython's safe builtins
        safe = safe_builtins.copy()

        # Remove any remaining dangerous operations
        for blocked in self.BLOCKED_OPERATIONS:
            safe.pop(blocked, None)

        return safe

    def _validate_code(self, code: str) -> tuple[bool, str]:
        """
        Validate code using AST parsing to detect dangerous operations.

        Checks for:
        - Dangerous function calls (open, exec, eval, __import__, etc.)
        - Blocked module imports (os, sys, subprocess, etc.)
        - Dangerous attribute access (__code__, __globals__, etc.)

        Args:
            code: Python code string to validate

        Returns:
            Tuple of (is_valid: bool, error_message: str)
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"Syntax error: {str(e)}"

        for node in ast.walk(tree):
            # Check for dangerous function calls
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in self.BLOCKED_OPERATIONS:
                        return False, f"Blocked operation: {node.func.id}"

            # Check for blocked imports
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                module_name = None
                if isinstance(node, ast.Import):
                    module_name = node.names[0].name
                elif isinstance(node, ast.ImportFrom):
                    module_name = node.module

                if module_name and module_name.split('.')[0] in self.BLOCKED_MODULES:
                    return False, f"Blocked module import: {module_name}"

            # Check for dangerous attribute access
            if isinstance(node, ast.Attribute):
                if node.attr in self.BLOCKED_ATTRIBUTES:
                    return False, f"Blocked attribute access: {node.attr}"

        return True, ""

    def _format_result(self, result: Any) -> str:
        """
        Format the result for display.

        Handles special formatting for:
        - pandas DataFrames (truncated to 20 rows)
        - pandas Series
        - numpy arrays
        - Other objects (converted to string)

        Args:
            result: The result object to format

        Returns:
            Formatted string representation
        """
        if isinstance(result, pd.DataFrame):
            return result.to_string(max_rows=20)
        elif isinstance(result, pd.Series):
            return result.to_string(max_rows=20)
        elif isinstance(result, np.ndarray):
            return str(result)
        else:
            return str(result)

    @contextmanager
    def _timeout(self, seconds: int):
        """
        Context manager for enforcing execution timeout.

        Uses SIGALRM on Unix systems to interrupt long-running code.

        Args:
            seconds: Maximum execution time in seconds

        Raises:
            TimeoutException: If execution exceeds the timeout
        """
        def timeout_handler(signum, frame):
            raise TimeoutException("Code execution timed out")

        # Set up the timeout
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(seconds)

        try:
            yield
        finally:
            # Restore the old handler and cancel the alarm
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

    # RestrictedPython guard functions

    def _safe_getattr(self, obj, attr, default=None):
        """Guard function for attribute access."""
        if attr in self.BLOCKED_ATTRIBUTES:
            raise AttributeError(f"Access to attribute '{attr}' is not allowed")
        return getattr(obj, attr, default)

    def _safe_getitem(self, obj, index):
        """Guard function for item access."""
        return obj[index]

    def _safe_write(self, obj):
        """Guard function for write operations."""
        return obj