"""
ReAct Data Analytics Agent

This module implements the core DataAnalyticsAgent class that orchestrates all tools
using the ReAct (Reasoning + Acting) pattern to answer user queries about data.

The agent follows a Think → Act → Observe loop:
1. THINK: Reason about what needs to be done
2. ACT: Execute a tool with specific parameters
3. OBSERVE: Review the result and decide next steps

Key Features:
- Multi-step reasoning and planning
- Safe code execution for data analysis
- Conversation memory across turns
- Comprehensive error handling
- Support for OpenAI LLM
- Visualization tracking and collection
"""

from typing import Dict, List, Any, Optional
import json
import logging

from .tools.base import BaseTool
from .tools.dataset import LoadCSVTool, ListDatasetsTool, InspectDatasetTool
from .tools.analytics import AnalyzeTool, VisualizeTool
from .tools.deterministic import HighInTool, LowInTool, TargetTool
from .executor import SafeCodeExecutor
from .memory import ConversationMemory
from .prompts import build_agent_prompt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataAnalyticsAgent:
    """
    ReAct agent for data analytics that orchestrates tools to answer user queries.

    This agent uses the ReAct (Reasoning + Acting) pattern to break down complex
    data analysis queries into step-by-step actions. It maintains conversation
    context, manages datasets, and coordinates tool execution to provide
    comprehensive answers to user questions.

    The agent supports:
    - Loading and inspecting datasets (CSV files)
    - Generating and executing pandas code for analysis
    - Creating visualizations using matplotlib/seaborn
    - Multi-turn conversations with memory
    - Error recovery and retry logic
    - The OpenAI LLM

    Example:
        >>> from openai import OpenAI
        >>> agent = DataAnalyticsAgent(OpenAI(), max_iterations=10)
        >>> result = agent.run("Load foods3.csv and show me the top 5 foods by calories")
        >>> print(result["answer"])
    """

    def __init__(
        self,
        llm_client,
        max_iterations: int = 20,
        verbose: bool = True
    ):
        """
        Initialize the ReAct agent.

        Args:
            llm_client: OpenAI client for LLM API calls
            max_iterations: Maximum number of ReAct iterations to prevent infinite loops (default: 10)
            verbose: Print detailed logs of agent reasoning and actions (default: True)
        """
        self.llm_client = llm_client
        self.max_iterations = max_iterations
        self.verbose = verbose

        # Initialize memory for conversation tracking
        self.memory = ConversationMemory()

        # Initialize executor with empty datasets (will be updated as datasets are loaded)
        self.executor = SafeCodeExecutor({})

        # Initialize all available tools
        self.tools: Dict[str, BaseTool] = {
            "load_csv": LoadCSVTool(),
            "list_datasets": ListDatasetsTool(),
            "inspect_dataset": InspectDatasetTool(),
            "high_in": HighInTool(),
            "low_in": LowInTool(),
            "target": TargetTool(),
            "analyze": AnalyzeTool(llm_client, self.executor),
            "visualize": VisualizeTool(llm_client, self.executor)
        }

        logger.info(f"DataAnalyticsAgent initialized with {len(self.tools)} tools")

    def run(self, user_query: str) -> Dict[str, Any]:
        """
        Execute ReAct loop to answer user query.

        This method orchestrates the agent's thinking and acting cycle:
        1. Build context with available tools, datasets, and recent conversation
        2. Loop up to max_iterations:
           a. Ask LLM to decide next action (think)
           b. Execute the selected tool (act)
           c. Observe the result and update memory
           d. If action is "DONE", return final answer
        3. Handle edge cases (max iterations, errors, etc.)

        Args:
            user_query: Natural language query from user

        Returns:
            Dict containing:
                - status: "success" or "error"
                - answer: Final answer text
                - visualizations: List of base64-encoded images
                - iterations: Number of iterations taken
                - conversation_history: List of turns for debugging
                - error_message: Optional error description
        """
        # Initialize tracking variables
        iterations = 0
        visualizations = []
        conversation_history = []

        if self.verbose:
            print(f"\n{'='*80}")
            print(f"USER QUERY: {user_query}")
            print(f"{'='*80}")

        # Main ReAct loop
        while iterations < self.max_iterations:
            iterations += 1

            # Step 1: Build agent prompt with current context
            try:
                agent_prompt = self._build_context_prompt(user_query, current_iteration=iterations)
            except Exception as e:
                logger.error(f"Failed to build context prompt: {e}")
                return {
                    "status": "error",
                    "error_message": f"Failed to build context: {str(e)}",
                    "iterations": iterations,
                    "conversation_history": conversation_history
                }

            # Step 2: Call LLM to decide next action
            try:
                llm_response = self._call_llm_for_action(agent_prompt)
            except Exception as e:
                logger.error(f"LLM call failed: {e}")
                return {
                    "status": "error",
                    "error_message": f"LLM error: {str(e)}",
                    "iterations": iterations,
                    "conversation_history": conversation_history
                }

            # Step 3: Parse LLM response
            thought = llm_response.get("thought", "")
            action = llm_response.get("action", "")
            parameters = llm_response.get("parameters", {})

            # ALWAYS log the ReAct reasoning (not just when verbose)
            logger.info(f"===== ITERATION {iterations} =====")
            logger.info(f"THOUGHT: {thought}")
            logger.info(f"ACTION: {action}")
            logger.info(f"PARAMETERS: {parameters}")

            # Print current step if verbose
            self._print_step(iterations, thought, action, parameters)

            # Step 4: Check if agent is done
            if action == "DONE":
                final_answer = parameters.get("answer", "")

                if not final_answer:
                    # Agent tried to finish without providing an answer
                    logger.warning("Agent used DONE without providing an answer")
                    return {
                        "status": "error",
                        "error_message": "Agent completed without providing an answer",
                        "iterations": iterations,
                        "conversation_history": conversation_history
                    }

                if self.verbose:
                    print(f"\n{'='*80}")
                    print(f"FINAL ANSWER: {final_answer}")
                    print(f"{'='*80}")

                return {
                    "status": "success",
                    "answer": final_answer,
                    "visualizations": visualizations,
                    "iterations": iterations,
                    "conversation_history": conversation_history
                }

            # Step 5: Execute the selected tool
            result = self._execute_tool(action, parameters)

            # Log the observation (always, not just when verbose)
            result_summary = self._summarize_result_for_logging(result)
            logger.info(f"OBSERVATION: {result_summary}")

            # Print result if verbose
            self._print_result(result)

            # Step 6: Track visualizations if this was a visualize tool call
            if action == "visualize" and result.get("status") == "success":
                data = result.get("data", {})
                if isinstance(data, dict) and "filepath" in data:
                    visualizations.append({
                        "filepath": data["filepath"],
                        "chart_type": data.get("chart_type", "chart"),
                        "query": data.get("query", "")
                    })

            # Step 7: Update executor with current datasets
            self._update_executor_datasets()

            # Step 8: Add turn to memory
            result_summary = self._summarize_result(result)
            self.memory.add_turn(
                query=user_query,
                thought=thought,
                action=action,
                action_params=parameters,
                result=result_summary
            )

            # Step 9: Add to conversation history
            conversation_history.append({
                "iteration": iterations,
                "thought": thought,
                "action": action,
                "parameters": parameters,
                "result": result
            })

        # Max iterations reached without completion
        logger.warning(f"Max iterations ({self.max_iterations}) reached")
        return {
            "status": "error",
            "error_message": f"Agent did not complete task within {self.max_iterations} iterations",
            "iterations": iterations,
            "conversation_history": conversation_history
        }

    def _build_context_prompt(self, user_query: str, current_iteration: int = 1) -> str:
        """
        Build the complete prompt for the agent including context and query.

        This combines:
        - Tool descriptions
        - Currently loaded datasets
        - Iteration budget awareness
        - Recent conversation history
        - Current user query

        Args:
            user_query: The user's current question
            current_iteration: Current iteration number for budget tracking

        Returns:
            Complete prompt string ready for LLM
        """
        from .tools.dataset import _DATASET_STORE

        # Get base agent prompt with tools, datasets, and iteration budget
        base_prompt = build_agent_prompt(
            tools=list(self.tools.values()),
            datasets=_DATASET_STORE,
            max_iterations=self.max_iterations
        )

        # Add recent conversation context
        recent_context = self.memory.get_recent_context(last_n=3)

        # Build final prompt with iteration awareness
        full_prompt = f"""{base_prompt}

{recent_context}

## Current Iteration: {current_iteration} / {self.max_iterations}

**Budget Status**: You have used {current_iteration} iteration(s) out of {self.max_iterations} available.
- If iteration >= 15 and query is exploratory: Begin synthesis immediately
- Reserve iterations for final answer synthesis

## Current User Query

{user_query}

Remember to respond with valid JSON in the format:
{{
  "thought": "your reasoning (include iteration budget consideration for exploratory queries)",
  "action": "tool_name or DONE",
  "parameters": {{...}}
}}
"""

        return full_prompt

    def _call_llm_for_action(self, prompt: str) -> Dict[str, Any]:
        """
        Call LLM to decide next action.

        Supports the OpenAI client. Handles JSON parsing
        and error cases.

        Args:
            prompt: Complete prompt with context and query

        Returns:
            Parsed JSON response with thought, action, and parameters

        Raises:
            RuntimeError: If LLM call fails or response is invalid
        """
        try:
            # Detect client type and call appropriate API
            if hasattr(self.llm_client, 'chat'):  # OpenAI
                # Use gpt-4o which supports JSON mode (gpt-4 base does not)
                response = self.llm_client.chat.completions.create(
                    model="o4-mini-2025-04-16",
                    # model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    # temperature=1,
                    response_format={"type": "json_object"}  # Force JSON output
                )
                content = response.choices[0].message.content

            else:
                raise RuntimeError("Unsupported LLM client type")

            # Parse JSON response
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError as e:
                # Try to extract JSON from markdown if present
                if "```json" in content:
                    start = content.find("```json") + 7
                    end = content.find("```", start)
                    if end != -1:
                        json_str = content[start:end].strip()
                        parsed = json.loads(json_str)
                    else:
                        raise RuntimeError(f"Invalid JSON response: {content[:200]}")
                else:
                    raise RuntimeError(f"Failed to parse JSON: {e}")

            # Validate required fields
            if "thought" not in parsed or "action" not in parsed:
                raise RuntimeError(f"Missing required fields in response: {parsed}")

            # Ensure parameters exists (default to empty dict)
            if "parameters" not in parsed:
                parsed["parameters"] = {}

            # Handle DONE action - extract answer from parameters
            if parsed["action"] == "DONE":
                if "answer" not in parsed["parameters"] and "final_answer" not in parsed["parameters"]:
                    # Check if answer is at top level (some LLMs do this)
                    if "answer" in parsed:
                        parsed["parameters"]["answer"] = parsed["answer"]
                    elif "final_answer" in parsed:
                        parsed["parameters"]["answer"] = parsed["final_answer"]
                # Normalize to "answer" key
                if "final_answer" in parsed["parameters"]:
                    parsed["parameters"]["answer"] = parsed["parameters"]["final_answer"]

            return parsed

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise RuntimeError(f"Failed to get LLM response: {str(e)}")

    def _execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool with given parameters.

        Validates tool exists, then delegates to the tool's execute method.
        All errors are caught and returned as error result dicts.

        Args:
            tool_name: Name of the tool to execute
            parameters: Parameters to pass to the tool

        Returns:
            Tool result dictionary with status, result, and optional error_message
        """
        # Validate tool exists
        if tool_name not in self.tools:
            logger.warning(f"Tool '{tool_name}' not found")
            return {
                "status": "error",
                "error_message": f"Tool '{tool_name}' not found. Available tools: {list(self.tools.keys())}"
            }

        tool = self.tools[tool_name]

        # Execute tool with error handling
        try:
            result = tool.execute(**parameters)
            return result

        except TypeError as e:
            # Usually means wrong parameters
            logger.error(f"Tool parameter error: {e}")
            return {
                "status": "error",
                "error_message": f"Invalid parameters for tool '{tool_name}': {str(e)}"
            }

        except Exception as e:
            # Catch-all for any other errors
            logger.error(f"Tool execution failed: {e}")
            return {
                "status": "error",
                "error_message": f"Tool execution error: {str(e)}"
            }

    def _update_executor_datasets(self):
        """
        Update executor with current datasets from the global store.

        This ensures the code executor has access to all datasets that have
        been loaded via the load_csv tool.
        """
        from .tools.dataset import _DATASET_STORE
        self.executor.update_datasets(_DATASET_STORE)

    def _summarize_result(self, result: Dict[str, Any]) -> str:
        """
        Create a concise summary of a tool result for memory storage.

        Args:
            result: Tool execution result dictionary

        Returns:
            String summary of the result (truncated if too long)
        """
        if result.get("status") == "error":
            return f"Error: {result.get('error_message', 'Unknown error')}"

        result_data = result.get("result", "")
        result_str = str(result_data)

        # Truncate long results
        if len(result_str) > 500:
            return result_str[:500] + "... (truncated)"

        return result_str

    def _summarize_result_for_logging(self, result: Dict[str, Any]) -> str:
        """
        Create a detailed summary of tool result for logging (includes more detail than memory).

        Args:
            result: Tool execution result dictionary

        Returns:
            String summary with status, key fields, and result preview
        """
        if result.get("status") == "error":
            return f"❌ ERROR: {result.get('error_message', 'Unknown error')}"

        # For successful results, show more detail
        summary_parts = ["✅ SUCCESS"]

        # Check if this is a visualization result (has filepath)
        data = result.get("data", {})
        if isinstance(data, dict) and "filepath" in data:
            # For visualizations, always show the full filepath (never truncate)
            filepath = data["filepath"]
            chart_type = data.get("chart_type", "chart")
            summary_parts.append(f"Created {chart_type}")
            summary_parts.append(f"Saved to: {filepath}")
        else:
            # For non-visualization results, show result data
            result_data = result.get("result", "")
            if result_data:
                result_str = str(result_data)
                # Show first 300 chars for logging (more than memory gets)
                if len(result_str) > 300:
                    summary_parts.append(f"Result: {result_str[:300]}... (truncated)")
                else:
                    summary_parts.append(f"Result: {result_str}")
            else:
                summary_parts.append("Result: (empty)")

        # Add visualization flag if present
        if result.get("visualizations"):
            viz_count = len(result.get("visualizations", []))
            summary_parts.append(f"[Generated {viz_count} visualization(s)]")

        return " | ".join(summary_parts)

    def _print_step(self, iteration: int, thought: str, action: str, parameters: Dict[str, Any]):
        """
        Print ReAct step if verbose mode is enabled.

        Args:
            iteration: Current iteration number
            thought: Agent's reasoning
            action: Selected action/tool
            parameters: Action parameters
        """
        if not self.verbose:
            return

        print(f"\n{'='*80}")
        print(f"ITERATION {iteration}")
        print(f"{'='*80}")
        print(f"THOUGHT: {thought}")
        print(f"ACTION: {action}")

        if parameters:
            print("PARAMETERS:")
            for key, value in parameters.items():
                # Truncate long parameter values
                value_str = str(value)
                if len(value_str) > 100:
                    value_str = value_str[:100] + "..."
                print(f"  {key}: {value_str}")

    def _print_result(self, result: Dict[str, Any]):
        """
        Print tool result if verbose mode is enabled.

        Args:
            result: Tool execution result
        """
        if not self.verbose:
            return

        print(f"\nRESULT:")
        status = result.get("status", "unknown")

        if status == "success":
            result_data = result.get("result", "")
            result_str = str(result_data)

            # Truncate long results for display
            if len(result_str) > 500:
                result_str = result_str[:500] + "... (truncated)"

            print(f"  Status: success")
            print(f"  Output: {result_str}")

        else:
            error_msg = result.get("error_message", "Unknown error")
            print(f"  Status: error")
            print(f"  Error: {error_msg}")

    def clear_memory(self):
        """
        Clear conversation memory.

        Useful for starting a fresh conversation without restarting the agent.
        """
        self.memory.clear()
        logger.info("Conversation memory cleared")

    def get_memory(self) -> ConversationMemory:
        """
        Get the memory object for inspection or advanced usage.

        Returns:
            ConversationMemory instance
        """
        return self.memory

    def get_tools(self) -> Dict[str, BaseTool]:
        """
        Get all available tools.

        Returns:
            Dictionary mapping tool names to tool instances
        """
        return self.tools

    def get_loaded_datasets(self) -> List[Dict[str, Any]]:
        """
        Get list of currently loaded datasets with metadata.

        Returns:
            List of dicts containing dataset info:
            [
                {
                    "alias": "foods",
                    "rows": 300,
                    "columns": 31,
                    "memory_mb": 0.15,
                    "column_names": ["fdc_id", "description", "calcium", "calories", "fat", "protein", "sodium"]
                },
                ...
            ]
        """
        from .tools.dataset import _DATASET_STORE

        datasets = []
        for alias, df in _DATASET_STORE.items():
            # Calculate memory usage
            memory_bytes = df.memory_usage(deep=True).sum()
            memory_mb = round(memory_bytes / (1024 * 1024), 2)

            datasets.append({
                "alias": alias,
                "rows": len(df),
                "columns": len(df.columns),
                "memory_mb": memory_mb,
                "column_names": df.columns.tolist()
            })

        return datasets