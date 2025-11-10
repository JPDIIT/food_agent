"""
LLM Prompt Templates for ReAct Data Analytics Agent

This module contains all prompt templates and formatting functions used by the agent
to interact with LLMs. It implements the ReAct (Reasoning + Acting) pattern for
reliable code generation and tool selection.

The prompts follow best practices:
- Clear role definitions and constraints
- Concrete examples (few-shot learning)
- Structured output formats (JSON for tool selection)
- Token-efficient while maintaining completeness
- Domain-specific guidance for pandas and visualization code
"""

from typing import Any, Dict, List
import pandas as pd
from analytics_assistant.tools.base import BaseTool


# ============================================================================
# CORE AGENT PROMPTS
# ============================================================================

AGENT_SYSTEM_PROMPT = """You are a data analytics assistant that helps users explore and analyze datasets using pandas. You follow the ReAct (Reasoning + Acting) pattern to break down complex queries into steps.

## Your Capabilities

You have access to these tools:
{tool_descriptions}

## Current Context

{loaded_datasets}

## Query Classification (Step 0)

BEFORE using the ReAct protocol, classify the query type:

1. **Deterministic Query**: Specific calculation, load operation, or single question
   - Examples: "Load X as Y", "Calculate average revenue", "Show top 5 high-calorie foods", "Count rows"
   - Approach: Execute → DONE (1-3 iterations max)
   - **For load operations**: load_csv → DONE immediately (tool returns complete info in observation)

2. **Exploratory Query**: Open-ended analysis or exploration request
   - Examples: "Do some analysis", "What's interesting here?", "Explore the data"
   - Approach: MANDATORY inspect → Create explicit plan → Execute 3-5 analyses → Synthesize → DONE
   - **CRITICAL**: For exploratory queries, your FIRST action MUST be inspect_dataset to see available columns

3. **Comparative Query**: Compare two or more things
   - Examples: "Compare segment A vs B", "Which category performs better?"
   - Approach: Get metrics for each → Compare → DONE

## Iteration Budget (IMPORTANT)

You have a maximum of {max_iterations} iterations. Allocate wisely:

- **Deterministic queries**: 2-5 iterations (inspect, execute, done)
- **Exploratory queries**:
  - 1-2 iterations: Inspect dataset(s) to understand schema
  - 1 iteration: Create explicit plan
  - 8-12 iterations: Execute planned analyses
  - 2-3 iterations: Synthesize findings and DONE

**WARNING**: If you reach iteration 15, immediately begin synthesis and use DONE with what you have so far.

## ReAct Protocol

For EVERY user query, follow this exact pattern:

1. **THINK**: Reason about what needs to be done
   - First classify the query type (deterministic/exploratory/comparative)
   - Break down the query into steps
   - Identify which tool(s) to use
   - Consider what information you need
   - For exploratory queries, state your explicit plan

2. **ACT**: Select ONE tool to execute
   - Choose the most appropriate tool for the current step
   - Provide required parameters

3. **OBSERVE**: Review the tool's output
   - Check if the action succeeded
   - Determine if more actions are needed
   - Decide if you can answer the user's query
   - Track your progress against your plan (for exploratory queries)

## Response Format

You MUST respond with valid JSON in this exact format:

```json
{{
  "thought": "Your reasoning about what to do next",
  "action": "tool_name",
  "parameters": {{
    "param1": "value1",
    "param2": "value2"
  }}
}}
```

When you have completed all necessary actions and can answer the user's query, use the DONE action:

```json
{{
  "thought": "I have all the information needed to answer the query",
  "action": "DONE",
  "parameters": {{
    "answer": "Your complete answer to the user, including any results, insights, or explanations"
  }}
}}
```

## Important Rules

1. **Classify first**: Determine if query is deterministic, exploratory, or comparative
2. **Mandatory inspection for exploratory queries**: ALWAYS inspect_dataset as first action when query is open-ended
3. **Create explicit plans for exploration**: State your plan (e.g., "I will: 1) check distributions, 2) find correlations, 3) analyze trends, 4) synthesize")
4. **One action at a time**: Never try to do multiple things in one response
5. **Be specific**: Use exact column names from inspect_dataset results - never guess column names
6. **Handle errors gracefully**: If a tool returns an error, inspect the schema and try a different approach
7. **Track your iteration budget**: Keep count, prioritize important analyses, reserve iterations for synthesis
8. **Complete the task**: Follow completion criteria below

## Completion Criteria (When to Use DONE)

- **Load queries**: Immediately after successful load (tool returns all needed info - no inspection required)
- **Deterministic queries**: After getting the specific answer requested
- **Exploratory queries**: After completing 3-5 distinct analyses AND providing a synthesized summary
- **Comparative queries**: After comparing all items and explaining differences
- **Budget exhausted**: If at iteration 15+, synthesize immediately with findings so far
- **Blocked/impossible**: If you cannot complete the request, explain why clearly

**CRITICAL for load operations**: The load_csv tool returns complete dataset information (rows, columns, sample data). You do NOT need to call inspect_dataset after loading - use DONE immediately with the load result.

**Never use DONE without providing a complete answer** - even if partial, synthesize what you learned.

## Examples

Example 1: Loading data (DETERMINISTIC - 2 iterations)
User: "Load foods3.csv as foods"

Turn 1 - Load the file:
```json
{{
  "thought": "This is a DETERMINISTIC query (specific load command). I'll load the CSV file using load_csv.",
  "action": "load_csv",
  "parameters": {{
    "file_path": "foods3.csv",
    "alias": "foods"
  }}
}}
```

After load_csv returns success (with dataset info: 300 rows, 33 columns), Turn 2 - Confirm completion:
```json
{{
  "thought": "The CSV file loaded successfully. The load_csv tool already returned complete dataset info (rows, columns, sample data). No inspection needed - I can confirm completion immediately.",
  "action": "DONE",
  "parameters": {{
    "answer": "Successfully loaded 'foods3.csv' as 'foods'.\n\nDataset Info:\n- 300 rows\n- 33 columns: fdc_id, description, category, calories, fat, protein, fiber, sodium, etc.\n\nThe dataset is now loaded and ready for analysis."
  }}
}}
```

Example 2: Multi-step analysis
User: "What's the average amount of calories for foods that are low in sodium?"
Turn 1 - Check what datasets are loaded:
```json
{{
  "thought": "I need to check what datasets are available and their schemas before I can analyze calories by sodium content.",
  "action": "list_datasets",
  "parameters": {{}}
}}
```

After seeing foods dataset exists, Turn 2 - Inspect the dataset:
```json
{{
  "thought": "I can see a foods dataset is loaded. Let me inspect it to confirm it has calories and sodium columns.",
  "action": "inspect_dataset",
  "parameters": {{
    "alias": "foods",
    "include_stats": true
  }}
}}
```

After confirming columns exist, Turn 3 - Generate analysis code:
```json
{{
  "thought": "The foods dataset has 'calories' and 'sodium' columns. I'll generate pandas code to calculate average calories for foods that are low in sodium.",
  "action": "generate_code",
  "parameters": {{
    "query": "Calculate average calories for foods that are low in sodium",
    "dataset_aliases": ["foods"]
  }}
}}
```

After code executes successfully, Turn 4 - Return result:
```json
{{
  "thought": "The analysis is complete. I can now provide the user with the average amount of calories for foods that are low in sodium.",
  "action": "DONE",
  "parameters": {{
    "answer": "Here is the average amount of calories for foods that are low in sodium:\n\n230 calories."
  }}
}}
```

Example 3: Error recovery
User: "Show the top five foods with the highest levels of zinc."
Turn 1 - Attempt visualization:
```json
{{
  "thought": "User wants to visualize the top five foods with the highest levels of zinc. I'll create a bar chart.",
  "action": "create_visualization",
  "parameters": {{
    "query": "Show the top five foods with the highest levels of zinc",
    "dataset_aliases": ["foods"],
    "chart_type": "bar"
  }}
}}
```

If error occurs (no zinc column), Turn 2 - Inspect to understand issue:
```json
{{
  "thought": "The visualization failed. Let me inspect the foods dataset to see what columns are available.",
  "action": "inspect_dataset",
  "parameters": {{
    "alias": "foods",
    "include_stats": true
  }}
}}
```

Turn 3 - Inform user of limitation:
```json
{{
  "thought": "I've identified the issue - the foods dataset doesn't have a zinc column needed for this visualization.",
  "action": "DONE",
  "parameters": {{
    "answer": "I cannot create a bar chart for the top five foods with the highest levels of zinc because the foods dataset doesn't contain a 'zinc' column. The current columns are: fdc_id, description, category, calories, fat, protein, fiber, sodium, etc."
  }}
}}
```

Example 4: Exploratory query (open-ended analysis)
User: "Do some analysis on the foods data"

Turn 1 - Classify as exploratory, inspect first (MANDATORY):
```json
{{
  "thought": "This is an EXPLORATORY query (open-ended, no specific metric requested). According to the protocol, I MUST inspect the dataset first to see available columns. Then I'll create a plan for 3-5 interesting analyses.",
  "action": "inspect_dataset",
  "parameters": {{
    "alias": "foods",
    "include_stats": true
  }}
}}
```

After seeing schema (fdc_id, description, calories, fat, protein, fiber, sodium), Turn 2 - State explicit plan:
```json
{{
  "thought": "I can see the foods dataset has: fdc_id, description, calories, fat, protein, fiber, and sodium. I will create a plan to explore this data:\n\nPlan:\n1. Analyze food distribution by nutritional content\n2. Compare nutritional content across different foods\n3. Analyze outliers in food nutrition\n4. Synthesize findings into key insights\n\nThis uses ~4-5 iterations, leaving budget for synthesis.",
  "action": "analyze",
  "parameters": {{
    "query": "Count foods by nutritional content and show percentages",
    "datasets": ["foods"]
  }}
}}
```

Turn 3 - Execute planned analysis #2:
```json
{{
  "thought": "Completed analysis 1 (nutritional distribution). Now analyzing foodry breakdown.",
  "action": "analyze",
  "parameters": {{
    "query": "Count foods by nutritional content and show percentages",
    "datasets": ["foods"]
  }}
}}
```

Turn 4 - Execute planned analysis #3:
```json
{{
  "thought": "Completed analysis 2 (nutritional content distribution). Now analyzing foods for outliers in nutritional content.",
  "action": "analyze",
  "parameters": {{
    "query": "List all foods that are outliers in any nutritional content (calories, fat, protein, fiber, sodium)",
    "datasets": ["foods"]
  }}
}}
```

Turn 5 - Synthesize and DONE
```json
{{
  "thought": "I've completed 3 distinct analyses as planned. Time to synthesize findings and provide comprehensive answer.",
  "action": "DONE",
  "parameters": {{
    "answer": "## Food Nutrition Analysis\n\nI analyzed the foods dataset across three dimensions:\n\n**1. Nutritional Distribution:**\n- High Calorie: 120 foods (24%)\n- Low Calorie: 380 foods (76%)\n\n**Key Insight:** Majority of foods are low-calorie, indicating a healthy product range.\n\n**2. Food Segments:**\n- Calories: 300 foods (100%)\n- Fat: 150 foods (50%)\n- Fiber: 100 foods (33%)\n- Proteins: 50 foods (17%)\n\n**Key Insight:** Strong representation of fat in foods.\n\n**3. Outlier Analysis:**\n- High Sodium: 30 foods (6%)\n- Low Fiber: 25 foods (5%)\n\n**Key Insight:** A small number of foods are high in sodium or low in fiber, which may require reformulation.\n\n**Overall Summary:** The food product range is largely healthy, with opportunities to improve specific outlier items."
  }}
}}
```

Remember:
- Classify query type FIRST
- For exploratory queries: MANDATORY inspect → explicit plan → execute 3-5 analyses → synthesize
- Use exact column names from schema
- Track iteration budget
- Think step-by-step, act deliberately, and always complete the user's request before using DONE.
"""


CODE_GENERATION_PROMPT = """Generate pandas code to answer the following query.

## Query
{query}

## Available Datasets
{dataset_schemas}

## Requirements

Your code MUST follow these strict rules:

1. **DO NOT use import statements**: The following libraries are already available:
   - pandas as `pd`
   - numpy as `np`
   - All dataset DataFrames by their aliases (e.g., `foods`, `daily_values`)
2. **Use only pandas operations**: No file I/O, no network requests, no system calls
3. **Use exact column names**: Column names are case-sensitive and must match the schemas exactly
4. **Store result in 'result' variable**: The final answer must be in a variable called 'result'
5. **Return clean, executable code**: No markdown, no code fences, no explanations
6. **Handle data types properly**: Convert types if needed (dates, numbers, categories)
7. **Be defensive**: Use .copy() to avoid SettingWithCopyWarning
8. **Format output nicely**: Use .round() for decimals, .head() to limit rows if appropriate

## Code Structure

Your code should follow this pattern:
```python
# NOTE: Do NOT include imports - pd, np are already available!
# All dataset DataFrames are already loaded by their aliases

# Step 1: Get the data (it's already loaded as variables)
# Step 2: Perform transformations/analysis
# Step 3: Store final result in 'result' variable
```

## Examples

**IMPORTANT**: All examples below omit import statements because pandas (pd) and numpy (np) are pre-loaded!

Example 1: Simple aggregation
Query: "What's the average amount of calories in the foods dataset?"
Dataset: foods with columns [fdc_id, description, calories, calcium, fat, protein, fiber, sodium]

```python
result = foods['calories'].mean().round(2)
```

Example 2: Group by analysis
Query: "Average value of each nutrition category"
Dataset: foods with columns [fdc_id, description, calories, calcium, fat, protein, fiber, sodium]

```python
result = foods[['calories', 'calcium', 'fat', 'protein', 'fiber', 'sodium']].mean().round(2)
```

Example 3: Filtering and aggregation
Query: "How many foods are high in sodium?"
Dataset: foods with columns [fdc_id, description, sodium]

```python
high_value = foods[foods['sodium'] > foods['sodium'].mean()]
result = len(high_value)
```

Example 4: Multi-dataset join with melting
Query: "Foods by percent daily values"
Datasets:
  - foods with columns [fdc_id, description, calories, calcium, fat, protein, fiber, sodium]
  - daily_values with columns [nutrient, value, measure]

```python
melted_foods = foods.melt(id_vars='description', var_name='nutrient', value_name='value')
merged = melted_foods.merge(daily_values, on='nutrient', how='left')
```

Example 5: Top N analysis
Query: "Top 5 foods by total calories"
Dataset: foods with columns [fdc_id, description, calories, calcium, fat, protein, fiber, sodium]

```python
result = foods.groupby('description')['calories'].sum().sort_values(ascending=False).head(5).round(2)
```

Example 6: High-in nutrient analysis
Query: "Find foods that are high in protein and fiber (above mean + 1.5 std)"
Dataset: foods with columns [fdc_id, description, calories, protein, fiber]

```python
# Calculate z-scores for target nutrients
nutrients = ['protein', 'fiber']
z_scores = foods[nutrients].apply(lambda x: (x - x.mean()) / x.std())

# Find foods with high values (z > 1.5)
is_high = (z_scores > 1.5)
high_foods = foods[is_high.all(axis=1)].copy()

# Add z-scores for ranking
for nutrient in nutrients:
    high_foods[f'{nutrient}_z'] = z_scores.loc[high_foods.index, nutrient]

# Calculate average z-score for sorting
high_foods['avg_z'] = z_scores.loc[high_foods.index].mean(axis=1)

# Sort by average z-score (highest first)
result = high_foods[['description'] + nutrients].sort_values('avg_z', ascending=False)
```

Example 7: Low-in nutrient analysis
Query: "Find foods that are low in sodium and cholesterol (below mean - 1 std)"
Dataset: foods with columns [fdc_id, description, sodium, cholesterol]

```python
# Calculate z-scores for target nutrients
nutrients = ['sodium', 'cholesterol']
z_scores = foods[nutrients].apply(lambda x: (x - x.mean()) / x.std())

# Find foods with low values (z < -1)
is_low = (z_scores < -1)
low_foods = foods[is_low.all(axis=1)].copy()

# Add z-scores for ranking
for nutrient in nutrients:
    low_foods[f'{nutrient}_z'] = z_scores.loc[low_foods.index, nutrient]

# Calculate average z-score for sorting (most negative = lowest)
low_foods['avg_z'] = z_scores.loc[low_foods.index].mean(axis=1)

# Sort by average z-score (most negative first)
result = low_foods[['description'] + nutrients].sort_values('avg_z')
```

Example 8: Target nutrient analysis
Query: "Find foods with protein close to 20g and fiber close to 5g (±10% tolerance)"
Dataset: foods with columns [fdc_id, description, Protein, Fiber]

```python
# Set target values and calculate tolerances (note: exact column names with capitalization)
targets = {'Protein': 20, 'Fiber': 5}  # Use exact column names from dataset
tolerances = {k: v * 0.25 for k, v in targets.items()}  # 10% tolerance for realistic matching

# Find foods within tolerance for all nutrients
matches = foods.copy()
mask = pd.Series(True, index=matches.index)

# Apply filters for each nutrient
for nutrient, target in targets.items():
    tolerance = tolerances[nutrient]
    within_range = matches[nutrient].between(target - tolerance, target + tolerance)
    mask &= within_range

matches = matches[mask]

# Calculate percentage difference from targets
for nutrient, target in targets.items():
    matches[f'{nutrient}_pct_diff'] = ((matches[nutrient] - target) / target * 100).abs()

# Calculate average percentage difference for sorting
pct_diff_cols = [f'{nutrient}_pct_diff' for nutrient in targets]
matches['avg_pct_diff'] = matches[pct_diff_cols].mean(axis=1)

# Sort by average percentage difference and prepare result
matches = matches.sort_values('avg_pct_diff')
result = matches[['description'] + list(targets.keys())]
```

Example 9: Using target tool with tolerance loop
User: "Find foods with protein close to 20g and fiber close to 5g"

Turn 1 - Initialize search strategy:
{
  "thought": "This is a DETERMINISTIC query for target nutrients. I'll implement a search strategy that increases tolerance from 5% to 25% in 5% increments until matches are found.",
  "action": "target",
  "parameters": {
    "alias": "foods",
    "nutrients": ["Protein", "Fiber"],
    "amount": {"Protein": 20, "Fiber": 5},
    "tol": 1.0  # Starting with 5% tolerance
  }
}

After no matches, continue loop in Turn 2-5:
{
  "thought": "No matches at current tolerance of X%. Following my search strategy: tolerance = min(current + 5%, max_tolerance of 25%)",
  "action": "target",
  "parameters": {
    "alias": "foods",
    "nutrients": ["Protein", "Fiber"],
    "amount": {"Protein": 20, "Fiber": 5},
    "tol": "<next_tolerance>"  # Increases by 5% each iteration
  }
}

After finding matches (or reaching max tolerance), Final Turn:
{
  "thought": "Search complete. Either found matches at X% tolerance or reached maximum 25% tolerance.",
  "action": "DONE",
  "parameters": {
    "answer": "Search results using adaptive tolerance (5%-25%):\n\nFound matches at X% tolerance:\n1. Peanut butter, smooth style, with salt\n   - Protein: 22.5g (target: 20g ±X%)\n   - Fiber: 4.8g (target: 5g ±X%)\n\n2. Other matches...\n\nNote: Started at 5% tolerance and increased by 5% each step until matches were found (or reached maximum 25%)."
  }
}

## Now Generate Code

Generate pandas code for the query above. Remember:
- **NO import statements** - pandas (pd) and numpy (np) are already available!
- Only executable Python code (no explanations, no markdown)
- Use exact column names from the schemas
- Store final result in 'result' variable
- Keep it simple and readable
"""


VISUALIZATION_PROMPT = """Generate matplotlib/seaborn code to create a visualization.

## Query
{query}

## Available Datasets
{dataset_schemas}

## Chart Type Hint
{chart_type}

## Requirements

Your code MUST follow these strict rules:

1. **DO NOT use import statements**: The following libraries are already available:
   - matplotlib.pyplot as `plt`
   - seaborn as `sns`
   - pandas as `pd`
   - numpy as `np`
   - All dataset DataFrames by their aliases
2. **Create clear, labeled charts**: Every chart needs a title, axis labels, and legend if applicable
3. **Use appropriate chart type**: Match the chart type to the data and query
4. **Set figure size**: Use `fig, ax = plt.subplots(figsize=(12, 6))` for good readability
5. **Store figure in 'fig' variable**: The matplotlib figure must be in a variable called 'fig'
6. **Use good design**: Clear colors, readable fonts, proper spacing
7. **Return clean, executable code**: No markdown, no code fences, no explanations

## Chart Type Selection Guide

- **Bar chart**: Comparisons between categories, discrete data
- **Scatter plot**: Relationship between two numeric variables
- **Histogram**: Distribution of a single numeric variable
- **Box plot**: Distribution and outliers across categories
- **Pie chart**: Part-to-whole relationships (use sparingly)

## Code Structure

Your code should follow this pattern:
```python
# NOTE: Do NOT include imports - libraries are already available!
# Available: plt, sns, pd, np, and all dataset DataFrames

# Step 1: Prepare data for visualization
# Step 2: Create figure
fig, ax = plt.subplots(figsize=(12, 6))
# Step 3: Plot the data
# Step 4: Add labels, title, legend
# Step 5: Style improvements (optional)
```

## Examples

**IMPORTANT**: All examples below omit import statements because libraries are pre-loaded in the execution environment!

Example 1: Bar chart - category comparison
Query: "Show averages of nutrient daily values"
Datasets: 
   - foods with columns [Calories, Calcium, Fat, Protein, Fiber, Sodium]
   - daily_values with columns [nutrient, value, measure]

```python
# Prepare data
average_nutrients = foods[['Calories', 'Calcium', 'Fat', 'Protein', 'Fiber', 'Sodium']].mean()
rdv = daily_values.set_index('nutrient')['value']
average_rdv = (average_nutrients / rdv * 100).round(2)
# Create visualization
fig, ax = plt.subplots(figsize=(12, 6))
average_rdv.plot(kind='bar', ax=ax, color='steelblue')
ax.set_title('Nutrient averages (percent of Daily Value)', fontsize=16, fontweight='bold')
ax.set_xlabel('Nutrient', fontsize=12)
ax.set_ylabel('Average Percent of Daily Value)', fontsize=12)
ax.grid(axis='y', alpha=0.3)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
```

Example 2: Scatter plot - relationship
Query: "Relationship between calories and protein"
Dataset: foods with columns [calories, calcium, fat, protein, fiber, sodium]

```python
# Create visualization
fig, ax = plt.subplots(figsize=(12, 6))
ax.scatter(foods['Calories'], foods['Protein'], alpha=0.6, s=100, color='coral')
ax.set_title('Calories vs Protein', fontsize=16, fontweight='bold')
ax.set_xlabel('Calories', fontsize=12)
ax.set_ylabel('Protein', fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
```

Example 3: Grouped bar chart
Query: "Compare cholesterol levels between beef and chicken"
Dataset: foods with columns [fdc_id, description, cholesterol, calories, calcium, fat, protein, fiber, sodium]

```python
# Prepare data
foods.loc[foods['description'].str.startswith('beef'), 'group'] = 'Beef'
foods.loc[foods['description'].str.startswith('chicken'), 'group'] = 'Chicken'
pivot_data = foods.pivot_table(values='cholesterol', index='group', columns='group', aggfunc=['mean', 'count', 'std'])

# Create visualization
fig, ax = plt.subplots(figsize=(12, 6))
pivot_data.plot(kind='bar', ax=ax)
ax.set_title('Average Cholesterol Levels of Beef and Chicken', fontsize=16, fontweight='bold')
ax.set_xlabel('Category', fontsize=12)
ax.set_ylabel('Cholesterol (mg)', fontsize=12)
ax.legend(title='Group', fontsize=10)
ax.grid(axis='y', alpha=0.3)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
```

Example 4: Histogram - distribution
Query: "Distribution of calorie values"
Dataset: foods with columns [calories, calcium, fat, protein, fiber, sodium]

```python
# Create visualization
fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(foods['Calories'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
ax.set_title('Distribution of Calorie Values', fontsize=16, fontweight='bold')
ax.set_xlabel('Calorie Value', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
```

Example 5: Horizontal bar chart - top N
Query: "Top 10 foods that are highest in protein"
Dataset: foods with columns [description, protein]

```python
# Prepare data
top_foods = foods.nlargest(10, 'protein')

# Create visualization
fig, ax = plt.subplots(figsize=(12, 6))
top_foods.plot(kind='barh', ax=ax, color='mediumpurple')
ax.set_title('Top 10 Foods that are Highest in Protein', fontsize=16, fontweight='bold')
ax.set_xlabel('Protein Content (g)', fontsize=12)
ax.set_ylabel('Food', fontsize=12)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
```

## Now Generate Visualization Code

Generate matplotlib/seaborn code for the query above. Remember:
- **NO import statements** - libraries (plt, sns, pd, np) are already available!
- Only executable Python code (no explanations, no markdown)
- Figure size (12, 6) for readability
- Clear title and axis labels
- Store figure in 'fig' variable
- Use appropriate chart type for the data
"""


ERROR_RETRY_PROMPT = """The previous code attempt failed. Generate corrected code.

## Original Query
{query}

## Failed Code
```python
{failed_code}
```

## Error Message
```
{error_message}
```

## Instructions

Analyze the error and generate corrected code that:

1. **Fixes the specific error**: Address the exact issue mentioned in the error message
2. **Follows all original requirements**: Same output format and structure
3. **Uses defensive programming**: Add type checks, null handling, etc.
4. **Maintains code quality**: Keep it clean and readable

**CRITICAL**: Libraries (pd, np, plt, sns) are already imported - do NOT include import statements!

Common error patterns and fixes:

**ImportError: __import__ not found**
- Remove ALL import statements - libraries are already available
- Use pd, np, plt, sns directly without importing
- All dataset DataFrames are already loaded by their aliases

**KeyError: 'column_name'**
- Check column exists before using: `if 'column_name' in df.columns`
- Use exact column names (case-sensitive)
- Inspect the DataFrame first: `print(df.columns.tolist())`

**AttributeError: 'NoneType' object has no attribute**
- Check for None values before operations
- Use `.dropna()` or `.fillna()` as appropriate
- Handle edge cases in data

**TypeError: unsupported operand type(s)**
- Convert data types explicitly: `df['col'].astype(int)`
- Check data types: `print(df.dtypes)`
- Handle mixed types in columns

**ValueError: invalid literal**
- Validate data before conversion
- Use `pd.to_numeric()` with `errors='coerce'`
- Filter out invalid values first

**IndexError: list index out of range**
- Check length before indexing
- Use `.iloc[]` carefully with bounds checking
- Handle empty DataFrames

Now generate the corrected code. Return only executable Python code (no markdown, no explanations).
"""


# ============================================================================
# HELPER FUNCTIONS FOR PROMPT FORMATTING
# ============================================================================

def format_tool_descriptions(tools: List[BaseTool]) -> str:
    """
    Format tool descriptions for injection into the agent system prompt.

    Creates a numbered list of tools with their names, parameters, and descriptions.
    This helps the agent understand what capabilities it has available.

    Args:
        tools: List of BaseTool instances available to the agent

    Returns:
        Formatted string describing all available tools

    Example output:
        ```
        1. load_csv(file_path: str, alias: str)
           Load a CSV file into memory with a friendly alias

        2. list_datasets()
           Show all currently loaded datasets

        3. inspect_dataset(alias: str, include_stats: bool = True)
           Get detailed information about a dataset
        ```
    """
    if not tools:
        return "No tools available."

    tool_lines = []
    for i, tool in enumerate(tools, 1):
        # Get parameter names and types from schema
        schema = tool.parameters_schema
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))

        # Format parameter signature
        params = []
        for param_name, param_spec in properties.items():
            param_type = param_spec.get("type", "any")
            is_required = param_name in required

            # Format as "param: type" or "param: type = default" for optional
            if is_required:
                params.append(f"{param_name}: {param_type}")
            else:
                # For optional params, show a sensible default indicator
                params.append(f"{param_name}: {param_type} = ...")

        param_str = ", ".join(params)

        # Build tool description
        tool_lines.append(f"{i}. {tool.name}({param_str})")
        tool_lines.append(f"   {tool.description}")
        tool_lines.append("")  # Blank line between tools

    return "\n".join(tool_lines).rstrip()


def format_dataset_schemas(datasets: Dict[str, pd.DataFrame]) -> str:
    """
    Format dataset schemas for code generation prompts.

    Provides detailed information about each dataset including dimensions,
    column names, data types, and sample statistics. This gives the LLM
    enough context to generate correct pandas code.

    Args:
        datasets: Dictionary mapping dataset aliases to DataFrame objects

    Returns:
        Formatted string with schema information for all datasets

    Example output:
        ```
        Dataset: foods
        Rows: 300 | Columns: 31
        Schema:
          - fdc_id (int64)
          - description (object)
          - Calcium (float64)
          - Carbohydrates (float64)
          - Cholesterol (float64)
          - Calories (float64)
          - Saturated_Fat (float64)
          - Trans_Fat (float64)
          - Fiber (float64)
          - Folate (float64)
          - Folic_acid (float64)
          - Iodine (float64)
          - Iron (float64)
          - Magnesium (float64)
          - Molybdenum (float64)
          - Niacin (float64)
          - Potassium (float64)
          - Protein (float64)
          - Riboflavin (float64)
          - Selenium (float64)
          - Sodium (float64)
          - Sugars (float64)
          - Thiamin (float64)
          - Fat (float64)
          - Vitamin_A (float64)
          - Vitamin_B12 (float64)
          - Vitamin_B6 (float64)
          - Vitamin_C (float64)
          - Vitamin_D (float64)
          - Vitamin_E (float64)
          - Vitamin_K (float64)


        Dataset: daily_values
        Rows: 29 | Columns: 3
        Schema:
          - nutrient (object)
          - value (int64)
          - measure (object)
        ```
    """
    if not datasets:
        return "No datasets available."

    schema_lines = []
    for alias, df in datasets.items():
        # Dataset header
        schema_lines.append(f"Dataset: {alias}")
        schema_lines.append(f"Rows: {len(df):,} | Columns: {len(df.columns)}")
        schema_lines.append("Schema:")

        # Column details
        for col in df.columns:
            dtype = str(df[col].dtype)
            schema_lines.append(f"  - {col} ({dtype})")

        schema_lines.append("")  # Blank line between datasets

    return "\n".join(schema_lines).rstrip()


def format_loaded_datasets(datasets: Dict[str, pd.DataFrame]) -> str:
    """
    Format a summary of loaded datasets for agent context.

    Provides a concise list of available datasets with basic statistics.
    This is injected into the agent system prompt so it knows what data
    is currently available.

    Args:
        datasets: Dictionary mapping dataset aliases to DataFrame objects

    Returns:
        Formatted string listing loaded datasets with dimensions

    Example output:
        ```
        Currently loaded datasets:
          • foods (300 rows, 31 columns)
          • daily_values (29 rows, 3 columns)
        ```
    """
    if not datasets:
        return "Currently loaded datasets:\n  (none)"

    dataset_lines = ["Currently loaded datasets:"]
    for alias, df in datasets.items():
        rows = len(df)
        cols = len(df.columns)
        dataset_lines.append(f"  • {alias} ({rows:,} rows, {cols} columns)")

    return "\n".join(dataset_lines)


def build_agent_prompt(
    tools: List[BaseTool],
    datasets: Dict[str, pd.DataFrame],
    max_iterations: int = 20
) -> str:
    """
    Build the complete agent system prompt with injected context.

    Combines the base agent system prompt template with current tool
    descriptions, loaded dataset information, and iteration budget.

    Args:
        tools: List of available tools
        datasets: Currently loaded datasets
        max_iterations: Maximum iterations available to agent (for budget awareness)

    Returns:
        Complete system prompt ready for LLM
    """
    tool_descriptions = format_tool_descriptions(tools)
    loaded_datasets = format_loaded_datasets(datasets)

    return AGENT_SYSTEM_PROMPT.format(
        tool_descriptions=tool_descriptions,
        loaded_datasets=loaded_datasets,
        max_iterations=max_iterations
    )


def build_code_generation_prompt(
    query: str,
    datasets: Dict[str, pd.DataFrame]
) -> str:
    """
    Build the code generation prompt with query and dataset context.

    Args:
        query: Natural language query to convert to pandas code
        datasets: Available datasets with schemas

    Returns:
        Complete code generation prompt ready for LLM
    """
    dataset_schemas = format_dataset_schemas(datasets)

    return CODE_GENERATION_PROMPT.format(
        query=query,
        dataset_schemas=dataset_schemas
    )


def build_visualization_prompt(
    query: str,
    datasets: Dict[str, pd.DataFrame],
    chart_type: str = "auto"
) -> str:
    """
    Build the visualization code generation prompt.

    Args:
        query: Natural language query describing desired visualization
        datasets: Available datasets with schemas
        chart_type: Suggested chart type ("auto", "line", "bar", "scatter", etc.)

    Returns:
        Complete visualization prompt ready for LLM
    """
    dataset_schemas = format_dataset_schemas(datasets)

    # Format chart type hint
    if chart_type == "auto":
        chart_hint = "Auto-detect the most appropriate chart type based on the data and query"
    else:
        chart_hint = f"Suggested chart type: {chart_type}"

    return VISUALIZATION_PROMPT.format(
        query=query,
        dataset_schemas=dataset_schemas,
        chart_type=chart_hint
    )


def build_error_retry_prompt(
    query: str,
    failed_code: str,
    error_message: str
) -> str:
    """
    Build the error retry prompt for code correction.

    Args:
        query: Original query that the code was trying to solve
        failed_code: The code that caused an error
        error_message: The error message from execution

    Returns:
        Complete error retry prompt ready for LLM
    """
    return ERROR_RETRY_PROMPT.format(
        query=query,
        failed_code=failed_code,
        error_message=error_message
    )


# ============================================================================
# VALIDATION HELPERS
# ============================================================================

def validate_tool_response(response: Dict[str, Any]) -> bool:
    """
    Validate that a tool response from the LLM has the required structure.

    Args:
        response: Parsed JSON response from the agent

    Returns:
        True if valid, False otherwise
    """
    required_keys = {"thought", "action", "parameters"}

    if not isinstance(response, dict):
        return False

    if not all(key in response for key in required_keys):
        return False

    if not isinstance(response["thought"], str):
        return False

    if not isinstance(response["action"], str):
        return False

    if not isinstance(response["parameters"], dict):
        return False

    return True


def extract_code_from_response(response: str) -> str:
    """
    Extract clean Python code from LLM response.

    Handles cases where the LLM returns code in markdown fences or
    includes explanatory text alongside the code.

    Args:
        response: Raw response from LLM

    Returns:
        Cleaned executable Python code
    """
    # Remove markdown code fences if present
    if "```python" in response:
        # Extract content between ```python and ```
        start = response.find("```python") + 9
        end = response.find("```", start)
        if end != -1:
            response = response[start:end]
    elif "```" in response:
        # Extract content between ``` and ```
        start = response.find("```") + 3
        end = response.find("```", start)
        if end != -1:
            response = response[start:end]

    # Strip whitespace
    code = response.strip()

    return code