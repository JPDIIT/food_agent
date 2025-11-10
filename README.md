# Food and Nutrition Data Analytics Agent

A ReAct-based AI agent that can query and analyze extensive nutrition data from CSV files through natural language queries. Built with code generation, safe execution, and conversational memory.
This project is an unoffical fork of this repo: https://github.com/00ber/build-fellowship-ai-assistant and builds off of much of its code. It has been re-worked and expanded to suit the specific nature, requirements and use cases of the food and nutition datasets.

## Quick Start

### 1. Setup Environment

```bash
# Install dependencies with uv
uv sync

# Or with pip
pip install -e .
```

### 2. Configure API Key

Export your OpenAI API key:

```bash
# OpenAI
export OPENAI_API_KEY='your-key-here'
```

### 3. Run the Agent

**Interactive Mode allowing ad-hoc queries** (default):
```bash
python demo.py
```

**Quick Demo of basic preset queries** (5 minutes):
```bash
python demo.py --scenario quick
```

## Usage

### Interactive CLI

The CLI supports natural language queries and special commands:

**Commands:**
- `/load <filepath> [alias]` - Load a CSV file
- `/list` - Show all loaded datasets
- `/clear` - Clear conversation history
- `/help` - Show help
- `/exit` - Exit the CLI

**Example Session:**
```
> /load data/foods3.csv foods
> What are the top 5 foods by calories?
> Show me a bar chart of foods that are high in vitamin D
> Show me foods that have close to 60mg of calcium and 7g of fiber
```

### Demo Datasets

Three sample datasets are included in `data/`:
- `foods3.csv` - 300 common foods with their amounts of 29 common nutrients
- `daily_values.csv` - The recommended daily values of the above nutrients
- `food_portion.csv` - The portion or "serving size" for foods. Note: this dataset will be more relevant to future features than this demo.

## Architecture

**Code-First Hybrid ReAct Agent:**
- **ReAct Loop**: Think → Act → Observe pattern for multi-step reasoning
- **Code Generation**: Dynamic pandas/matplotlib code for flexible analytics
- **Safe Execution**: RestrictedPython sandbox with AST validation and 5s timeout
- **Conversation Memory**: Context tracking across multiple turns

**Tools (8):**
1. `load_csv` - Load CSV files
2. `list_datasets` - View loaded datasets
3. `inspect_dataset` - Examine schema and statistics
4. `analyze` - Generate and execute pandas code
5. `visualize` - Generate and execute matplotlib/seaborn code
6. `high_in` - Find foods that are high in a particular nutrient, by z-score
7. `low_in` - Find foods that are low in a particular nutrient, by z-score
8. `target` - Find foods close to a specified amount of a nutrient

## Example Queries

**Data Exploration:**
- "What columns are in the foods dataset?"
- "Show me the first 5 rows"
- "Give me relevant statistics for all numeric columns"

**Analytics:**
- "Show top 10 foods by calories"
- "Calculate the average amounts of each nutrient"
- "Which foods are high in protein?"
- "Which foods are low in sodium?"

**Multi-Table:**
- "Show the percent daily value of calcium in Broccoli"

**Visualizations:**
- "Create a bar chart of the 5 foods with the most Vitamin C"
- "Show a scatter plot of calories vs protein"
- "Generate a bar chart of iron in beef versus chicken"

**Targeting:**
- "Show foods that are close to 10g of carbohydrates"
- "Show foods that are close to 5g of fat and 15g of protein"

## Project Structure

```
food_agent/
├── analytics_assistant/      # Main package
│   ├── agent.py              # ReAct orchestrator
│   ├── prompts.py            # LLM prompts
│   ├── executor.py           # Safe code execution
│   ├── memory.py             # Conversation memory
│   ├── cli.py                # Interactive CLI
│   └── tools/                # Tool implementations
│       ├── base.py           # BaseTool interface
│       ├── analytics.py      # Analysis & visualization
│       ├── dataset.py        # Dataset management
│       └── deterministic.py  # Deterministic, rule-based tools
│
├── data/                     # Food and nutrition datasets
│   ├── foods.csv             # List of foods with nutrients
│   ├── daily_values.py       # List of recommended daily values
├── outputs/                  # Generated visualizations
├── main.py                   # Main entry point for the application
└── README.md                 # This file
```

## Troubleshooting

**API Key Issues:**
```bash
# Check if key is set
echo $OPENAI_API_KEY

# Set it if missing
export OPENAI_API_KEY='sk-...'
```

**Missing Data:**
If demo data files are missing, they should be in the `data/` directory.

**Import Errors:**
```bash
# Reinstall dependencies
uv sync --force
```

**Execution Timeout:**
The safe executor has a 5-second timeout. Complex queries may need simplification.
