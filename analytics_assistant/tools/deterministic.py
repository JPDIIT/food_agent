"""
Deterministic Tools

Tools that implement deterministic, rule-based operations on loaded datasets.

This module provides three tools:
- HighInTool: Find foods that are abnormally high in one or more nutrients.
- LowInTool: Find foods that are abnormally low in one or more nutrients.
- TargetTool: Find foods close to a targeted amount for given nutrients.

All tools operate on datasets loaded via the dataset tools and return
structured results that the agent can present to the user.
"""

from typing import Any, Dict, List
import pandas as pd

from .base import BaseTool
from .dataset import get_dataset


def _match_columns(df: pd.DataFrame, requested: List[str]) -> Dict[str, str]:
    """Match requested nutrient names to DataFrame columns (case-insensitive).

    Returns a mapping from normalized requested name -> actual column name in df.
    """
    cols = {c.lower(): c for c in df.columns}
    mapping = {}
    for name in requested:
        lname = name.lower()
        if lname in cols:
            mapping[name] = cols[lname]
        else:
            # try substring match
            found = None
            for c_low, c in cols.items():
                if lname in c_low or c_low in lname:
                    found = c
                    break
            if found:
                mapping[name] = found
    return mapping


class HighInTool(BaseTool):
    """Find foods with abnormally high nutrient values.

    Definition of "abnormally high" uses a simple z-score rule by default:
    value > mean + (z_thresh * std). This is deterministic and fast.
    """

    @property
    def name(self) -> str:
        return "high_in"

    @property
    def description(self) -> str:
        return (
            "Find items in a dataset that are abnormally high for one or more nutrients. "
            "Provide a list of nutrient names. Returns matching rows and summary stats."
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "alias": {"type": "string", "description": "Dataset alias (e.g., 'foods')"},
                "nutrients": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of nutrient column names to check"
                },
                "z_thresh": {"type": "number", "description": "Z-score threshold (default: 2.0)", "default": 2.0},
                "top_n": {"type": "integer", "description": "Number of top matches to return per nutrient", "default": 10}
            },
            "required": ["alias", "nutrients"]
        }

    def execute(self, **kwargs) -> Dict[str, Any]:
        alias = kwargs.get("alias")
        nutrients = kwargs.get("nutrients", [])
        z_thresh = float(kwargs.get("z_thresh", 2.0))
        top_n = int(kwargs.get("top_n", 10))

        if not alias:
            return {"status": "error", "result": None, "error_message": "alias is required"}

        try:
            df = get_dataset(alias)
        except KeyError as e:
            return {"status": "error", "result": None, "error_message": str(e)}

        col_map = _match_columns(df, nutrients)
        if not col_map:
            return {"status": "error", "result": None, "error_message": "No matching nutrient columns found"}

        findings = {}
        for req_name, col in col_map.items():
            try:
                series = pd.to_numeric(df[col], errors='coerce').dropna()
            except Exception:
                findings[req_name] = {"error": f"Failed to parse column {col} as numeric"}
                continue

            mean = series.mean()
            std = series.std()
            thresh = mean + z_thresh * std if pd.notna(std) else mean

            mask = pd.to_numeric(df[col], errors='coerce') > thresh
            matches = df.loc[mask].copy()
            count = int(matches.shape[0])

            top_matches = matches.sort_values(by=col, ascending=False).head(top_n)

            findings[req_name] = {
                "column": col,
                "count": count,
                "threshold": float(round(thresh, 4)) if pd.notna(thresh) else None,
                "mean": float(round(mean, 4)) if pd.notna(mean) else None,
                "std": float(round(std, 4)) if pd.notna(std) else None,
                "top_matches": top_matches.head(top_n).to_dict(orient='records')
            }

        result_msg = f"High-in analysis for alias '{alias}' completed." 
        return {"status": "success", "result": result_msg, "data": findings}


class LowInTool(BaseTool):
    """Find foods with abnormally low nutrient values.

    Definition uses z-score rule: value < mean - (z_thresh * std).
    """

    @property
    def name(self) -> str:
        return "low_in"

    @property
    def description(self) -> str:
        return (
            "Find items in a dataset that are abnormally low for one or more nutrients. "
            "Provide a list of nutrient names. Returns matching rows and summary stats."
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "alias": {"type": "string"},
                "nutrients": {"type": "array", "items": {"type": "string"}},
                "z_thresh": {"type": "number", "default": 2.0},
                "top_n": {"type": "integer", "default": 10}
            },
            "required": ["alias", "nutrients"]
        }

    def execute(self, **kwargs) -> Dict[str, Any]:
        alias = kwargs.get("alias")
        nutrients = kwargs.get("nutrients", [])
        z_thresh = float(kwargs.get("z_thresh", 2.0))
        top_n = int(kwargs.get("top_n", 10))

        if not alias:
            return {"status": "error", "result": None, "error_message": "alias is required"}

        try:
            df = get_dataset(alias)
        except KeyError as e:
            return {"status": "error", "result": None, "error_message": str(e)}

        col_map = _match_columns(df, nutrients)
        if not col_map:
            return {"status": "error", "result": None, "error_message": "No matching nutrient columns found"}

        findings = {}
        for req_name, col in col_map.items():
            try:
                series = pd.to_numeric(df[col], errors='coerce').dropna()
            except Exception:
                findings[req_name] = {"error": f"Failed to parse column {col} as numeric"}
                continue

            mean = series.mean()
            std = series.std()
            thresh = mean - z_thresh * std if pd.notna(std) else mean

            mask = pd.to_numeric(df[col], errors='coerce') < thresh
            matches = df.loc[mask].copy()
            count = int(matches.shape[0])

            top_matches = matches.sort_values(by=col, ascending=True).head(top_n)

            findings[req_name] = {
                "column": col,
                "count": count,
                "threshold": float(round(thresh, 4)) if pd.notna(thresh) else None,
                "mean": float(round(mean, 4)) if pd.notna(mean) else None,
                "std": float(round(std, 4)) if pd.notna(std) else None,
                "top_matches": top_matches.head(top_n).to_dict(orient='records')
            }

        result_msg = f"Low-in analysis for alias '{alias}' completed." 
        return {"status": "success", "result": result_msg, "data": findings}


class TargetTool(BaseTool):
    """Find foods close to a target amount for given nutrients.

    Parameters:
    - nutrients: list of nutrient names
    - amount: numeric target (applied per nutrient or single value)
    - tol: allowable absolute tolerance (or range) around amount
    """

    @property
    def name(self) -> str:
        return "target"

    @property
    def description(self) -> str:
        return (
            "Find items whose nutrient values are within a target range. "
            "Useful for finding foods that match a target nutrient profile."
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "alias": {"type": "string"},
                "nutrients": {"type": "array", "items": {"type": "string"}},
                "amount": {"type": "number"},
                "tol": {"type": "number", "description": "Absolute tolerance (default: 5.0)", "default": 5.0},
                "max_results": {"type": "integer", "default": 20}
            },
            "required": ["alias", "nutrients", "amount"]
        }

    def execute(self, **kwargs) -> Dict[str, Any]:
        alias = kwargs.get("alias")
        nutrients = kwargs.get("nutrients", [])
        amount = kwargs.get("amount")
        tol = float(kwargs.get("tol", 5.0))
        max_results = int(kwargs.get("max_results", 20))

        if not alias:
            return {"status": "error", "result": None, "error_message": "alias is required"}
        if amount is None:
            return {"status": "error", "result": None, "error_message": "amount is required"}

        try:
            df = get_dataset(alias)
        except KeyError as e:
            return {"status": "error", "result": None, "error_message": str(e)}

        col_map = _match_columns(df, nutrients)
        if not col_map:
            return {"status": "error", "result": None, "error_message": "No matching nutrient columns found"}

        # Build a score = sum of absolute differences across nutrients
        working = df.copy()
        diffs = []
        for req_name, col in col_map.items():
            working[f"_diff_{col}"] = (pd.to_numeric(working[col], errors='coerce') - float(amount)).abs()
            diffs.append(f"_diff_{col}")

        # Drop rows with all NaN diffs
        working = working.dropna(subset=diffs, how='all')

        # Find rows where all diffs are within tolerance
        within_mask = working[diffs].le(tol).all(axis=1)
        matches = working.loc[within_mask].copy()

        # Additionally compute a combined score for ranking
        working['_score'] = working[diffs].sum(axis=1)
        ranked = working.sort_values('_score').head(max_results)

        result_data = {
            "requested_nutrients": list(col_map.keys()),
            "column_mapping": col_map,
            "matches_within_tolerance_count": int(matches.shape[0]),
            "matches_within_tolerance": matches.head(max_results).to_dict(orient='records'),
            "top_ranked_by_score": ranked.head(max_results).to_dict(orient='records')
        }

        result_msg = f"Target search for amount={amount} ±{tol} on alias '{alias}' completed." 
        return {"status": "success", "result": result_msg, "data": result_data}
