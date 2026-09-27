"""Test-plan redundancy analysis for experiment portfolios.

This module is a software-only extension for studying how historical experiment
metadata can be used to identify overlapping tests and construct a smaller test
portfolio while preserving explicit requirement coverage.

It is intentionally generic: it does not claim to model Volvo Penta test
facilities, legislation, physical rigs, or proprietary test data.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import math
import pandas as pd


@dataclass(frozen=True)
class PlanSummary:
    original_tests: int
    selected_tests: int
    removed_tests: int
    total_requirements: int
    covered_requirements: int
    requirement_coverage: float
    original_cost: float
    selected_cost: float
    original_duration: float
    selected_duration: float

    @property
    def test_reduction(self) -> float:
        if self.original_tests == 0:
            return 0.0
        return 1.0 - self.selected_tests / self.original_tests


def _token_set(value) -> set[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return set()
    if isinstance(value, str):
        return {x.strip() for x in value.replace(",", "|").split("|") if x.strip()}
    if isinstance(value, Iterable):
        return {str(x).strip() for x in value if str(x).strip()}
    return {str(value).strip()}


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    a_set, b_set = set(a), set(b)
    union = a_set | b_set
    if not union:
        return 1.0
    return len(a_set & b_set) / len(union)


def numeric_similarity(row_a: pd.Series, row_b: pd.Series, columns: Sequence[str]) -> float:
    """Similarity in [0,1] using scale-aware absolute differences."""
    if not columns:
        return 1.0

    scores = []
    for col in columns:
        a = float(row_a[col])
        b = float(row_b[col])
        scale = max(abs(a), abs(b), 1.0)
        scores.append(max(0.0, 1.0 - abs(a - b) / scale))
    return float(sum(scores) / len(scores))


def find_redundant_pairs(
    tests: pd.DataFrame,
    outcome_columns: Sequence[str] = (),
    requirement_threshold: float = 0.80,
    outcome_threshold: float = 0.90,
) -> pd.DataFrame:
    """Find pairs with highly overlapping requirements and similar outcomes."""
    required = {"test_id", "requirements"}
    missing = required - set(tests.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")

    rows = []
    data = tests.reset_index(drop=True)
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            a, b = data.iloc[i], data.iloc[j]
            req_sim = jaccard(_token_set(a["requirements"]), _token_set(b["requirements"]))
            out_sim = numeric_similarity(a, b, outcome_columns)
            if req_sim >= requirement_threshold and out_sim >= outcome_threshold:
                rows.append(
                    {
                        "test_a": str(a["test_id"]),
                        "test_b": str(b["test_id"]),
                        "requirement_similarity": round(req_sim, 4),
                        "outcome_similarity": round(out_sim, 4),
                    }
                )

    return pd.DataFrame(
        rows,
        columns=["test_a", "test_b", "requirement_similarity", "outcome_similarity"],
    )


def greedy_requirement_cover(
    tests: pd.DataFrame,
    requirement_col: str = "requirements",
    cost_col: str = "cost",
    duration_col: str = "duration",
    cost_weight: float = 0.5,
    duration_weight: float = 0.5,
) -> list[str]:
    """Select a compact test set while preserving all explicit requirements.

    At each step, choose the test with the largest uncovered-requirement gain
    per normalized cost/duration burden. This is a transparent greedy heuristic,
    not proof of a mathematically minimal test set.
    """
    required = {"test_id", requirement_col, cost_col, duration_col}
    missing = required - set(tests.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    if cost_weight < 0 or duration_weight < 0 or cost_weight + duration_weight <= 0:
        raise ValueError("cost and duration weights must be nonnegative and not both zero")

    data = tests.copy()
    data["_reqs"] = data[requirement_col].map(_token_set)
    universe = set().union(*data["_reqs"].tolist()) if len(data) else set()
    uncovered = set(universe)
    selected: list[str] = []

    max_cost = max(float(data[cost_col].max()), 1.0) if len(data) else 1.0
    max_duration = max(float(data[duration_col].max()), 1.0) if len(data) else 1.0

    while uncovered:
        best_idx = None
        best_score = -1.0
        best_gain = -1

        for idx, row in data.iterrows():
            test_id = str(row["test_id"])
            if test_id in selected:
                continue
            gain = len(row["_reqs"] & uncovered)
            if gain == 0:
                continue

            burden = (
                cost_weight * float(row[cost_col]) / max_cost
                + duration_weight * float(row[duration_col]) / max_duration
            )
            score = gain / max(burden, 1e-9)

            if score > best_score or (score == best_score and gain > best_gain):
                best_idx = idx
                best_score = score
                best_gain = gain

        if best_idx is None:
            break

        chosen = data.loc[best_idx]
        chosen_id = str(chosen["test_id"])
        selected.append(chosen_id)
        uncovered -= chosen["_reqs"]

    if uncovered:
        raise ValueError(f"requirements cannot be covered: {sorted(uncovered)}")
    return selected


def summarize_plan(
    tests: pd.DataFrame,
    selected_ids: Sequence[str],
    requirement_col: str = "requirements",
    cost_col: str = "cost",
    duration_col: str = "duration",
) -> PlanSummary:
    data = tests.copy()
    data["_reqs"] = data[requirement_col].map(_token_set)
    original_requirements = set().union(*data["_reqs"].tolist()) if len(data) else set()
    selected = data[data["test_id"].astype(str).isin({str(x) for x in selected_ids})]
    selected_requirements = (
        set().union(*selected["_reqs"].tolist()) if len(selected) else set()
    )

    return PlanSummary(
        original_tests=len(data),
        selected_tests=len(selected),
        removed_tests=len(data) - len(selected),
        total_requirements=len(original_requirements),
        covered_requirements=len(selected_requirements),
        requirement_coverage=(
            len(selected_requirements) / len(original_requirements)
            if original_requirements
            else 1.0
        ),
        original_cost=float(data[cost_col].sum()) if len(data) else 0.0,
        selected_cost=float(selected[cost_col].sum()) if len(selected) else 0.0,
        original_duration=float(data[duration_col].sum()) if len(data) else 0.0,
        selected_duration=float(selected[duration_col].sum()) if len(selected) else 0.0,
    )
