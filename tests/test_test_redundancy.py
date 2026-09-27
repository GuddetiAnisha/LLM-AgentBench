import pandas as pd

from agentbench.test_redundancy import (
    find_redundant_pairs,
    greedy_requirement_cover,
    summarize_plan,
)


def _sample():
    return pd.DataFrame(
        [
            {"test_id": "T1", "requirements": "R1|R2", "cost": 10, "duration": 20, "metric_a": 1.00},
            {"test_id": "T2", "requirements": "R1|R2", "cost": 14, "duration": 28, "metric_a": 1.02},
            {"test_id": "T3", "requirements": "R2|R3", "cost": 12, "duration": 18, "metric_a": 2.00},
            {"test_id": "T4", "requirements": "R3|R4", "cost": 9, "duration": 16, "metric_a": 3.00},
        ]
    )


def test_redundant_pair_detected():
    pairs = find_redundant_pairs(_sample(), outcome_columns=["metric_a"])
    assert ((pairs["test_a"] == "T1") & (pairs["test_b"] == "T2")).any()


def test_greedy_cover_preserves_all_requirements():
    df = _sample()
    selected = greedy_requirement_cover(df)
    summary = summarize_plan(df, selected)
    assert summary.requirement_coverage == 1.0
    assert summary.selected_tests <= summary.original_tests


def test_summary_tracks_savings():
    df = _sample()
    selected = greedy_requirement_cover(df)
    summary = summarize_plan(df, selected)
    assert summary.selected_cost <= summary.original_cost
    assert summary.selected_duration <= summary.original_duration
