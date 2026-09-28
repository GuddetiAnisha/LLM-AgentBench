import pandas as pd

from agentbench.red_blue_sim import (
    Scenario,
    benchmark_policies,
    cautious_red_policy,
    evaluate_matchup,
    rule_blue_policy,
    rule_red_policy,
    run_episode,
    scenarios_from_frame,
)


def sample_scenarios():
    return [
        Scenario("RB1", 1, 0.30, 0.40, 0.80),
        Scenario("RB2", 2, 0.45, 0.35, 0.70),
        Scenario("RB3", 3, 0.55, 0.30, 0.60),
    ]


def test_episode_produces_trace():
    summary, trace = run_episode(sample_scenarios()[0])
    assert summary.steps >= 1
    assert not trace.empty
    assert {"red_action", "blue_action", "detected", "contained"}.issubset(trace.columns)


def test_rule_blue_detects_or_contains_activity():
    summary, _ = run_episode(sample_scenarios()[0], rule_red_policy, rule_blue_policy)
    assert summary.detected or summary.contained


def test_evaluation_metrics_are_bounded():
    metrics = evaluate_matchup(sample_scenarios(), cautious_red_policy, rule_blue_policy)
    for key in [
        "red_success_rate",
        "detection_rate",
        "containment_rate",
        "recovery_rate",
        "false_positive_rate",
    ]:
        assert 0.0 <= metrics[key] <= 1.0


def test_policy_benchmark_contains_all_matchups():
    table = benchmark_policies(sample_scenarios())
    assert len(table) == 4
    assert "mean_response_step" in table.columns


def test_scenarios_from_frame():
    frame = pd.DataFrame(
        [
            {
                "scenario_id": "X",
                "difficulty": 2,
                "detection_threshold": 0.4,
                "containment_strength": 0.3,
                "recovery_speed": 0.7,
            }
        ]
    )
    scenarios = scenarios_from_frame(frame)
    assert scenarios[0].scenario_id == "X"
