import pandas as pd

from agentbench.resource_orchestrator import (
    DEFAULT_NODES,
    compare_policies,
    estimate_task_latency,
    predictive_policy,
    simulate,
    static_policy,
    Task,
)


def sample_tasks():
    return pd.DataFrame(
        [
            {"task_id": "A1", "cpu_demand": 2.0, "memory_gb": 3.0, "tool_calls": 2, "latency_target_ms": 80, "priority": 2},
            {"task_id": "A2", "cpu_demand": 6.0, "memory_gb": 8.0, "tool_calls": 4, "latency_target_ms": 120, "priority": 1},
            {"task_id": "A3", "cpu_demand": 10.0, "memory_gb": 24.0, "tool_calls": 5, "latency_target_ms": 180, "priority": 1},
            {"task_id": "A4", "cpu_demand": 3.0, "memory_gb": 4.0, "tool_calls": 1, "latency_target_ms": 60, "priority": 3},
        ]
    )


def test_latency_is_finite_and_positive():
    task = Task("x", 2.0, 2.0, 2, 100)
    latency = estimate_task_latency(task, DEFAULT_NODES[0], predicted_load=0.2)
    assert latency > 0


def test_predictive_policy_returns_feasible_node():
    row = sample_tasks().iloc[0]
    task = Task(
        row.task_id, row.cpu_demand, row.memory_gb,
        row.tool_calls, row.latency_target_ms, row.priority
    )
    node = predictive_policy(task, DEFAULT_NODES, {"edge-1": 0.9, "cloud-1": 0.1})
    assert task.cpu_demand <= node.cpu_capacity
    assert task.memory_gb <= node.memory_gb


def test_simulation_produces_trace_and_summary():
    summary, trace = simulate(sample_tasks(), static_policy)
    assert summary.tasks == 4
    assert len(trace) == 4
    assert {"node", "latency_ms", "cpu_utilization", "sla_violation"}.issubset(trace.columns)


def test_policy_comparison_has_all_strategies():
    comparison = compare_policies(sample_tasks())
    assert set(comparison["policy"]) == {"static", "rule_based", "predictive"}
    assert (comparison["completion_rate"] >= 0).all()
    assert (comparison["completion_rate"] <= 1).all()
