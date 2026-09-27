"""Resource orchestration simulation for agentic AI workloads.

This software-only extension studies monitoring, prediction, and adaptive resource
allocation for synthetic agentic AI workloads across abstract edge and cloud nodes.

It does not control real Kubernetes clusters, cloud accounts, production edge
devices, or Ericsson systems. Resource and latency values are simulation inputs
for reproducible experiments.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import math
import pandas as pd


@dataclass(frozen=True)
class Node:
    name: str
    cpu_capacity: float
    memory_gb: float
    base_latency_ms: float
    cost_per_cpu_s: float
    kind: str


@dataclass(frozen=True)
class Task:
    task_id: str
    cpu_demand: float
    memory_gb: float
    tool_calls: int
    latency_target_ms: float
    priority: int = 1


@dataclass(frozen=True)
class SimulationSummary:
    tasks: int
    completed: int
    completion_rate: float
    mean_latency_ms: float
    p95_latency_ms: float
    sla_violations: int
    mean_cpu_utilization: float
    mean_memory_utilization: float
    total_cost: float
    edge_assignments: int
    cloud_assignments: int


DEFAULT_NODES = (
    Node("edge-1", cpu_capacity=8.0, memory_gb=16.0, base_latency_ms=8.0,
         cost_per_cpu_s=0.00010, kind="edge"),
    Node("cloud-1", cpu_capacity=24.0, memory_gb=64.0, base_latency_ms=28.0,
         cost_per_cpu_s=0.00006, kind="cloud"),
)


def _validate_nodes(nodes: Sequence[Node]) -> None:
    if not nodes:
        raise ValueError("at least one node is required")
    for node in nodes:
        if node.cpu_capacity <= 0 or node.memory_gb <= 0:
            raise ValueError("node capacities must be positive")
        if node.kind not in {"edge", "cloud"}:
            raise ValueError("node kind must be 'edge' or 'cloud'")


def _task_from_row(row: pd.Series) -> Task:
    return Task(
        task_id=str(row["task_id"]),
        cpu_demand=float(row["cpu_demand"]),
        memory_gb=float(row["memory_gb"]),
        tool_calls=int(row["tool_calls"]),
        latency_target_ms=float(row["latency_target_ms"]),
        priority=int(row.get("priority", 1)),
    )


def estimate_task_latency(task: Task, node: Node, predicted_load: float = 0.0) -> float:
    """Estimate end-to-end latency for a task-node pairing.

    The model combines base network latency, CPU pressure, memory pressure,
    tool-call overhead, and predicted background load.
    """
    cpu_ratio = task.cpu_demand / node.cpu_capacity
    mem_ratio = task.memory_gb / node.memory_gb
    pressure = max(cpu_ratio, mem_ratio, 0.0)

    compute_ms = 35.0 * cpu_ratio + 18.0 * mem_ratio
    tool_ms = 6.0 * max(task.tool_calls, 0)
    queue_ms = 55.0 * max(0.0, predicted_load + pressure - 0.85) ** 2

    return float(node.base_latency_ms + compute_ms + tool_ms + queue_ms)


def can_host(task: Task, node: Node) -> bool:
    return task.cpu_demand <= node.cpu_capacity and task.memory_gb <= node.memory_gb


def static_policy(task: Task, nodes: Sequence[Node], predicted_loads=None) -> Node:
    """Always choose the first node that can host the task."""
    for node in nodes:
        if can_host(task, node):
            return node
    return nodes[-1]


def rule_based_policy(task: Task, nodes: Sequence[Node], predicted_loads=None) -> Node:
    """Prefer edge for latency-sensitive tasks; otherwise choose lower cost."""
    feasible = [n for n in nodes if can_host(task, n)]
    if not feasible:
        return nodes[-1]

    if task.latency_target_ms <= 80:
        edge = [n for n in feasible if n.kind == "edge"]
        if edge:
            return min(edge, key=lambda n: n.base_latency_ms)

    return min(feasible, key=lambda n: (n.cost_per_cpu_s, n.base_latency_ms))


def predictive_policy(task: Task, nodes: Sequence[Node], predicted_loads=None) -> Node:
    """Select the feasible node with the best latency/cost/utilization score."""
    predicted_loads = predicted_loads or {}
    feasible = [n for n in nodes if can_host(task, n)]
    if not feasible:
        return nodes[-1]

    scored = []
    for node in feasible:
        load = float(predicted_loads.get(node.name, 0.0))
        latency = estimate_task_latency(task, node, predicted_load=load)
        cpu_util = task.cpu_demand / node.cpu_capacity
        mem_util = task.memory_gb / node.memory_gb
        sla_penalty = max(0.0, latency - task.latency_target_ms)
        cost = node.cost_per_cpu_s * task.cpu_demand * max(latency / 1000.0, 1e-6)
        score = (
            latency
            + 3.0 * sla_penalty
            + 35.0 * load
            + 8.0 * max(cpu_util, mem_util)
            + 5000.0 * cost
        )
        scored.append((score, node.name, node))

    return min(scored, key=lambda x: (x[0], x[1]))[2]


def moving_average_predict(history: Sequence[float], window: int = 5) -> float:
    if not history:
        return 0.0
    values = [float(x) for x in history[-window:]]
    return float(sum(values) / len(values))


def simulate(
    tasks: pd.DataFrame,
    policy,
    nodes: Sequence[Node] = DEFAULT_NODES,
) -> tuple[SimulationSummary, pd.DataFrame]:
    """Run a sequential orchestration simulation and return summary + trace."""
    _validate_nodes(nodes)

    required = {
        "task_id", "cpu_demand", "memory_gb", "tool_calls",
        "latency_target_ms"
    }
    missing = required - set(tasks.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")

    load_history = {node.name: [] for node in nodes}
    rows = []

    for _, row in tasks.iterrows():
        task = _task_from_row(row)
        predicted = {
            node.name: moving_average_predict(load_history[node.name])
            for node in nodes
        }

        node = policy(task, nodes, predicted)
        feasible = can_host(task, node)
        predicted_load = predicted[node.name]
        latency = estimate_task_latency(task, node, predicted_load)

        cpu_util = task.cpu_demand / node.cpu_capacity
        mem_util = task.memory_gb / node.memory_gb
        load = min(1.5, max(cpu_util, mem_util))
        load_history[node.name].append(load)

        completed = bool(feasible)
        sla_violation = completed and latency > task.latency_target_ms
        duration_s = latency / 1000.0
        cost = (
            node.cost_per_cpu_s * task.cpu_demand * duration_s
            if completed else 0.0
        )

        rows.append(
            {
                "task_id": task.task_id,
                "priority": task.priority,
                "node": node.name,
                "node_kind": node.kind,
                "predicted_load": round(predicted_load, 4),
                "cpu_utilization": round(cpu_util, 4),
                "memory_utilization": round(mem_util, 4),
                "latency_ms": round(latency, 4),
                "latency_target_ms": task.latency_target_ms,
                "sla_violation": bool(sla_violation),
                "completed": completed,
                "cost": float(cost),
            }
        )

    trace = pd.DataFrame(rows)

    if len(trace) == 0:
        summary = SimulationSummary(
            0, 0, 1.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0, 0
        )
        return summary, trace

    completed = trace[trace["completed"]]
    latencies = completed["latency_ms"] if len(completed) else pd.Series(dtype=float)

    summary = SimulationSummary(
        tasks=len(trace),
        completed=int(trace["completed"].sum()),
        completion_rate=float(trace["completed"].mean()),
        mean_latency_ms=float(latencies.mean()) if len(latencies) else 0.0,
        p95_latency_ms=float(latencies.quantile(0.95)) if len(latencies) else 0.0,
        sla_violations=int(trace["sla_violation"].sum()),
        mean_cpu_utilization=float(trace["cpu_utilization"].mean()),
        mean_memory_utilization=float(trace["memory_utilization"].mean()),
        total_cost=float(trace["cost"].sum()),
        edge_assignments=int((trace["node_kind"] == "edge").sum()),
        cloud_assignments=int((trace["node_kind"] == "cloud").sum()),
    )
    return summary, trace


def compare_policies(
    tasks: pd.DataFrame,
    nodes: Sequence[Node] = DEFAULT_NODES,
) -> pd.DataFrame:
    rows = []
    for name, policy in (
        ("static", static_policy),
        ("rule_based", rule_based_policy),
        ("predictive", predictive_policy),
    ):
        summary, _ = simulate(tasks, policy, nodes)
        rows.append(
            {
                "policy": name,
                "tasks": summary.tasks,
                "completion_rate": summary.completion_rate,
                "mean_latency_ms": summary.mean_latency_ms,
                "p95_latency_ms": summary.p95_latency_ms,
                "sla_violations": summary.sla_violations,
                "mean_cpu_utilization": summary.mean_cpu_utilization,
                "mean_memory_utilization": summary.mean_memory_utilization,
                "total_cost": summary.total_cost,
                "edge_assignments": summary.edge_assignments,
                "cloud_assignments": summary.cloud_assignments,
            }
        )
    return pd.DataFrame(rows)
