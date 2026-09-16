import json
import math
import pandas as pd

WEIGHTS = {"accuracy": 0.45, "reliability": 0.25, "latency_s": 0.20, "client_peak_rss_mb": 0.10, "estimated_cost_usd": 0.0}


def summarize(experiment, hourly_cost_usd=None):
    rows = []
    for model in experiment["config"]["models"]:
        runs = [r for r in experiment["runs"] if r["model"] == model]
        frame = pd.DataFrame(runs)
        n = len(runs)
        correct = int(frame.correct.sum())
        p = correct / n
        # Wilson interval is descriptive: repeats of the same task are correlated.
        z = 1.96
        center = (p + z*z/(2*n)) / (1 + z*z/n)
        half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
        tools = [step["tool_ok"] for r in runs for step in r["trace"] if "tool_ok" in step]
        repeated = frame.groupby("task_id").correct
        latency = float(frame.latency_s.mean())
        rows.append(dict(model=model, provenance=experiment["mode"], runs=n,
            accuracy=p, accuracy_low=max(0, center-half), accuracy_high=min(1, center+half),
            reliability=float(repeated.all().mean()), completion_rate=float(frame.schema_valid.mean()),
            latency_s=latency, p95_latency_s=float(frame.latency_s.quantile(.95)),
            throughput_tasks_min=60/latency if latency else 0,
            client_peak_rss_mb=float(frame.client_peak_rss_mb.max()), client_cpu_s=float(frame.client_cpu_s.mean()),
            input_tokens=sum(r["input_tokens"] for r in runs) if all(r["input_tokens"] is not None for r in runs) else None,
            output_tokens=sum(r["output_tokens"] for r in runs) if all(r["output_tokens"] is not None for r in runs) else None,
            tool_success_rate=sum(tools)/len(tools) if tools else None,
            estimated_cost_usd=latency*hourly_cost_usd/3600 if hourly_cost_usd is not None else None))
    return pd.DataFrame(rows)


def recommend(summary, weights=None, min_accuracy=0, max_latency_s=None, max_client_memory_mb=None):
    weights = WEIGHTS if weights is None else weights
    if any(k not in WEIGHTS or not math.isfinite(v) or v < 0 for k, v in weights.items()) or sum(weights.values()) <= 0:
        raise ValueError("Use nonnegative finite weights with a positive total")
    if summary.provenance.nunique() != 1:
        raise ValueError("Cannot rank demo and real results together")
    frame = summary.copy()
    frame = frame[frame.accuracy >= min_accuracy]
    if max_latency_s is not None:
        frame = frame[frame.latency_s <= max_latency_s]
    if max_client_memory_mb is not None:
        frame = frame[frame.client_peak_rss_mb <= max_client_memory_mb]
    if frame.empty:
        return frame
    total = sum(weights.values())
    frame["score"] = 0.0
    for metric, weight in weights.items():
        if weight == 0:
            continue
        if frame[metric].isna().any():
            raise ValueError(f"{metric} is unknown; supply assumptions or set its weight to zero")
        values = frame[metric]
        span = values.max() - values.min()
        utility = (values-values.min())/span if span else pd.Series(1.0, index=frame.index)
        if metric not in ("accuracy", "reliability") and span:
            utility = 1 - utility
        frame[f"contribution_{metric}"] = utility * weight / total
        frame["score"] += frame[f"contribution_{metric}"]
    return frame.sort_values(["score", "model"], ascending=[False, True])


def export_json(experiment):
    return json.dumps(experiment, indent=2, allow_nan=False)


def export_csv(experiment):
    frame = pd.DataFrame(experiment["runs"])
    for column in ("answer", "expected", "trace"):
        frame[column] = frame[column].map(json.dumps)
    frame.insert(0, "experiment_id", experiment["id"])
    return frame.to_csv(index=False)
