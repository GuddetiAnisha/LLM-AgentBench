"""Synthetic tasks and an independent reference evaluator. All distances are mm."""
import math
import random

KINDS = ("dimension", "position", "stack", "clearance")


def generate_tasks(count=24, seed=42):
    if not 1 <= count <= 1000:
        raise ValueError("Task count must be 1–1000")
    rng = random.Random(seed)
    tasks = []
    for i in range(count):
        kind = KINDS[i % len(KINDS)]
        tolerance = rng.choice([0.1, 0.2, 0.5])
        if kind == "dimension":
            nominal = rng.choice([10, 25, 50])
            data = dict(nominal=nominal, measured=round(nominal + rng.uniform(-0.7, 0.7), 4), tolerance=tolerance)
            question = "Compute absolute dimensional deviation; pass when deviation <= tolerance."
        elif kind == "position":
            data = dict(reference=[10, 20, 30], measured=[round(v + rng.uniform(-0.3, 0.3), 4) for v in [10, 20, 30]], tolerance=tolerance)
            question = "Compute Euclidean 3D point displacement; pass when displacement <= tolerance. This is not GD&T true position."
        elif kind == "stack":
            data = dict(tolerances=[round(rng.uniform(0.01, 0.15), 4) for _ in range(4)], limit=tolerance)
            question = "Compute worst-case stack half-width as sum of absolute bilateral tolerances; pass when <= limit."
        else:
            data = dict(hole=round(rng.uniform(10, 10.4), 4), shaft=round(rng.uniform(9.9, 10.3), 4), minimum=0.05)
            question = "Compute signed clearance hole minus shaft; pass when clearance >= minimum."
        tasks.append(dict(id=f"GA_{seed}_{i:04d}", kind=kind, data=data, question=question, unit="mm"))
    return tasks


def ground_truth(task):
    d, kind = task["data"], task["kind"]
    if kind == "dimension":
        value = abs(d["measured"] - d["nominal"])
        passed = value <= d["tolerance"] + 1e-12
    elif kind == "position":
        value = math.dist(d["reference"], d["measured"])
        passed = value <= d["tolerance"] + 1e-12
    elif kind == "stack":
        value = math.fsum(abs(x) for x in d["tolerances"])
        passed = value <= d["limit"] + 1e-12
    elif kind == "clearance":
        value = d["hole"] - d["shaft"]
        passed = value >= d["minimum"] - 1e-12
    else:
        raise ValueError("Unknown task kind")
    return {"value": value, "passed": passed, "unit": "mm"}


def evaluate(task, answer):
    expected = ground_truth(task)
    value = answer.get("value") if isinstance(answer, dict) else None
    try:
        finite = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        finite = False
    valid = (isinstance(answer, dict) and type(answer.get("passed")) is bool
             and finite and answer.get("unit") == "mm")
    numeric = bool(valid and math.isclose(answer["value"], expected["value"], abs_tol=1e-4, rel_tol=1e-4))
    decision = bool(valid and answer["passed"] == expected["passed"])
    return dict(schema_valid=valid, numeric_correct=numeric, decision_correct=decision,
                correct=numeric and decision, expected=expected)
