"""Allowlisted mathematical tools; no eval, shell, filesystem, or arbitrary code."""
import math

DESCRIPTIONS = {
    "subtract": "args: {a: number, b: number}; returns a - b",
    "absolute": "args: {value: number}; returns absolute value",
    "distance": "args: {a: [x,y,z], b: [x,y,z]}; returns Euclidean distance",
    "sum_abs": "args: {values: [numbers]}; returns sum of absolute values",
}


def execute(name, args):
    def number(x):
        if type(x) not in (int, float) or not math.isfinite(x) or abs(x) > 1e9:
            raise ValueError("Expected a finite number within +/- 1e9")
        return x

    def vector(x, length=None):
        if not isinstance(x, list) or not 1 <= len(x) <= 100 or (length and len(x) != length):
            raise ValueError("Invalid vector size")
        return [number(v) for v in x]

    if not isinstance(args, dict):
        raise ValueError("args must be an object")
    if name == "subtract":
        return number(args["a"]) - number(args["b"])
    if name == "absolute":
        return abs(number(args["value"]))
    if name == "distance":
        return math.dist(vector(args["a"], 3), vector(args["b"], 3))
    if name == "sum_abs":
        return sum(abs(v) for v in vector(args["values"]))
    raise ValueError(f"Tool not allowed: {name}")
