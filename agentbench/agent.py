import json
from .tools import DESCRIPTIONS, execute

SYSTEM = '''Solve the synthetic engineering task. Return only a JSON object.
Either request one tool with {"tool": "name", "args": {...}}, or finish with
{"value": number, "passed": boolean, "unit": "mm", "explanation": "brief justification"}.
Do not round intermediate results. A final answer is mandatory. Available tools: ''' + json.dumps(DESCRIPTIONS)


def strict_json(raw):
    def reject(value):
        raise ValueError(f"Non-finite JSON number: {value}")
    def finite_float(value):
        import math
        result = float(value)
        if not math.isfinite(result):
            reject(value)
        return result
    return json.loads(raw, parse_constant=reject, parse_float=finite_float)


def run_agent(adapter, task, seed=42, max_steps=5):
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": json.dumps(task)}]
    trace, inputs, outputs = [], [], []
    answer, error = None, None
    for step in range(max_steps):
        try:
            raw, inp, out = adapter.chat(messages, seed)
            inputs.append(inp)
            outputs.append(out)
            trace.append({"step": step, "response": raw})
            action = strict_json(raw)
            if not isinstance(action, dict):
                raise ValueError("Response must be a JSON object")
            if "tool" not in action:
                answer = action
                break
            messages.append({"role": "assistant", "content": raw})
            try:
                result = execute(action["tool"], action.get("args"))
                feedback = {"result": result}
                trace[-1]["tool_ok"] = True
            except (ValueError, KeyError, TypeError, OverflowError) as exc:
                feedback = {"error": str(exc)}
                trace[-1]["tool_ok"] = False
            trace[-1]["feedback"] = feedback
            messages.append({"role": "user", "content": json.dumps(feedback)})
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            break
    if answer is None and error is None:
        error = "Agent step budget exhausted"
    return dict(answer=answer, error=error, trace=trace,
                input_tokens=sum(inputs) if inputs and all(x is not None for x in inputs) else None,
                output_tokens=sum(outputs) if outputs and all(x is not None for x in outputs) else None)
