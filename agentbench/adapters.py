import json
import requests


class OllamaAdapter:
    provenance = "ollama"

    def __init__(self, model, base_url="http://localhost:11434", timeout=60):
        self.model, self.base_url, self.timeout = model, base_url.rstrip("/"), timeout

    def chat(self, messages, seed=42):
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "format": "json",
            # Qwen3 and other reasoning models can spend the whole generation
            # budget on hidden/visible thinking and return an empty content
            # string. AgentBench needs the JSON action itself, so reasoning is
            # disabled for deterministic structured benchmark calls.
            "think": False,
            "options": {
                "temperature": 0,
                "seed": seed,
                "num_predict": 512,
            },
        }
        response = requests.post(
            self.base_url + "/api/chat",
            json=payload,
            timeout=(5, self.timeout),
        )
        response.raise_for_status()
        body = response.json()

        message = body.get("message") or {}
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(
                "Ollama returned an empty response. Ensure the model supports "
                "non-thinking structured output and that Ollama is up to date."
            )

        # Validate here as well as in agent.py so malformed model output is
        # reported clearly as an adapter/structured-output failure.
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            preview = content[:200].replace("\n", " ")
            raise ValueError(
                f"Ollama returned invalid JSON: {exc}; response={preview!r}"
            ) from exc
        if not isinstance(parsed, dict):
            raise ValueError("Ollama JSON response must be an object")

        return (
            json.dumps(parsed, separators=(",", ":")),
            body.get("prompt_eval_count"),
            body.get("eval_count"),
        )


class DemoAdapter:
    """Scripted agent exercises tools; never presented as an actual LLM."""
    provenance = "demo"

    def __init__(self, model="demo-exact"):
        if model not in ("demo-exact", "demo-imperfect"):
            raise ValueError("Unknown demo profile")
        self.model = model

    def chat(self, messages, seed=42):
        task = json.loads(messages[1]["content"])
        d, kind = task["data"], task["kind"]
        results = [json.loads(m["content"])["result"] for m in messages[2:] if m["role"] == "user" and "result" in json.loads(m["content"])]
        if not results:
            if kind == "dimension":
                tool, args = "subtract", {"a": d["measured"], "b": d["nominal"]}
            elif kind == "position":
                tool, args = "distance", {"a": d["reference"], "b": d["measured"]}
            elif kind == "stack":
                tool, args = "sum_abs", {"values": d["tolerances"]}
            else:
                tool, args = "subtract", {"a": d["hole"], "b": d["shaft"]}
            answer = {"tool": tool, "args": args}
        else:
            value = abs(results[-1]) if kind == "dimension" else results[-1]
            passed = value >= d["minimum"] - 1e-12 if kind == "clearance" else value <= d.get("tolerance", d.get("limit")) + 1e-12
            if self.model == "demo-imperfect" and (int(task["id"].split("_")[-1]) + seed) % 3 == 0:
                passed = not passed
            answer = {"value": value, "passed": passed, "unit": "mm", "explanation": "Scripted demo using an allowlisted arithmetic tool."}
        return json.dumps(answer), None, None
