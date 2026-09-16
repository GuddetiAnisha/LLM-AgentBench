import json
import requests


class OllamaAdapter:
    provenance = "ollama"

    def __init__(self, model, base_url="http://localhost:11434", timeout=60):
        self.model, self.base_url, self.timeout = model, base_url.rstrip("/"), timeout

    def chat(self, messages, seed=42):
        response = requests.post(self.base_url + "/api/chat", json={
            "model": self.model, "messages": messages, "stream": False, "format": "json",
            "options": {"temperature": 0, "seed": seed, "num_predict": 512},
        }, timeout=(5, self.timeout))
        response.raise_for_status()
        body = response.json()
        return body["message"]["content"], body.get("prompt_eval_count"), body.get("eval_count")


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
