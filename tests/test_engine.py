import json
from unittest.mock import Mock
import pytest
import requests
from agentbench.adapters import DemoAdapter, OllamaAdapter
from agentbench.agent import run_agent
from agentbench.analysis import export_csv, export_json, recommend, summarize
from agentbench.engine import benchmark
from agentbench.storage import Store
from agentbench.tasks import generate_tasks


@pytest.fixture
def experiment():
    return benchmark([DemoAdapter(), DemoAdapter("demo-imperfect")], count=8, repeats=2)


def test_end_to_end(experiment, tmp_path):
    assert len(experiment["runs"]) == 32
    summary = summarize(experiment).set_index("model")
    assert summary.loc["demo-exact", "accuracy"] == 1
    assert 0 < summary.loc["demo-imperfect", "accuracy"] < 1
    assert summary.loc["demo-exact", "tool_success_rate"] == 1
    assert summary.loc["demo-exact", "input_tokens"] is None
    assert (summary.latency_s >= 0).all()
    store = Store(tmp_path / "runs.sqlite")
    store.save(experiment)
    assert store.load(experiment["id"]) == experiment
    assert len(store.history()) == 1
    with pytest.raises(KeyError):
        store.load("' OR 1=1 --")
    assert json.loads(export_json(experiment)) == experiment
    assert "demo-exact" in export_csv(experiment)
    assert "provenance" in export_csv(experiment)


def test_recommendations(experiment):
    summary = summarize(experiment)
    ranked = recommend(summary, {"accuracy": 1})
    assert ranked.iloc[0].model == "demo-exact"
    assert recommend(summary, min_accuracy=1).shape[0] == 1
    assert recommend(summary, max_latency_s=0).empty
    with pytest.raises(ValueError, match="unknown"):
        recommend(summary, {"estimated_cost_usd": 1})
    with pytest.raises(ValueError):
        recommend(summary, {"accuracy": 0})
    assert not recommend(summarize(experiment, 2), {"estimated_cost_usd": 1}).empty


def test_no_mixing():
    with pytest.raises(ValueError, match="separate"):
        benchmark([DemoAdapter(), OllamaAdapter("example")], 4, 1)


def test_failure_is_recorded_without_fallback(monkeypatch):
    def fail(*a, **kw):
        raise requests.Timeout("test timeout")
    monkeypatch.setattr(requests, "post", fail)
    result = benchmark([OllamaAdapter("offline")], 4, 1)
    assert result["mode"] == "ollama"
    assert all(not r["correct"] and "Timeout" in r["error"] for r in result["runs"])


def test_ollama_contract(monkeypatch):
    response = Mock()
    response.json.return_value = {"message": {"content": '{"value":0}'}, "prompt_eval_count": 10, "eval_count": 4}
    post = Mock(return_value=response)
    monkeypatch.setattr(requests, "post", post)
    result = OllamaAdapter("test-model").chat([{"role": "user", "content": "hi"}], 9)
    assert result == ('{"value":0}', 10, 4)
    payload = post.call_args.kwargs["json"]
    assert payload["stream"] is False and payload["options"]["seed"] == 9
    assert payload["format"] == "json"


def test_ollama_multistep_integration(monkeypatch):
    responses = [
        {"message": {"content": '{"tool":"subtract","args":{"a":10.1,"b":10}}'}, "prompt_eval_count": 20, "eval_count": 12},
        {"message": {"content": '{"value":0.1,"passed":true,"unit":"mm"}'}, "prompt_eval_count": 30, "eval_count": 15},
    ]
    observed_messages = []
    def post(url, json, timeout):
        observed_messages.append(list(json["messages"]))
        response = Mock()
        response.json.return_value = responses.pop(0)
        return response
    monkeypatch.setattr(requests, "post", post)
    task = {"id": "known", "kind": "dimension", "data": {"nominal": 10, "measured": 10.1, "tolerance": .2}}
    result = run_agent(OllamaAdapter("stubbed-model"), task)
    from agentbench.tasks import evaluate
    assert evaluate(task, result["answer"])["correct"]
    assert result["input_tokens"] == 50 and result["output_tokens"] == 27
    assert "result" in observed_messages[1][-1]["content"]
    assert "expected" not in observed_messages[0][1]["content"]


class Script:
    def __init__(self, responses):
        self.responses = iter(responses)
    def chat(self, *args):
        return next(self.responses), 10, 2


def test_tool_error_recovery_and_token_aggregation():
    result = run_agent(Script(['{"tool":"shell","args":{}}', '{"value":1,"passed":true,"unit":"mm"}']), generate_tasks(1)[0])
    assert result["trace"][0]["tool_ok"] is False
    assert result["answer"]["value"] == 1
    assert result["input_tokens"] == 20


def test_invalid_json_and_budget():
    assert run_agent(Script(["not json"]), generate_tasks(1)[0])["error"]
    result = run_agent(Script(['{"tool":"absolute","args":{"value":1}}']), generate_tasks(1)[0], max_steps=1)
    assert result["error"] == "Agent step budget exhausted"


@pytest.mark.parametrize("raw", ['{"value":NaN}', '{"value":Infinity}', '{"value":1e999}'])
def test_nonfinite_json_does_not_break_storage(raw):
    result = run_agent(Script([raw]), generate_tasks(1)[0])
    assert result["error"] and result["answer"] is None
    json.dumps(result, allow_nan=False)
