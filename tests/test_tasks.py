import math
import pytest
from agentbench.tasks import evaluate, generate_tasks, ground_truth
from agentbench.tools import execute


def test_seed_and_coverage():
    assert generate_tasks(20, 7) == generate_tasks(20, 7)
    assert generate_tasks(20, 7) != generate_tasks(20, 8)
    assert len({t["kind"] for t in generate_tasks(4)}) == 4


@pytest.mark.parametrize("kind,data,value,passed", [
    ("dimension", {"nominal": 10, "measured": 10.2, "tolerance": .2}, .2, True),
    ("dimension", {"nominal": 10, "measured": 9.7, "tolerance": .2}, .3, False),
    ("position", {"reference": [0, 0, 0], "measured": [3, 4, 0], "tolerance": 5}, 5, True),
    ("stack", {"tolerances": [.1, -.2, .3], "limit": .5}, .6, False),
    ("clearance", {"hole": 10, "shaft": 10.1, "minimum": 0}, -.1, False),
    ("clearance", {"hole": 10.05, "shaft": 10, "minimum": .05}, .05, True),
])
def test_known_reference_cases(kind, data, value, passed):
    result = ground_truth({"kind": kind, "data": data})
    assert result["value"] == pytest.approx(value)
    assert result["passed"] is passed


@pytest.mark.parametrize("answer", [None, {}, [], {"value": float("nan"), "passed": True, "unit": "mm"},
    {"value": 10**400, "passed": True, "unit": "mm"},
    {"value": True, "passed": True, "unit": "mm"}, {"value": .2, "passed": "true", "unit": "mm"},
    {"value": .2, "passed": True, "unit": "cm"}])
def test_invalid_answers_fail(answer):
    assert not evaluate(generate_tasks(1)[0], answer)["correct"]


def test_correct_answer_and_numeric_error():
    task = generate_tasks(1)[0]
    answer = ground_truth(task)
    assert evaluate(task, answer)["correct"]
    answer["value"] += .1
    assert not evaluate(task, answer)["correct"]


@pytest.mark.parametrize("name,args", [("shell", {}), ("subtract", {"a": "1", "b": 2}),
    ("distance", {"a": [1], "b": [1, 2, 3]}), ("sum_abs", {"values": [math.inf]}),
    ("absolute", {"value": 1e20})])
def test_tool_boundaries(name, args):
    with pytest.raises(ValueError):
        execute(name, args)
