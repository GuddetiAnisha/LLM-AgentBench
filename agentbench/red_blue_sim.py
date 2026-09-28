"""Safe red-blue teaming simulation for AI-agent evaluation.

This module provides an abstract, non-operational cybersecurity simulation for
benchmarking offensive (red) and defensive (blue) agent decision policies.

It intentionally contains no exploit code, malware, credential theft, scanning,
payload execution, or live-target capability. Actions are symbolic state
transitions in an isolated simulation for research and portfolio use.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import pandas as pd


RED_ACTIONS = (
    "observe",
    "probe_access",
    "attempt_privilege_gain",
    "attempt_lateral_move",
    "attempt_data_access",
    "wait",
)

BLUE_ACTIONS = (
    "monitor",
    "block_activity",
    "isolate_service",
    "rotate_credentials",
    "apply_policy",
    "recover_service",
    "wait",
)


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    difficulty: int
    detection_threshold: float
    containment_strength: float
    recovery_speed: float


@dataclass
class CyberState:
    step: int = 0
    suspicious_score: float = 0.0
    privilege_level: int = 0
    lateral_position: int = 0
    data_accessed: bool = False
    service_isolated: bool = False
    credentials_rotated: bool = False
    service_available: bool = True
    contained: bool = False
    detected: bool = False


@dataclass(frozen=True)
class EpisodeSummary:
    scenario_id: str
    steps: int
    red_success: bool
    detected: bool
    contained: bool
    recovered: bool
    blue_false_positives: int
    blue_response_step: int | None
    red_action_successes: int
    blue_action_successes: int


def _red_action_effect(action: str, state: CyberState, scenario: Scenario) -> bool:
    """Apply one symbolic red action. Returns whether it advanced red objectives."""
    if action not in RED_ACTIONS:
        raise ValueError(f"unknown red action: {action}")

    success = False

    if state.contained:
        return False

    if action == "observe":
        state.suspicious_score += 0.08
        success = True

    elif action == "probe_access":
        state.suspicious_score += 0.18
        if not state.credentials_rotated:
            state.privilege_level = max(state.privilege_level, 1)
            success = True

    elif action == "attempt_privilege_gain":
        state.suspicious_score += 0.24
        if state.privilege_level >= 1 and not state.credentials_rotated:
            state.privilege_level = min(2, state.privilege_level + 1)
            success = True

    elif action == "attempt_lateral_move":
        state.suspicious_score += 0.20
        if state.privilege_level >= 2 and not state.service_isolated:
            state.lateral_position += 1
            success = True

    elif action == "attempt_data_access":
        state.suspicious_score += 0.28
        if state.privilege_level >= 2 and state.lateral_position >= 1 and not state.service_isolated:
            state.data_accessed = True
            success = True

    elif action == "wait":
        state.suspicious_score = max(0.0, state.suspicious_score - 0.03)

    return success


def _blue_action_effect(action: str, state: CyberState, scenario: Scenario) -> tuple[bool, bool]:
    """Apply symbolic blue action. Returns (success, false_positive)."""
    if action not in BLUE_ACTIONS:
        raise ValueError(f"unknown blue action: {action}")

    false_positive = False
    success = False
    truly_suspicious = (
        state.suspicious_score >= scenario.detection_threshold
        or state.privilege_level > 0
        or state.lateral_position > 0
        or state.data_accessed
    )

    if action == "monitor":
        if state.suspicious_score >= scenario.detection_threshold:
            state.detected = True
            success = True

    elif action == "block_activity":
        if truly_suspicious:
            state.detected = True
            state.suspicious_score = max(0.0, state.suspicious_score - scenario.containment_strength)
            success = True
        else:
            false_positive = True

    elif action == "isolate_service":
        if truly_suspicious:
            state.detected = True
            state.service_isolated = True
            state.service_available = False
            state.contained = True
            success = True
        else:
            false_positive = True
            state.service_available = False

    elif action == "rotate_credentials":
        if truly_suspicious:
            state.credentials_rotated = True
            state.privilege_level = min(state.privilege_level, 1)
            state.detected = True
            success = True
        else:
            false_positive = True

    elif action == "apply_policy":
        if truly_suspicious:
            state.detected = True
            state.contained = True
            state.suspicious_score = max(0.0, state.suspicious_score - scenario.containment_strength)
            success = True
        else:
            false_positive = True

    elif action == "recover_service":
        if state.service_isolated or state.contained:
            state.service_available = True
            state.service_isolated = False
            state.contained = True
            success = True

    return success, false_positive


def rule_red_policy(state: CyberState, scenario: Scenario) -> str:
    if state.privilege_level == 0:
        return "probe_access"
    if state.privilege_level < 2:
        return "attempt_privilege_gain"
    if state.lateral_position < 1:
        return "attempt_lateral_move"
    if not state.data_accessed:
        return "attempt_data_access"
    return "wait"


def cautious_red_policy(state: CyberState, scenario: Scenario) -> str:
    if state.suspicious_score > scenario.detection_threshold * 0.8:
        return "wait"
    return rule_red_policy(state, scenario)


def rule_blue_policy(state: CyberState, scenario: Scenario) -> str:
    if state.service_isolated and not state.service_available:
        return "recover_service"
    if state.data_accessed or state.lateral_position > 0:
        return "isolate_service"
    if state.privilege_level >= 2:
        return "rotate_credentials"
    if state.suspicious_score >= scenario.detection_threshold:
        return "block_activity"
    return "monitor"


def conservative_blue_policy(state: CyberState, scenario: Scenario) -> str:
    if state.suspicious_score >= scenario.detection_threshold * 1.2:
        return rule_blue_policy(state, scenario)
    return "monitor"


def run_episode(
    scenario: Scenario,
    red_policy: Callable[[CyberState, Scenario], str] = rule_red_policy,
    blue_policy: Callable[[CyberState, Scenario], str] = rule_blue_policy,
    max_steps: int = 12,
) -> tuple[EpisodeSummary, pd.DataFrame]:
    state = CyberState()
    trace = []
    red_successes = 0
    blue_successes = 0
    false_positives = 0
    blue_response_step = None

    for step in range(1, max_steps + 1):
        state.step = step

        red_action = red_policy(state, scenario)
        red_success = _red_action_effect(red_action, state, scenario)
        red_successes += int(red_success)

        blue_action = blue_policy(state, scenario)
        blue_success, false_positive = _blue_action_effect(blue_action, state, scenario)
        blue_successes += int(blue_success)
        false_positives += int(false_positive)

        if (state.detected or state.contained) and blue_response_step is None:
            blue_response_step = step

        trace.append(
            {
                "scenario_id": scenario.scenario_id,
                "step": step,
                "red_action": red_action,
                "red_action_success": red_success,
                "blue_action": blue_action,
                "blue_action_success": blue_success,
                "false_positive": false_positive,
                "suspicious_score": round(state.suspicious_score, 4),
                "privilege_level": state.privilege_level,
                "lateral_position": state.lateral_position,
                "data_accessed": state.data_accessed,
                "detected": state.detected,
                "contained": state.contained,
                "service_available": state.service_available,
            }
        )

        if state.data_accessed or state.contained:
            if state.contained and not state.service_available:
                # one recovery opportunity before ending
                recovery_action = blue_policy(state, scenario)
                if recovery_action == "recover_service":
                    ok, fp = _blue_action_effect(recovery_action, state, scenario)
                    blue_successes += int(ok)
                    false_positives += int(fp)
            break

    summary = EpisodeSummary(
        scenario_id=scenario.scenario_id,
        steps=len(trace),
        red_success=bool(state.data_accessed),
        detected=bool(state.detected),
        contained=bool(state.contained),
        recovered=bool(state.service_available and state.contained),
        blue_false_positives=false_positives,
        blue_response_step=blue_response_step,
        red_action_successes=red_successes,
        blue_action_successes=blue_successes,
    )
    return summary, pd.DataFrame(trace)


def evaluate_matchup(
    scenarios: Sequence[Scenario],
    red_policy: Callable[[CyberState, Scenario], str],
    blue_policy: Callable[[CyberState, Scenario], str],
) -> dict:
    summaries = [run_episode(s, red_policy, blue_policy)[0] for s in scenarios]
    total = len(summaries)
    if total == 0:
        return {
            "episodes": 0,
            "red_success_rate": 0.0,
            "detection_rate": 0.0,
            "containment_rate": 0.0,
            "recovery_rate": 0.0,
            "false_positive_rate": 0.0,
            "mean_response_step": 0.0,
        }

    response_steps = [s.blue_response_step for s in summaries if s.blue_response_step is not None]
    return {
        "episodes": total,
        "red_success_rate": sum(s.red_success for s in summaries) / total,
        "detection_rate": sum(s.detected for s in summaries) / total,
        "containment_rate": sum(s.contained for s in summaries) / total,
        "recovery_rate": sum(s.recovered for s in summaries) / total,
        "false_positive_rate": sum(s.blue_false_positives > 0 for s in summaries) / total,
        "mean_response_step": (
            sum(response_steps) / len(response_steps) if response_steps else 0.0
        ),
    }


def benchmark_policies(scenarios: Sequence[Scenario]) -> pd.DataFrame:
    matchups = (
        ("rule_red_vs_rule_blue", rule_red_policy, rule_blue_policy),
        ("cautious_red_vs_rule_blue", cautious_red_policy, rule_blue_policy),
        ("rule_red_vs_conservative_blue", rule_red_policy, conservative_blue_policy),
        ("cautious_red_vs_conservative_blue", cautious_red_policy, conservative_blue_policy),
    )

    rows = []
    for name, red, blue in matchups:
        metrics = evaluate_matchup(scenarios, red, blue)
        rows.append({"matchup": name, **metrics})
    return pd.DataFrame(rows)


def scenarios_from_frame(frame: pd.DataFrame) -> list[Scenario]:
    required = {
        "scenario_id",
        "difficulty",
        "detection_threshold",
        "containment_strength",
        "recovery_speed",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing scenario columns: {sorted(missing)}")

    return [
        Scenario(
            scenario_id=str(row["scenario_id"]),
            difficulty=int(row["difficulty"]),
            detection_threshold=float(row["detection_threshold"]),
            containment_strength=float(row["containment_strength"]),
            recovery_speed=float(row["recovery_speed"]),
        )
        for _, row in frame.iterrows()
    ]
