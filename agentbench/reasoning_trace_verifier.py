"""Step-level reasoning-trace verification for tool-using AI agents.

This module is intentionally software-only and deterministic.  It combines:
1. symbolic checks (explicit support/contradiction markers and tool evidence), and
2. lightweight statistical text similarity based on token overlap.

It is designed for reproducible experiments, not as a formal proof system.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Iterable, Sequence
import json
import math
import re


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


class ClaimStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    UNRESOLVED = "UNRESOLVED"


@dataclass(frozen=True)
class Evidence:
    source_id: str
    text: str
    source_type: str = "tool"


@dataclass(frozen=True)
class TraceStep:
    step_id: str
    claim: str
    context: str = ""
    tool_name: str | None = None
    tool_result: str | None = None


@dataclass(frozen=True)
class StepVerification:
    step_id: str
    claim: str
    status: ClaimStatus
    support_score: float
    contradiction_score: float
    relevance_score: float
    coherence_score: float
    matched_evidence: tuple[str, ...]
    notes: tuple[str, ...]


@dataclass(frozen=True)
class TraceMetrics:
    total_steps: int
    supported_steps: int
    contradicted_steps: int
    unresolved_steps: int
    grounding_score: float
    contradiction_rate: float
    unresolved_rate: float
    contextual_relevance: float
    coherence_score: float
    tool_evidence_consistency: float
    final_answer_support_ratio: float


def _tokens(text: str) -> set[str]:
    return {tok.lower() for tok in _TOKEN_RE.findall(text or "")}


def jaccard_similarity(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _normalize_polarity_tokens(text: str) -> set[str]:
    """Normalize common positive/negative status words to comparable roots."""
    tokens = _tokens(text)
    normalized = set(tokens)
    mapping = {
        "succeeded": "success",
        "successful": "success",
        "successfully": "success",
        "completed": "complete",
        "failed": "failure",
        "failing": "failure",
        "disabled": "disable",
        "enabled": "enable",
    }
    normalized.update(mapping[token] for token in tokens if token in mapping)
    return normalized


def _negation_signature(text: str) -> set[str]:
    tokens = list(_TOKEN_RE.findall((text or "").lower()))
    signature = set()
    for i, token in enumerate(tokens):
        if token in {"not", "no", "never", "false", "failed", "failure", "cannot", "can't"}:
            window = tokens[i + 1 : i + 4]
            signature.update(window)
    return signature


def contradiction_score(claim: str, evidence_text: str) -> float:
    overlap = jaccard_similarity(claim, evidence_text)
    if overlap == 0:
        return 0.0

    c_neg = _negation_signature(claim)
    e_neg = _negation_signature(evidence_text)

    claim_tokens = _normalize_polarity_tokens(claim)
    evidence_tokens = _normalize_polarity_tokens(evidence_text)

    polarity_conflict = bool(
        (c_neg & evidence_tokens) or
        (e_neg & claim_tokens)
    )

    explicit_conflict_terms = (
        ("success" in claim_tokens and "failure" in evidence_tokens)
        or ("failure" in claim_tokens and "success" in evidence_tokens)
        or ("complete" in claim_tokens and "failure" in evidence_tokens)
        or ("failure" in claim_tokens and "complete" in evidence_tokens)
        or ("enable" in claim_tokens and "disable" in evidence_tokens)
        or ("disable" in claim_tokens and "enable" in evidence_tokens)
        or ("true" in claim_tokens and "false" in evidence_tokens)
        or ("false" in claim_tokens and "true" in evidence_tokens)
    )

    if polarity_conflict or explicit_conflict_terms:
        return min(1.0, 0.55 + 0.45 * overlap)

    return 0.0


def _combined_evidence(step: TraceStep, evidence: Sequence[Evidence]) -> list[Evidence]:
    combined = list(evidence)
    if step.tool_result:
        combined.append(
            Evidence(
                source_id=f"{step.step_id}:tool_result",
                text=step.tool_result,
                source_type="tool_result",
            )
        )
    return combined


def verify_step(
    step: TraceStep,
    evidence: Sequence[Evidence],
    previous_step: TraceStep | None = None,
    support_threshold: float = 0.35,
    contradiction_threshold: float = 0.45,
) -> StepVerification:
    candidates = _combined_evidence(step, evidence)

    scored = []
    for item in candidates:
        support = jaccard_similarity(step.claim, item.text)
        contra = contradiction_score(step.claim, item.text)
        scored.append((item, support, contra))

    best_support = max((s for _, s, _ in scored), default=0.0)
    best_contra = max((c for _, _, c in scored), default=0.0)

    matched = tuple(
        item.source_id
        for item, support, contra in scored
        if support >= support_threshold or contra >= contradiction_threshold
    )

    relevance_basis = " ".join(
        part for part in [step.context, step.tool_result or ""] if part
    )
    relevance = jaccard_similarity(step.claim, relevance_basis) if relevance_basis else 0.0

    if previous_step is None:
        coherence = 1.0
    else:
        coherence = jaccard_similarity(previous_step.claim, step.claim)
        if step.context:
            coherence = max(coherence, jaccard_similarity(previous_step.claim, step.context))

    notes = []
    if best_contra >= contradiction_threshold:
        status = ClaimStatus.CONTRADICTED
        notes.append("Contradictory evidence detected.")
    elif best_support >= support_threshold:
        status = ClaimStatus.SUPPORTED
        notes.append("Claim is grounded in available evidence.")
    else:
        status = ClaimStatus.UNRESOLVED
        notes.append("Available evidence is insufficient to verify the claim.")

    if step.tool_name and not step.tool_result:
        notes.append("A tool is referenced but no tool result is attached.")

    return StepVerification(
        step_id=step.step_id,
        claim=step.claim,
        status=status,
        support_score=round(best_support, 4),
        contradiction_score=round(best_contra, 4),
        relevance_score=round(relevance, 4),
        coherence_score=round(coherence, 4),
        matched_evidence=matched,
        notes=tuple(notes),
    )


def verify_trace(
    steps: Sequence[TraceStep],
    evidence: Sequence[Evidence],
    final_answer: str = "",
) -> tuple[list[StepVerification], TraceMetrics]:
    results: list[StepVerification] = []

    previous = None
    for step in steps:
        result = verify_step(step, evidence, previous_step=previous)
        results.append(result)
        previous = step

    total = len(results)
    supported = sum(r.status == ClaimStatus.SUPPORTED for r in results)
    contradicted = sum(r.status == ClaimStatus.CONTRADICTED for r in results)
    unresolved = sum(r.status == ClaimStatus.UNRESOLVED for r in results)

    grounding = supported / total if total else 0.0
    contradiction_rate = contradicted / total if total else 0.0
    unresolved_rate = unresolved / total if total else 0.0
    relevance = sum(r.relevance_score for r in results) / total if total else 0.0
    coherence = sum(r.coherence_score for r in results) / total if total else 0.0

    tool_steps = [s for s in steps if s.tool_name]
    if tool_steps:
        consistent = 0
        for step, result in zip(steps, results):
            if not step.tool_name:
                continue
            if step.tool_result and result.status != ClaimStatus.CONTRADICTED:
                consistent += 1
        tool_consistency = consistent / len(tool_steps)
    else:
        tool_consistency = 1.0

    if final_answer:
        supported_claim_text = " ".join(
            r.claim for r in results if r.status == ClaimStatus.SUPPORTED
        )
        final_support = jaccard_similarity(final_answer, supported_claim_text)
    else:
        final_support = grounding

    metrics = TraceMetrics(
        total_steps=total,
        supported_steps=supported,
        contradicted_steps=contradicted,
        unresolved_steps=unresolved,
        grounding_score=round(grounding, 4),
        contradiction_rate=round(contradiction_rate, 4),
        unresolved_rate=round(unresolved_rate, 4),
        contextual_relevance=round(relevance, 4),
        coherence_score=round(coherence, 4),
        tool_evidence_consistency=round(tool_consistency, 4),
        final_answer_support_ratio=round(final_support, 4),
    )

    return results, metrics


def export_verification_report(
    results: Sequence[StepVerification],
    metrics: TraceMetrics,
) -> str:
    payload = {
        "steps": [
            {
                **asdict(result),
                "status": result.status.value,
            }
            for result in results
        ],
        "metrics": asdict(metrics),
    }
    return json.dumps(payload, indent=2)
