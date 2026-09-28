from agentbench.reasoning_trace_verifier import (
    ClaimStatus,
    Evidence,
    TraceStep,
    contradiction_score,
    jaccard_similarity,
    verify_trace,
)


def test_jaccard_similarity_identical():
    assert jaccard_similarity("tool returned success", "tool returned success") == 1.0


def test_supported_claim():
    steps = [
        TraceStep(
            step_id="s1",
            claim="database query returned 12 records",
            context="The agent is checking the query result.",
            tool_name="database",
            tool_result="database query returned 12 records successfully",
        )
    ]

    results, metrics = verify_trace(steps, evidence=[])

    assert results[0].status == ClaimStatus.SUPPORTED
    assert metrics.supported_steps == 1
    assert metrics.grounding_score == 1.0


def test_contradicted_claim():
    steps = [
        TraceStep(
            step_id="s1",
            claim="deployment succeeded",
            tool_name="deploy",
            tool_result="deployment failed because health check failed",
        )
    ]

    results, metrics = verify_trace(steps, evidence=[])

    assert results[0].status == ClaimStatus.CONTRADICTED
    assert metrics.contradicted_steps == 1


def test_unresolved_claim():
    steps = [
        TraceStep(
            step_id="s1",
            claim="the configuration is optimal for production",
        )
    ]

    results, metrics = verify_trace(steps, evidence=[])

    assert results[0].status == ClaimStatus.UNRESOLVED
    assert metrics.unresolved_steps == 1


def test_external_evidence_can_support_claim():
    evidence = [
        Evidence(
            source_id="doc-1",
            text="the retry limit is configured to 3 attempts",
            source_type="config",
        )
    ]

    steps = [
        TraceStep(
            step_id="s1",
            claim="retry limit is 3 attempts",
        )
    ]

    results, _ = verify_trace(steps, evidence)

    assert results[0].status == ClaimStatus.SUPPORTED
    assert "doc-1" in results[0].matched_evidence


def test_trace_metrics_cover_mixed_trace():
    evidence = [
        Evidence(
            source_id="policy",
            text="requests above 1000 items require batch processing",
        )
    ]

    steps = [
        TraceStep(
            step_id="s1",
            claim="requests above 1000 items require batch processing",
        ),
        TraceStep(
            step_id="s2",
            claim="the batch job completed successfully",
            tool_name="batch_runner",
            tool_result="the batch job failed during validation",
        ),
        TraceStep(
            step_id="s3",
            claim="the output is safe for production deployment",
        ),
    ]

    results, metrics = verify_trace(
        steps,
        evidence,
        final_answer="batch processing is required",
    )

    assert metrics.total_steps == 3
    assert metrics.supported_steps == 1
    assert metrics.contradicted_steps == 1
    assert metrics.unresolved_steps == 1
    assert 0.0 <= metrics.final_answer_support_ratio <= 1.0


def test_contradiction_score_detects_success_failure_conflict():
    score = contradiction_score(
        "the operation succeeded",
        "the operation failed",
    )
    assert score > 0.0
