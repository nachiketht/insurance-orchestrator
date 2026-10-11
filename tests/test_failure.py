import random

from underwriting.errors import AgentOutputError
from underwriting.models import RawApplication
from underwriting.pipeline import run_case
from underwriting.retry import RetryPolicy
from underwriting.testing_clients import FaultInjectingLLM, ScriptedClient
from underwriting.tracing import Tracer

RAW = RawApplication(
    applicant_id="A-100",
    applicant_name="Ada Lovelace",
    date_of_birth="12/10/1990",
    address="1 Analytical Engine Rd",
    product="car insurance",
    coverage_amount="$250k",
    stated_income="$80,000",
    claims_history_text="none",
)

INTAKE = {
    "applicant_id": "A-100",
    "applicant_name": "Ada Lovelace",
    "date_of_birth": "1990-12-10",
    "address": "1 Analytical Engine Rd",
    "product": "auto",
    "coverage_amount": 250000,
    "stated_income": 80000,
    "claims_history_text": "none",
    "missing_fields": [],
}

ENRICHMENT = {
    "credit_band": "a",
    "prior_claims_count": 0,
    "asset_age_years": 6,
    "flags": [],
    "notes": "Clean file.",
}

RISK = {
    "factors": [{"name": "prior_claims", "points": 15, "detail": "scripted"}],
    "score": 0,
    "rationale": "Scripted.",
}

RECOMMENDATION = {
    "decision": "approve",
    "reasons": ["Low score."],
    "conditions": [],
    "source": "agent",
}


def _policy(seed: int = 1) -> tuple[RetryPolicy, list[float]]:
    slept: list[float] = []
    policy = RetryPolicy(sleep=slept.append, rng=random.Random(seed))
    return policy, slept


def _client(agent: str, failures: int, error: type[Exception] | None = None) -> FaultInjectingLLM:
    inner = ScriptedClient(
        {
            "intake": INTAKE,
            "enrichment": ENRICHMENT,
            "risk_scoring": RISK,
            "recommendation": RECOMMENDATION,
        }
    )
    return FaultInjectingLLM(inner, agent=agent, failures=failures, error=error)


def test_risk_timeout_retries_then_escalates() -> None:
    policy, slept = _policy()
    preview = RetryPolicy(rng=random.Random(1))
    expected = [preview.delay(1), preview.delay(2)]
    tracer = Tracer()
    case = run_case(RAW, llm=_client("risk_scoring", failures=99), tracer=tracer, policy=policy)

    assert case.status == "escalated"
    assert case.risk is None
    assert case.intake is not None and case.enrichment is not None
    assert case.recommendation is not None
    assert case.recommendation.decision == "refer"
    assert case.recommendation.source == "system_escalation"
    assert "risk_scoring unavailable after 3 attempts" in case.recommendation.reasons[0]
    assert slept == expected
    assert 0.5 <= expected[0] <= 1.0
    assert 1.0 <= expected[1] <= 2.0

    risk = [span for span in tracer.spans if span.agent == "risk_scoring"]
    assert [span.status for span in risk] == ["error", "error", "error"]
    assert [span.will_retry for span in risk] == [True, True, False]
    assert [span.backoff_ms for span in risk[:2]] == [round(wait * 1000, 1) for wait in expected]
    skipped = [span.agent for span in tracer.spans if span.status == "skipped"]
    assert skipped == ["recommendation"]


def test_risk_retries_once_then_succeeds() -> None:
    policy, slept = _policy()
    tracer = Tracer()
    case = run_case(RAW, llm=_client("risk_scoring", failures=1), tracer=tracer, policy=policy)

    assert case.status == "completed"
    assert case.risk is not None and case.risk.score == 15
    assert case.recommendation is not None
    assert case.recommendation.source == "agent"
    risk = [span for span in tracer.spans if span.agent == "risk_scoring"]
    assert [span.status for span in risk] == ["error", "ok"]
    assert risk[0].will_retry is True
    assert len(slept) == 1


def test_bad_output_is_not_retried() -> None:
    policy, slept = _policy()
    tracer = Tracer()
    case = run_case(
        RAW,
        llm=_client("risk_scoring", failures=5, error=AgentOutputError),
        tracer=tracer,
        policy=policy,
    )

    assert case.status == "escalated"
    assert case.recommendation is not None
    assert "after 1 attempt" in case.recommendation.reasons[0]
    risk = [span for span in tracer.spans if span.agent == "risk_scoring"]
    assert len(risk) == 1
    assert risk[0].will_retry is False
    assert slept == []


def test_intake_timeout_escalates_at_the_first_step() -> None:
    policy, _slept = _policy()
    tracer = Tracer()
    case = run_case(RAW, llm=_client("intake", failures=99), tracer=tracer, policy=policy)

    assert case.status == "escalated"
    assert case.intake is None
    assert case.enrichment is None
    assert case.risk is None
    assert case.recommendation is not None
    assert case.recommendation.decision == "refer"
    assert case.recommendation.source == "system_escalation"
    skipped = [span.agent for span in tracer.spans if span.status == "skipped"]
    assert skipped == ["enrichment", "risk_scoring", "recommendation"]
