"""Run the underwriting steps in order. Only this module writes the case record."""

from collections.abc import Callable
from datetime import datetime, timezone
from typing import NamedTuple

from underwriting.agents.enrichment import enrichment_agent
from underwriting.agents.intake import intake_agent
from underwriting.agents.recommendation import recommendation_agent
from underwriting.agents.risk import risk_scoring_agent
from underwriting.bureau import lookup
from underwriting.llm import LLMClient
from underwriting.models import CaseRecord, Escalation, RawApplication, RecommendationResult
from underwriting.retry import RetryPolicy, StepFailed, run_step
from underwriting.tracing import Span, TracedLLM, Tracer


class _Step(NamedTuple):
    number: int
    agent: str
    source: list[str]
    received: Callable[[], dict]
    call: Callable


def run_case(
    raw: RawApplication,
    *,
    llm: LLMClient,
    tracer: Tracer | None = None,
    policy: RetryPolicy | None = None,
) -> CaseRecord:
    """Run all four agents. A failed step is retried, then the case is referred."""
    policy = policy or RetryPolicy()
    case = CaseRecord.new(raw)
    client = TracedLLM(llm, tracer) if tracer is not None else llm
    bureau = lookup(raw.applicant_id)
    steps = [
        _Step(
            1,
            "intake",
            ["raw_application"],
            lambda: raw.model_dump(),
            lambda: intake_agent(raw, llm=client),
        ),
        _Step(
            2,
            "enrichment",
            ["case.intake", "bureau"],
            lambda: {
                "intake": case.intake.model_dump(mode="json"),
                "bureau": bureau.model_dump(),
            },
            lambda: enrichment_agent(case.intake, bureau, llm=client),
        ),
        _Step(
            3,
            "risk_scoring",
            ["case.intake", "case.enrichment"],
            lambda: {
                "intake": case.intake.model_dump(mode="json"),
                "enrichment": case.enrichment.model_dump(mode="json"),
            },
            lambda: risk_scoring_agent(case.intake, case.enrichment, llm=client),
        ),
        _Step(
            4,
            "recommendation",
            ["case.intake", "case.enrichment", "case.risk"],
            lambda: {
                "intake": case.intake.model_dump(mode="json"),
                "enrichment": case.enrichment.model_dump(mode="json"),
                "risk": case.risk.model_dump(mode="json"),
            },
            lambda: recommendation_agent(
                case.intake, case.enrichment, case.risk, llm=client
            ),
        ),
    ]
    fields = ("intake", "enrichment", "risk", "recommendation")
    for index, step in enumerate(steps):
        try:
            value = run_step(
                step.call,
                policy=policy,
                tracer=tracer,
                step=step.number,
                agent=step.agent,
                source=step.source,
                received=step.received(),
            )
        except StepFailed as failed:
            _escalate(case, failed)
            _skip_remaining(tracer, steps[index + 1 :])
            return case
        setattr(case, fields[index], value)
    case.status = "completed"
    return case


def _escalate(case: CaseRecord, failed: StepFailed) -> None:
    cause = failed.cause
    case.status = "escalated"
    case.escalation = Escalation(
        step=failed.agent,
        error_type=type(cause).__name__,
        message=str(cause),
        attempts=failed.attempts,
    )
    noun = "attempt" if failed.attempts == 1 else "attempts"
    case.recommendation = RecommendationResult(
        decision="refer",
        source="system_escalation",
        reasons=[
            f"{failed.agent} unavailable after {failed.attempts} {noun}: "
            f"{type(cause).__name__}"
        ],
    )


def _skip_remaining(tracer: Tracer | None, steps: list[_Step]) -> None:
    if tracer is None:
        return
    for step in steps:
        tracer.spans.append(
            Span(
                step=step.number,
                agent=step.agent,
                timestamp=datetime.now(timezone.utc).isoformat(),
                input={},
                input_source=step.source,
                status="skipped",
            )
        )
