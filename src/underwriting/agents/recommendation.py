"""Recommend approve, deny, or refer. Code sets the decision; the model explains it."""

import json

from underwriting.agents._base import call_structured
from underwriting.llm import LLMClient
from underwriting.models import EnrichmentResult, IntakeResult, RecommendationResult, RiskResult

APPROVE_AT_OR_BELOW = 30
DENY_AT_OR_ABOVE = 70

SYSTEM = """\
Write the reasons for an underwriting decision that code has already made.
The baseline decision is included in the input.
You may return that same decision.
The only change you may make is from approve to refer, when the file still worries you.
You may not change refer or deny, and you may not return deny unless the baseline is deny.
Give at least one short reason. Use an empty conditions list unless the baseline is approve.
"""

MAX_TOKENS = 800


def recommendation_agent(
    intake: IntakeResult,
    enrichment: EnrichmentResult,
    risk: RiskResult,
    *,
    llm: LLMClient,
) -> RecommendationResult:
    decided = _baseline(intake, enrichment, risk)
    user = json.dumps(
        {
            "baseline_decision": decided,
            "intake": intake.model_dump(),
            "enrichment": enrichment.model_dump(),
            "risk": risk.model_dump(),
        }
    )
    result = call_structured(
        llm,
        agent="recommendation",
        system=SYSTEM,
        user=user,
        model=RecommendationResult,
        max_tokens=MAX_TOKENS,
    )
    assert isinstance(result, RecommendationResult)
    return _check(decided, result)


def _baseline(intake: IntakeResult, enrichment: EnrichmentResult, risk: RiskResult) -> str:
    if intake.missing_fields or enrichment.credit_band == "no_hit":
        return "refer"
    if risk.score <= APPROVE_AT_OR_BELOW:
        return "approve"
    if risk.score >= DENY_AT_OR_ABOVE:
        return "deny"
    return "refer"


def _check(baseline_decision: str, result: RecommendationResult) -> RecommendationResult:
    allowed = {baseline_decision}
    if baseline_decision == "approve":
        allowed.add("refer")
    reasons = list(result.reasons)
    decision = result.decision
    if decision not in allowed:
        reasons.append(f"decision held at {baseline_decision}; model returned {decision}")
        decision = baseline_decision
    return result.model_copy(
        update={"decision": decision, "reasons": reasons, "source": "agent"}
    )
