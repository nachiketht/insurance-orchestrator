"""Score risk from factors. The model assigns points; code computes the total."""

import json

from underwriting.agents._base import call_structured
from underwriting.llm import LLMClient
from underwriting.models import EnrichmentResult, IntakeResult, RiskResult

FACTORS = (
    "prior_claims",
    "credit",
    "coverage_income",
    "asset_age",
    "missing_data",
)

SYSTEM = """\
Score an insurance application from the intake and enrichment you are given.
Return one factor per reason, using only these names:
- prior_claims: 0 if prior_claims_count is 0, 15 if it is 1 or 2, 30 if it is 3 or more
- credit: 0 for band a or b, 10 for c, 25 for d. Use missing_data instead when the band is no_hit
- coverage_income: 20 when coverage_amount is more than 5 times stated_income, otherwise 0
- asset_age: 15 when asset_age_years is 30 or more, otherwise 0
- missing_data: 20 when credit_band is no_hit or missing_fields is not empty, otherwise 0
Omit a factor whose points would be 0. Each factor needs a short detail.
The score field may be 0. Code replaces it with the sum of the points.
"""

MAX_TOKENS = 1000


def risk_scoring_agent(
    intake: IntakeResult,
    enrichment: EnrichmentResult,
    *,
    llm: LLMClient,
) -> RiskResult:
    user = json.dumps(
        {"intake": intake.model_dump(), "enrichment": enrichment.model_dump()},
    )
    result = call_structured(
        llm,
        agent="risk_scoring",
        system=SYSTEM,
        user=user,
        model=RiskResult,
        max_tokens=MAX_TOKENS,
    )
    assert isinstance(result, RiskResult)
    return _check(result)


def _check(result: RiskResult) -> RiskResult:
    kept = []
    seen: set[str] = set()
    for factor in result.factors:
        if factor.name not in FACTORS or factor.name in seen:
            continue
        seen.add(factor.name)
        kept.append(factor)
    score = min(100, sum(factor.points for factor in kept))
    return result.model_copy(update={"factors": kept, "score": score})
