"""Interpret bureau facts. The model may flag them; it may not change them."""

import json

from underwriting.agents._base import call_structured
from underwriting.llm import LLMClient
from underwriting.models import BureauRecord, EnrichmentResult, IntakeResult

FLAGS = (
    "high_claims_frequency",
    "coverage_income_mismatch",
    "weak_credit",
    "old_asset",
    "no_bureau_record",
)

SYSTEM = """\
You are given a normalized application and a bureau record.
Write notes and choose flags only from this list:
- high_claims_frequency: prior_claims_count is 3 or more
- coverage_income_mismatch: coverage_amount is more than 5 times stated_income
- weak_credit: credit_band is c or d
- old_asset: asset_age_years is 30 or more
- no_bureau_record: credit_band is no_hit
Do not change credit_band, prior_claims_count, or asset_age_years.
"""

MAX_TOKENS = 800


def enrichment_agent(
    intake: IntakeResult,
    bureau: BureauRecord,
    *,
    llm: LLMClient,
) -> EnrichmentResult:
    user = json.dumps(
        {"intake": intake.model_dump(), "bureau": bureau.model_dump()},
    )
    result = call_structured(
        llm,
        agent="enrichment",
        system=SYSTEM,
        user=user,
        model=EnrichmentResult,
        max_tokens=MAX_TOKENS,
    )
    assert isinstance(result, EnrichmentResult)
    return _check(bureau, result)


def _check(bureau: BureauRecord, result: EnrichmentResult) -> EnrichmentResult:
    flags = [flag for flag in result.flags if flag in FLAGS]
    if bureau.credit_band == "no_hit" and "no_bureau_record" not in flags:
        flags.append("no_bureau_record")
    if bureau.credit_band != "no_hit":
        flags = [flag for flag in flags if flag != "no_bureau_record"]
    return result.model_copy(
        update={
            "credit_band": bureau.credit_band,
            "prior_claims_count": bureau.prior_claims_count,
            "asset_age_years": bureau.asset_age_years,
            "flags": flags,
        }
    )
