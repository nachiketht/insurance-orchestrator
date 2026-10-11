"""Normalize a raw application. The model reads messy text; code checks the result."""

from datetime import date

from underwriting.agents._base import call_structured
from underwriting.llm import LLMClient
from underwriting.models import IntakeResult, RawApplication

SYSTEM = """\
Normalize this insurance application into the schema.
- product is auto, home, or life. A car or vehicle is auto. A house or property is home.
- coverage_amount and stated_income are whole US dollars. "$250k" is 250000.
- date_of_birth is YYYY-MM-DD.
Use null for anything you cannot read. Do not guess.
"""

MAX_TOKENS = 800


def intake_agent(raw: RawApplication, *, llm: LLMClient) -> IntakeResult:
    result = call_structured(
        llm,
        agent="intake",
        system=SYSTEM,
        user=raw.model_dump_json(),
        model=IntakeResult,
        max_tokens=MAX_TOKENS,
    )
    assert isinstance(result, IntakeResult)
    return _check(raw, result)


def _check(raw: RawApplication, result: IntakeResult) -> IntakeResult:
    missing: list[str] = []
    product = result.product if result.product in {"auto", "home", "life"} else None
    if product is None:
        missing.append("product")
    coverage = result.coverage_amount
    if coverage is None or coverage <= 0:
        coverage = None
        missing.append("coverage_amount")
    dob = result.date_of_birth if _plausible_dob(result.date_of_birth) else None
    if dob is None:
        missing.append("date_of_birth")
    return result.model_copy(
        update={
            "applicant_id": raw.applicant_id,
            "product": product,
            "coverage_amount": coverage,
            "date_of_birth": dob,
            "missing_fields": missing,
        }
    )


def _plausible_dob(value: str | None) -> bool:
    if not value:
        return False
    try:
        born = date.fromisoformat(value)
    except ValueError:
        return False
    today = date.today()
    age = today.year - born.year - ((today.month, today.day) < (born.month, born.day))
    return 18 <= age <= 100
