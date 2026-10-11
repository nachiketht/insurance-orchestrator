import pytest

from underwriting.errors import LLMTimeoutError, LLMTransientError, ModelNotFoundError
from underwriting.llm import OllamaClient
from underwriting.models import RawApplication
from underwriting.pipeline import run_case

pytestmark = pytest.mark.live

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


def test_live_intake_hands_off_to_enrichment() -> None:
    client = OllamaClient()
    try:
        client.preflight()
        client.warm_up()
        case = run_case(RAW, llm=client)
    except (ModelNotFoundError, LLMTransientError, LLMTimeoutError) as exc:
        pytest.skip(f"Ollama is not ready: {exc}")
    finally:
        client.close()

    assert case.intake is not None and case.intake.product == "auto"
    assert case.enrichment is not None
    assert case.enrichment.credit_band == "a"
    assert case.risk is not None
    assert case.risk.score == sum(factor.points for factor in case.risk.factors)
    assert case.recommendation is not None
    assert case.recommendation.decision in {"approve", "refer", "deny"}
    assert case.status == "completed"
