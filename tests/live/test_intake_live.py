import pytest

from underwriting.agents.intake import intake_agent
from underwriting.errors import LLMTimeoutError, LLMTransientError, ModelNotFoundError
from underwriting.llm import OllamaClient
from underwriting.models import RawApplication

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


def test_live_intake_normalizes_car_insurance() -> None:
    client = OllamaClient()
    try:
        client.preflight()
        client.warm_up()
        result = intake_agent(RAW, llm=client)
    except (ModelNotFoundError, LLMTransientError, LLMTimeoutError) as exc:
        pytest.skip(f"Ollama is not ready: {exc}")
    finally:
        client.close()

    assert result.applicant_id == "A-100"
    assert result.product == "auto"
    assert result.coverage_amount == 250000
    assert result.date_of_birth == "1990-12-10"
    assert result.missing_fields == []
