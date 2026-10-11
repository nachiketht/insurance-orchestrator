import pytest

from underwriting.agents.enrichment import enrichment_agent
from underwriting.bureau import lookup
from underwriting.errors import LLMTimeoutError, LLMTransientError, ModelNotFoundError
from underwriting.llm import OllamaClient
from underwriting.models import IntakeResult

pytestmark = pytest.mark.live

INTAKE = IntakeResult(
    applicant_id="A-200",
    applicant_name="Grace Hopper",
    date_of_birth="1970-01-01",
    address="1 Compiler Way",
    product="home",
    coverage_amount=600000,
    stated_income=50000,
    claims_history_text="three water claims",
    missing_fields=[],
)


def test_live_enrichment_flags_frequent_claims() -> None:
    bureau = lookup("A-200")
    client = OllamaClient()
    try:
        client.preflight()
        client.warm_up()
        result = enrichment_agent(INTAKE, bureau, llm=client)
    except (ModelNotFoundError, LLMTransientError, LLMTimeoutError) as exc:
        pytest.skip(f"Ollama is not ready: {exc}")
    finally:
        client.close()

    assert result.credit_band == "d"
    assert result.prior_claims_count == 6
    assert "high_claims_frequency" in result.flags
