import pytest

from underwriting.agents.risk import risk_scoring_agent
from underwriting.errors import LLMTimeoutError, LLMTransientError, ModelNotFoundError
from underwriting.llm import OllamaClient
from underwriting.models import EnrichmentResult, IntakeResult

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

ENRICHMENT = EnrichmentResult(
    credit_band="d",
    prior_claims_count=6,
    asset_age_years=70,
    flags=["high_claims_frequency", "weak_credit", "old_asset", "coverage_income_mismatch"],
    notes="Frequent claims, weak credit, old house, coverage far above income.",
)


def test_live_risk_scores_the_high_claim_case() -> None:
    client = OllamaClient()
    try:
        client.preflight()
        client.warm_up()
        result = risk_scoring_agent(INTAKE, ENRICHMENT, llm=client)
    except (ModelNotFoundError, LLMTransientError, LLMTimeoutError) as exc:
        pytest.skip(f"Ollama is not ready: {exc}")
    finally:
        client.close()

    names = [factor.name for factor in result.factors]
    assert "prior_claims" in names
    assert result.score == min(100, sum(factor.points for factor in result.factors))
    assert result.score >= 30
