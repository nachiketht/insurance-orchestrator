import pytest

from underwriting.agents.recommendation import recommendation_agent
from underwriting.errors import LLMTimeoutError, LLMTransientError, ModelNotFoundError
from underwriting.llm import OllamaClient
from underwriting.models import EnrichmentResult, IntakeResult, RiskFactor, RiskResult

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
    flags=["high_claims_frequency", "weak_credit"],
    notes="Frequent claims and weak credit.",
)

RISK = RiskResult(
    factors=[
        RiskFactor(name="prior_claims", points=30, detail="Six prior claims."),
        RiskFactor(name="credit", points=25, detail="Credit band d."),
        RiskFactor(name="coverage_income", points=20, detail="Coverage is 12 times income."),
    ],
    score=75,
    rationale="Claims, credit, and coverage all add risk.",
)


def test_live_recommendation_denies_a_high_score() -> None:
    client = OllamaClient()
    try:
        client.preflight()
        client.warm_up()
        result = recommendation_agent(INTAKE, ENRICHMENT, RISK, llm=client)
    except (ModelNotFoundError, LLMTransientError, LLMTimeoutError) as exc:
        pytest.skip(f"Ollama is not ready: {exc}")
    finally:
        client.close()

    assert result.decision == "deny"
    assert result.reasons
    assert result.source == "agent"
