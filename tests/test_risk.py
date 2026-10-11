from underwriting.agents.risk import risk_scoring_agent
from underwriting.models import EnrichmentResult, IntakeResult
from underwriting.testing_clients import ScriptedClient

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


def _reply(**overrides) -> dict:
    data = {
        "factors": [
            {"name": "prior_claims", "points": 30, "detail": "Six prior claims."},
            {"name": "credit", "points": 25, "detail": "Credit band d."},
            {"name": "prior_claims", "points": 30, "detail": "Counted twice."},
            {"name": "not_a_factor", "points": 40, "detail": "Ignore me."},
        ],
        "score": 99,
        "rationale": "Claims and credit dominate.",
    }
    data.update(overrides)
    return data


def test_risk_score_is_the_sum_of_allowed_factors() -> None:
    llm = ScriptedClient({"risk_scoring": _reply()})
    result = risk_scoring_agent(INTAKE, ENRICHMENT, llm=llm)
    assert [factor.name for factor in result.factors] == ["prior_claims", "credit"]
    assert result.score == 55
    assert result.rationale.startswith("Claims")


def test_risk_score_caps_at_100() -> None:
    factors = [
        {"name": "prior_claims", "points": 30, "detail": "claims"},
        {"name": "credit", "points": 25, "detail": "credit"},
        {"name": "coverage_income", "points": 20, "detail": "coverage"},
        {"name": "asset_age", "points": 15, "detail": "age"},
        {"name": "missing_data", "points": 20, "detail": "missing"},
    ]
    llm = ScriptedClient({"risk_scoring": _reply(factors=factors, score=0)})
    result = risk_scoring_agent(INTAKE, ENRICHMENT, llm=llm)
    assert result.score == 100


def test_risk_repairs_one_invalid_response() -> None:
    broken = _reply(
        factors=[{"name": "prior_claims", "points": 80, "detail": "too high"}]
    )
    llm = ScriptedClient({"risk_scoring": [broken, _reply()]})
    result = risk_scoring_agent(INTAKE, ENRICHMENT, llm=llm)
    assert result.score == 55
    assert llm.calls == ["risk_scoring", "risk_scoring"]
