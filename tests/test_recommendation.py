from underwriting.agents.recommendation import recommendation_agent
from underwriting.models import EnrichmentResult, IntakeResult, RiskResult
from underwriting.testing_clients import ScriptedClient

INTAKE = IntakeResult(
    applicant_id="A-100",
    applicant_name="Ada Lovelace",
    date_of_birth="1990-12-10",
    address="1 Analytical Engine Rd",
    product="auto",
    coverage_amount=250000,
    stated_income=80000,
    claims_history_text="none",
    missing_fields=[],
)

ENRICHMENT = EnrichmentResult(
    credit_band="a",
    prior_claims_count=0,
    asset_age_years=6,
    flags=[],
    notes="Clean file.",
)


def _risk(score: int) -> RiskResult:
    return RiskResult(factors=[], score=score, rationale="scripted")


def _reply(decision: str, **overrides) -> dict:
    data = {
        "decision": decision,
        "reasons": ["Score is within the band for this decision."],
        "conditions": [],
        "source": "agent",
    }
    data.update(overrides)
    return data


def test_low_score_can_stay_approved() -> None:
    llm = ScriptedClient({"recommendation": _reply("approve")})
    result = recommendation_agent(INTAKE, ENRICHMENT, _risk(20), llm=llm)
    assert result.decision == "approve"
    assert result.source == "agent"


def test_model_may_refer_an_approval_but_not_deny_it() -> None:
    referred = recommendation_agent(
        INTAKE,
        ENRICHMENT,
        _risk(20),
        llm=ScriptedClient({"recommendation": _reply("refer")}),
    )
    assert referred.decision == "refer"

    denied = recommendation_agent(
        INTAKE,
        ENRICHMENT,
        _risk(20),
        llm=ScriptedClient({"recommendation": _reply("deny")}),
    )
    assert denied.decision == "approve"
    assert any("held at approve" in reason for reason in denied.reasons)


def test_middle_score_stays_referred() -> None:
    llm = ScriptedClient({"recommendation": _reply("approve")})
    result = recommendation_agent(INTAKE, ENRICHMENT, _risk(50), llm=llm)
    assert result.decision == "refer"


def test_high_score_stays_denied() -> None:
    llm = ScriptedClient({"recommendation": _reply("approve")})
    result = recommendation_agent(INTAKE, ENRICHMENT, _risk(80), llm=llm)
    assert result.decision == "deny"


def test_missing_bureau_record_is_referred() -> None:
    miss = ENRICHMENT.model_copy(update={"credit_band": "no_hit", "flags": ["no_bureau_record"]})
    llm = ScriptedClient({"recommendation": _reply("approve")})
    result = recommendation_agent(INTAKE, miss, _risk(10), llm=llm)
    assert result.decision == "refer"


def test_recommendation_repairs_one_invalid_response() -> None:
    llm = ScriptedClient(
        {"recommendation": [_reply("approve", reasons="not a list"), _reply("approve")]}
    )
    result = recommendation_agent(INTAKE, ENRICHMENT, _risk(20), llm=llm)
    assert result.decision == "approve"
    assert llm.calls == ["recommendation", "recommendation"]
