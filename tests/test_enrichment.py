from underwriting.agents.enrichment import enrichment_agent
from underwriting.bureau import lookup
from underwriting.models import BureauRecord, IntakeResult
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

BUREAU = BureauRecord(credit_band="d", prior_claims_count=6, asset_age_years=70)


def _reply(**overrides) -> dict:
    data = {
        "credit_band": "a",
        "prior_claims_count": 0,
        "asset_age_years": 1,
        "flags": ["high_claims_frequency", "weak_credit", "made_up"],
        "notes": "Frequent claims and a weak credit band.",
    }
    data.update(overrides)
    return data


def test_bureau_returns_no_hit_for_an_unknown_id() -> None:
    record = lookup("missing")
    assert record.credit_band == "no_hit"
    assert lookup("A-100").credit_band == "a"


def test_enrichment_keeps_bureau_facts_and_known_flags() -> None:
    llm = ScriptedClient({"enrichment": _reply()})
    result = enrichment_agent(INTAKE, BUREAU, llm=llm)
    assert result.credit_band == "d"
    assert result.prior_claims_count == 6
    assert result.asset_age_years == 70
    assert result.flags == ["high_claims_frequency", "weak_credit"]
    assert result.notes.startswith("Frequent")


def test_enrichment_flags_a_bureau_miss() -> None:
    miss = BureauRecord(credit_band="no_hit", prior_claims_count=0, asset_age_years=None)
    llm = ScriptedClient({"enrichment": _reply(flags=[], notes="No bureau record.")})
    result = enrichment_agent(INTAKE, miss, llm=llm)
    assert result.credit_band == "no_hit"
    assert result.flags == ["no_bureau_record"]


def test_enrichment_repairs_one_invalid_response() -> None:
    llm = ScriptedClient(
        {"enrichment": [_reply(prior_claims_count="six"), _reply()]}
    )
    result = enrichment_agent(INTAKE, BUREAU, llm=llm)
    assert result.prior_claims_count == 6
    assert llm.calls == ["enrichment", "enrichment"]
