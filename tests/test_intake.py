from underwriting.agents.intake import intake_agent
from underwriting.models import RawApplication
from underwriting.testing_clients import ScriptedClient

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


def _normalized(**overrides) -> dict:
    data = {
        "applicant_id": "CHANGED-BY-MODEL",
        "applicant_name": "Ada Lovelace",
        "date_of_birth": "1990-12-10",
        "address": "1 Analytical Engine Rd",
        "product": "auto",
        "coverage_amount": 250000,
        "stated_income": 80000,
        "claims_history_text": "none",
        "missing_fields": [],
    }
    data.update(overrides)
    return data


def test_intake_normalizes_a_messy_application() -> None:
    llm = ScriptedClient({"intake": _normalized()})
    result = intake_agent(RAW, llm=llm)
    assert result.applicant_id == "A-100"
    assert result.product == "auto"
    assert result.coverage_amount == 250000
    assert result.stated_income == 80000
    assert result.date_of_birth == "1990-12-10"
    assert result.missing_fields == []


def test_intake_flags_a_date_the_model_did_not_fix() -> None:
    llm = ScriptedClient({"intake": _normalized(date_of_birth="not-a-date")})
    result = intake_agent(RAW, llm=llm)
    assert result.date_of_birth is None
    assert result.missing_fields == ["date_of_birth"]
    assert result.product == "auto"


def test_intake_repairs_one_invalid_response() -> None:
    llm = ScriptedClient(
        {
            "intake": [
                _normalized(coverage_amount="250k"),
                _normalized(),
            ]
        }
    )
    result = intake_agent(RAW, llm=llm)
    assert result.coverage_amount == 250000
    assert llm.calls == ["intake", "intake"]
