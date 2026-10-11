import json

from underwriting.models import RawApplication
from underwriting.pipeline import run_case
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

INTAKE = {
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

ENRICHMENT = {
    "credit_band": "z",
    "prior_claims_count": 9,
    "asset_age_years": 99,
    "flags": ["high_claims_frequency"],
    "notes": "Clean file.",
}

RISK = {
    "factors": [
        {"name": "prior_claims", "points": 15, "detail": "scripted"},
        {"name": "not_a_factor", "points": 40, "detail": "drop"},
    ],
    "score": 99,
    "rationale": "Scripted.",
}


def test_each_step_receives_the_previous_output() -> None:
    llm = ScriptedClient({"intake": INTAKE, "enrichment": ENRICHMENT, "risk_scoring": RISK})
    case = run_case(RAW, llm=llm)

    assert llm.calls == ["intake", "enrichment", "risk_scoring"]
    assert case.intake is not None
    assert case.intake.product == "auto"
    assert case.intake.applicant_id == "A-100"
    assert case.recommendation is None

    sent = json.loads(llm.prompts[1][1])
    assert sent["intake"]["product"] == "auto"
    assert sent["intake"]["coverage_amount"] == 250000
    assert sent["intake"]["applicant_id"] == "A-100"
    assert sent["bureau"]["credit_band"] == "a"
    assert case.enrichment is not None
    assert case.enrichment.credit_band == "a"
    assert case.enrichment.prior_claims_count == 0

    scored = json.loads(llm.prompts[2][1])
    assert scored["intake"]["product"] == "auto"
    assert scored["enrichment"]["credit_band"] == "a"
    assert scored["enrichment"]["prior_claims_count"] == 0
    assert case.risk is not None
    assert case.risk.score == 15
