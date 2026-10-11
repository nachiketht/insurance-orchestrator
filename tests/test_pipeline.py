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


def test_intake_output_is_what_enrichment_receives() -> None:
    llm = ScriptedClient({"intake": INTAKE, "enrichment": ENRICHMENT})
    case = run_case(RAW, llm=llm)

    assert llm.calls == ["intake", "enrichment"]
    assert case.intake is not None
    assert case.intake.product == "auto"
    assert case.intake.applicant_id == "A-100"
    assert case.risk is None
    assert case.recommendation is None

    sent = json.loads(llm.prompts[1][1])
    assert sent["intake"]["product"] == "auto"
    assert sent["intake"]["coverage_amount"] == 250000
    assert sent["intake"]["applicant_id"] == "A-100"
    assert sent["bureau"]["credit_band"] == "a"
    assert case.enrichment is not None
    assert case.enrichment.credit_band == "a"
    assert case.enrichment.prior_claims_count == 0
