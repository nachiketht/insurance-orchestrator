import json

from underwriting.models import RawApplication
from underwriting.pipeline import run_case
from underwriting.testing_clients import ScriptedClient
from underwriting.tracing import Tracer, _current

RAW_A = RawApplication(
    applicant_id="A-100",
    applicant_name="Ada Lovelace",
    date_of_birth="12/10/1990",
    address="1 Analytical Engine Rd",
    product="car insurance",
    coverage_amount="$250k",
    stated_income="$80,000",
    claims_history_text="none",
)

RAW_B = RawApplication(
    applicant_id="A-200",
    applicant_name="Grace Hopper",
    date_of_birth="01/01/1970",
    address="1 Compiler Way",
    product="house insurance",
    coverage_amount="$600k",
    stated_income="$50,000",
    claims_history_text="three water claims",
)

INTAKE_A = {
    "applicant_id": "CHANGED",
    "applicant_name": "Ada Lovelace",
    "date_of_birth": "1990-12-10",
    "address": "1 Analytical Engine Rd",
    "product": "auto",
    "coverage_amount": 250000,
    "stated_income": 80000,
    "claims_history_text": "none",
    "missing_fields": [],
}

INTAKE_B = {
    "applicant_id": "CHANGED",
    "applicant_name": "Grace Hopper",
    "date_of_birth": "1970-01-01",
    "address": "1 Compiler Way",
    "product": "home",
    "coverage_amount": 600000,
    "stated_income": 50000,
    "claims_history_text": "three water claims",
    "missing_fields": [],
}

ENRICHMENT = {
    "credit_band": "a",
    "prior_claims_count": 0,
    "asset_age_years": 6,
    "flags": [],
    "notes": "Bureau facts are copied by code.",
}

RISK = {
    "factors": [{"name": "prior_claims", "points": 15, "detail": "scripted"}],
    "score": 0,
    "rationale": "Scripted.",
}

RECOMMENDATION = {
    "decision": "approve",
    "reasons": ["Scripted."],
    "conditions": [],
    "source": "agent",
}


def test_second_case_does_not_receive_the_first() -> None:
    llm = ScriptedClient(
        {
            "intake": [INTAKE_A, INTAKE_B],
            "enrichment": [ENRICHMENT, ENRICHMENT],
            "risk_scoring": [RISK, RISK],
            "recommendation": [RECOMMENDATION, RECOMMENDATION],
        }
    )
    tracer = Tracer()
    first = run_case(RAW_A, llm=llm, tracer=tracer)
    second = run_case(RAW_B, llm=llm, tracer=tracer)

    assert first.case_id != second.case_id
    assert second.intake is not None
    assert second.intake.applicant_id == "A-200"
    assert second.intake.applicant_name == "Grace Hopper"

    later = "\n".join(user for _, user in llm.prompts[4:])
    assert "Ada Lovelace" not in later
    assert "A-100" not in later
    assert first.case_id not in later

    span_text = json.dumps([span.as_dict() for span in tracer.spans[4:]])
    assert "Ada Lovelace" not in span_text
    assert "A-100" not in span_text
    assert first.case_id not in span_text
    assert _current.get() is None
