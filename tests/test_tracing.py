import json
import subprocess
import sys
from pathlib import Path

from underwriting.models import RawApplication
from underwriting.pipeline import run_case
from underwriting.testing_clients import ScriptedClient
from underwriting.tracing import Tracer

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
    "credit_band": "a",
    "prior_claims_count": 0,
    "asset_age_years": 6,
    "flags": [],
    "notes": "Clean file.",
}

RISK = {
    "factors": [{"name": "prior_claims", "points": 15, "detail": "scripted"}],
    "score": 99,
    "rationale": "Scripted.",
}

RECOMMENDATION = {
    "decision": "approve",
    "reasons": ["Low score."],
    "conditions": [],
    "source": "agent",
}

REQUIRED = (
    "step",
    "agent",
    "timestamp",
    "input",
    "input_source",
    "output",
    "duration_ms",
    "tokens_used",
    "cost_usd",
)


def test_trace_records_both_steps_and_the_handoff() -> None:
    llm = ScriptedClient(
        {
            "intake": INTAKE,
            "enrichment": ENRICHMENT,
            "risk_scoring": RISK,
            "recommendation": RECOMMENDATION,
        }
    )
    tracer = Tracer()
    case = run_case(RAW, llm=llm, tracer=tracer)
    payload = tracer.dump(case.case_id, case.model_dump(mode="json"))

    assert [span["agent"] for span in payload["spans"]] == [
        "intake",
        "enrichment",
        "risk_scoring",
        "recommendation",
    ]
    for span in payload["spans"]:
        for field in REQUIRED:
            assert field in span
        assert span["tokens_used"] == 160
        assert span["cost_usd"] == 0.0
        assert span["duration_ms"] >= 0
        assert span["output"] is not None

    enrichment = payload["spans"][1]
    assert enrichment["input_source"] == ["case.intake", "bureau"]
    assert enrichment["input"]["intake"]["product"] == "auto"
    assert enrichment["input"]["intake"]["coverage_amount"] == 250000
    risk = payload["spans"][2]
    assert risk["input_source"] == ["case.intake", "case.enrichment"]
    assert risk["input"]["enrichment"]["credit_band"] == "a"
    assert risk["output"]["score"] == 15
    advice = payload["spans"][3]
    assert advice["input_source"] == ["case.intake", "case.enrichment", "case.risk"]
    assert advice["input"]["risk"]["score"] == 15
    assert advice["output"]["decision"] == "approve"
    path = Path("traces") / "handoff-test.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(payload))
    shown = subprocess.run(
        [sys.executable, "scripts/show_trace.py", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "tokens_used=160" in shown.stdout
    assert "intake.product=auto" in shown.stdout
    path.unlink()


def test_repair_call_is_a_second_entry_on_the_same_step() -> None:
    llm = ScriptedClient(
        {
            "intake": [dict(INTAKE, coverage_amount="250k"), INTAKE],
            "enrichment": ENRICHMENT,
            "risk_scoring": RISK,
            "recommendation": RECOMMENDATION,
        }
    )
    tracer = Tracer()
    run_case(RAW, llm=llm, tracer=tracer)
    intake = tracer.spans[0]
    assert len(intake.calls) == 2
    assert intake.tokens_used == 320
    assert intake.status == "ok"
