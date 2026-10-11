import json
from pathlib import Path

import pytest

from underwriting.errors import LLMTimeoutError, LLMTransientError, ModelNotFoundError
from underwriting.llm import OllamaClient
from underwriting.models import RawApplication
from underwriting.pipeline import run_case
from underwriting.tracing import Tracer

pytestmark = pytest.mark.live

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


def test_live_trace_has_tokens_for_both_steps(tmp_path: Path) -> None:
    client = OllamaClient()
    tracer = Tracer()
    try:
        client.preflight()
        client.warm_up()
        case = run_case(RAW, llm=client, tracer=tracer)
    except (ModelNotFoundError, LLMTransientError, LLMTimeoutError) as exc:
        pytest.skip(f"Ollama is not ready: {exc}")
    finally:
        client.close()

    payload = tracer.dump(case.case_id, case.model_dump(mode="json"))
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(payload))
    assert [span["agent"] for span in payload["spans"]] == ["intake", "enrichment"]
    for span in payload["spans"]:
        assert span["tokens_used"] > 0
        assert span["duration_ms"] > 0
        assert span["cost_usd"] == 0.0
        assert span["output"]
    assert payload["spans"][1]["input"]["intake"]["product"] == "auto"
