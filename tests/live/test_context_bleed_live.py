import json

import pytest

from underwriting.errors import LLMTimeoutError, LLMTransientError, ModelNotFoundError
from underwriting.llm import OllamaClient
from underwriting.models import RawApplication
from underwriting.pipeline import run_case
from underwriting.tracing import Tracer

pytestmark = pytest.mark.live

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


def test_live_second_case_does_not_contain_the_first() -> None:
    client = OllamaClient()
    tracer = Tracer()
    try:
        client.preflight()
        client.warm_up()
        first = run_case(RAW_A, llm=client, tracer=tracer)
        second = run_case(RAW_B, llm=client, tracer=tracer)
    except (ModelNotFoundError, LLMTransientError, LLMTimeoutError) as exc:
        pytest.skip(f"Ollama is not ready: {exc}")
    finally:
        client.close()

    blob = json.dumps(second.model_dump(mode="json"))
    assert "Ada Lovelace" not in blob
    assert "A-100" not in blob
    assert second.intake is not None
    assert second.intake.applicant_id == "A-200"
    later = json.dumps([span.as_dict() for span in tracer.spans[4:]])
    assert first.case_id not in later
    assert "Ada Lovelace" not in later
