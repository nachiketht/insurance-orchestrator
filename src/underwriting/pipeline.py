"""Run the underwriting steps in order. Only this module writes the case record."""

from contextlib import contextmanager
from typing import Iterator

from underwriting.agents.enrichment import enrichment_agent
from underwriting.agents.intake import intake_agent
from underwriting.agents.risk import risk_scoring_agent
from underwriting.bureau import lookup
from underwriting.llm import LLMClient
from underwriting.models import CaseRecord, RawApplication
from underwriting.tracing import Span, TracedLLM, Tracer


class _Quiet:
    output: dict | None = None

    def __enter__(self) -> "_Quiet":
        return self

    def __exit__(self, *args: object) -> bool:
        return False


@contextmanager
def _open(
    tracer: Tracer | None,
    step: int,
    agent: str,
    source: list[str],
    received: dict,
) -> Iterator[Span | _Quiet]:
    if tracer is None:
        yield _Quiet()
        return
    with tracer.step(step, agent, source, received) as span:
        yield span


def run_case(
    raw: RawApplication,
    *,
    llm: LLMClient,
    tracer: Tracer | None = None,
) -> CaseRecord:
    """Intake, enrichment, then risk scoring. Recommendation is added after this handoff."""
    case = CaseRecord.new(raw)
    client = TracedLLM(llm, tracer) if tracer is not None else llm
    bureau = lookup(raw.applicant_id)

    with _open(tracer, 1, "intake", ["raw_application"], raw.model_dump()) as span:
        case.intake = intake_agent(raw, llm=client)
        span.output = case.intake.model_dump(mode="json")

    received = {"intake": case.intake.model_dump(mode="json"), "bureau": bureau.model_dump()}
    with _open(tracer, 2, "enrichment", ["case.intake", "bureau"], received) as span:
        case.enrichment = enrichment_agent(case.intake, bureau, llm=client)
        span.output = case.enrichment.model_dump(mode="json")

    received = {
        "intake": case.intake.model_dump(mode="json"),
        "enrichment": case.enrichment.model_dump(mode="json"),
    }
    with _open(tracer, 3, "risk_scoring", ["case.intake", "case.enrichment"], received) as span:
        case.risk = risk_scoring_agent(case.intake, case.enrichment, llm=client)
        span.output = case.risk.model_dump(mode="json")

    return case
