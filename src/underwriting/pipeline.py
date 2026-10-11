"""Run the underwriting steps in order. Only this module writes the case record."""

from underwriting.agents.enrichment import enrichment_agent
from underwriting.agents.intake import intake_agent
from underwriting.bureau import lookup
from underwriting.llm import LLMClient
from underwriting.models import CaseRecord, RawApplication


def run_case(raw: RawApplication, *, llm: LLMClient) -> CaseRecord:
    """Intake, then enrichment. Later steps are added once this handoff is confirmed."""
    case = CaseRecord.new(raw)
    case.intake = intake_agent(raw, llm=llm)
    case.enrichment = enrichment_agent(case.intake, lookup(raw.applicant_id), llm=llm)
    return case
