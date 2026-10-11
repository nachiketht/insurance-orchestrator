"""Case record and the result each agent is allowed to write."""

from uuid import uuid4

from pydantic import BaseModel, Field


class RawApplication(BaseModel):
    """Applicant input, kept as submitted. Amounts and dates may be messy."""

    applicant_id: str
    applicant_name: str
    date_of_birth: str
    address: str
    product: str
    coverage_amount: str
    stated_income: str | None = None
    claims_history_text: str | None = None


class IntakeResult(BaseModel):
    applicant_id: str
    applicant_name: str
    date_of_birth: str | None
    address: str
    product: str | None  # "auto", "home", "life", or null when it could not be normalized
    coverage_amount: int | None
    stated_income: int | None = None
    claims_history_text: str | None = None
    missing_fields: list[str] = Field(default_factory=list)


class BureauRecord(BaseModel):
    credit_band: str  # "a" through "d", or "no_hit"
    prior_claims_count: int
    asset_age_years: int | None = None


class EnrichmentResult(BaseModel):
    credit_band: str
    prior_claims_count: int
    asset_age_years: int | None = None
    flags: list[str] = Field(default_factory=list)
    notes: str


class RiskFactor(BaseModel):
    name: str
    points: int = Field(ge=0, le=40)
    detail: str


class RiskResult(BaseModel):
    factors: list[RiskFactor]
    score: int = Field(ge=0, le=100)
    rationale: str


class RecommendationResult(BaseModel):
    decision: str  # "approve", "deny", or "refer"
    reasons: list[str] = Field(default_factory=list)
    conditions: list[str] = Field(default_factory=list)
    source: str = "agent"  # "agent" or "system_escalation"


class Escalation(BaseModel):
    step: str
    error_type: str
    message: str
    attempts: int


class CaseRecord(BaseModel):
    case_id: str
    raw: RawApplication
    intake: IntakeResult | None = None
    enrichment: EnrichmentResult | None = None
    risk: RiskResult | None = None
    recommendation: RecommendationResult | None = None
    status: str = "in_progress"  # "in_progress", "completed", or "escalated"
    escalation: Escalation | None = None

    @classmethod
    def new(cls, raw: RawApplication) -> "CaseRecord":
        return cls(case_id=uuid4().hex, raw=raw)
