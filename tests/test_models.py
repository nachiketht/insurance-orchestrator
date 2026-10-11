from underwriting.models import CaseRecord, RawApplication, RiskFactor


def test_new_case_has_its_own_id_and_no_sections() -> None:
    raw = RawApplication(
        applicant_id="A-1",
        applicant_name="Ada Lovelace",
        date_of_birth="12/10/1990",
        address="1 Analytical Engine Rd",
        product="car insurance",
        coverage_amount="$250k",
    )
    case = CaseRecord.new(raw)
    assert case.case_id
    assert case.status == "in_progress"
    assert case.intake is None
    assert case.recommendation is None


def test_risk_factor_rejects_points_above_40() -> None:
    try:
        RiskFactor(name="prior_claims", points=41, detail="too many")
    except ValueError:
        return
    raise AssertionError("points above 40 should be rejected")
