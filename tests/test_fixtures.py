from underwriting.bureau import lookup
from underwriting.fixtures import CASES


def test_three_applications_hit_different_bureau_records() -> None:
    assert list(CASES) == ["clean_auto", "high_risk_home", "no_hit_life"]
    assert lookup(CASES["clean_auto"].applicant_id).credit_band == "a"
    assert lookup(CASES["high_risk_home"].applicant_id).prior_claims_count == 6
    assert lookup(CASES["no_hit_life"].applicant_id).credit_band == "no_hit"
