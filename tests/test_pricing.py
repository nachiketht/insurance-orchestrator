from underwriting.analysis import load_cases, render
from underwriting.pricing import hosted_cost


def test_hosted_cost_matches_the_hand_calculation() -> None:
    # clean_auto input and output token totals
    assert hosted_cost(1229, 393) == (1229 * 0.10 + 393 * 0.50) / 1_000_000


def test_analysis_uses_the_three_live_traces() -> None:
    cases = load_cases()
    assert [case["fixture"] for case in cases] == [
        "clean_auto",
        "high_risk_home",
        "no_hit_life",
    ]
    text = render(cases)
    assert "Real time fails" in text
    assert "batch fails" in text
    assert "$0.000319" in text
