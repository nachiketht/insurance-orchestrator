"""Three applications used for the full-pipeline traces."""

from underwriting.models import RawApplication

CASES = {
    "clean_auto": RawApplication(
        applicant_id="A-100",
        applicant_name="Ada Lovelace",
        date_of_birth="12/10/1990",
        address="1 Analytical Engine Rd",
        product="car insurance",
        coverage_amount="$250k",
        stated_income="$80,000",
        claims_history_text="none",
    ),
    "high_risk_home": RawApplication(
        applicant_id="A-200",
        applicant_name="Grace Hopper",
        date_of_birth="01/01/1970",
        address="1 Compiler Way",
        product="house insurance",
        coverage_amount="$600k",
        stated_income="$50,000",
        claims_history_text="three water claims in two years",
    ),
    "no_hit_life": RawApplication(
        applicant_id="A-999",
        applicant_name="Alan Turing",
        date_of_birth="06/23/1988",
        address="1 Bletchley Park",
        product="life insurance",
        coverage_amount="$500k",
        stated_income="$90,000",
        claims_history_text="none",
    ),
}
