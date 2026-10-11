"""Local stand-in for a credit and claims bureau. Unknown ids are a no-hit."""

from underwriting.models import BureauRecord

_RECORDS = {
    "A-100": BureauRecord(credit_band="a", prior_claims_count=0, asset_age_years=6),
    "A-200": BureauRecord(credit_band="d", prior_claims_count=6, asset_age_years=70),
}

_NO_HIT = BureauRecord(credit_band="no_hit", prior_claims_count=0, asset_age_years=None)


def lookup(applicant_id: str) -> BureauRecord:
    return _RECORDS.get(applicant_id, _NO_HIT)
