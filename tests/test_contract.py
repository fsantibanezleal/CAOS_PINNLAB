"""CONTRACT 1 (ingestion) tests: valid observation rows validate; bad rows are rejected with a reason; out-of-band
values are flagged (accepted, but recorded). This is the bring-your-own-data gate for inverse cases."""
from pinnlab.io.contract import validate_observations


def test_good_rows_accepted():
    rep = validate_observations([{"case_id": "a", "x0": "0.5", "x1": "0.2", "value": "0.31"}], n_coords=2)
    assert rep.ok and len(rep.accepted) == 1 and not rep.rejected
    assert rep.accepted[0].coords == (0.5, 0.2) and rep.accepted[0].value == 0.31


def test_bad_rows_rejected_not_coerced():
    rows = [
        {"case_id": "nan", "x0": "nan", "x1": "0.2", "value": "0.1"},      # NaN coord
        {"case_id": "oob", "x0": "9e9", "x1": "0.2", "value": "0.1"},      # coord out of range
        {"case_id": "missing", "x0": "0.5", "value": "0.1"},               # missing x1
        {"case_id": "text", "x0": "left", "x1": "0.2", "value": "0.1"},    # non-numeric
    ]
    rep = validate_observations(rows, n_coords=2)
    assert len(rep.accepted) == 0
    assert len(rep.rejected) == len(rows)
    assert all("reason" in r for r in rep.rejected)


def test_outlier_flagged_but_accepted():
    rep = validate_observations(
        [{"case_id": "hot", "x0": "0.5", "x1": "0.5", "value": "1e9"}], n_coords=2, value_range=(-10.0, 10.0)
    )
    assert rep.ok and rep.flagged and "value" in rep.flagged[0]["flag"]
