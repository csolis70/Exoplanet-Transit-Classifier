import pandas as pd
import pytest

from transit_classifier import lightcurves
from transit_classifier.sampling import select_pilot_sample


def _catalog():
    rows = []
    for i in range(10):  # 10 positive stars, 10 negative stars
        rows.append({"toi": i, "tic_id": 100 + i, "label": 1.0})
        rows.append({"toi": 50 + i, "tic_id": 200 + i, "label": 0.0})
    rows.append({"toi": 90, "tic_id": 100, "label": 1.0})  # second TOI on a positive star
    rows.append({"toi": 91, "tic_id": 101, "label": 0.0})  # star 101 now has conflicting labels
    rows.append({"toi": 92, "tic_id": 300, "label": float("nan")})  # unlabeled
    return pd.DataFrame(rows)


def test_pilot_sample_is_balanced_unique_and_consistent():
    out = select_pilot_sample(_catalog(), n_per_class=5, seed=1)
    assert out["label"].value_counts().to_dict() == {1.0: 5, 0.0: 5}
    assert out["tic_id"].is_unique
    assert 101 not in set(out["tic_id"])  # conflicting labels excluded
    assert 300 not in set(out["tic_id"])  # unlabeled excluded


def test_pilot_sample_is_reproducible():
    a = select_pilot_sample(_catalog(), n_per_class=5, seed=3)
    b = select_pilot_sample(_catalog(), n_per_class=5, seed=3)
    pd.testing.assert_frame_equal(a, b)


def test_pilot_sample_raises_when_too_few_stars():
    with pytest.raises(ValueError):
        select_pilot_sample(_catalog(), n_per_class=50)


def test_download_star_raises_when_no_data(monkeypatch):
    monkeypatch.setattr(lightcurves.lk, "search_lightcurve", lambda *a, **k: [])
    with pytest.raises(lightcurves.NoLightCurveError):
        lightcurves.download_star(123)
