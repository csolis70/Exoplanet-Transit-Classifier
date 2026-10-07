import pandas as pd
import pytest

from transit_classifier.catalog import COLUMNS, clean_toi_catalog


def _raw(rows):
    return pd.DataFrame(rows, columns=list(COLUMNS))


def test_labels_follow_disposition():
    raw = _raw(
        [
            [1, 10, "CP", 2.0, 2459000.0, 2.0, 1000, 10.0],
            [2, 11, "kp", 3.0, 2459000.0, 2.0, 1000, 10.0],
            [3, 12, "FP", 4.0, 2459000.0, 2.0, 1000, 10.0],
            [4, 13, "FA", 5.0, 2459000.0, 2.0, 1000, 10.0],
            [5, 14, "PC", 6.0, 2459000.0, 2.0, 1000, 10.0],
            [6, 15, "APC", 7.0, 2459000.0, 2.0, 1000, 10.0],
        ]
    )
    out = clean_toi_catalog(raw)
    assert out["label"].iloc[:4].tolist() == [1.0, 1.0, 0.0, 0.0]
    assert out["label"].iloc[4:].isna().all()


def test_unfoldable_rows_are_dropped():
    raw = _raw(
        [
            [1, 10, "CP", None, 2459000.0, 2.0, 1000, 10.0],
            [2, 11, "CP", 3.0, None, 2.0, 1000, 10.0],
            [3, 12, "CP", 4.0, 2459000.0, 2.0, 1000, 10.0],
        ]
    )
    assert clean_toi_catalog(raw)["toi"].tolist() == [3]


def test_missing_columns_raise():
    with pytest.raises(ValueError):
        clean_toi_catalog(pd.DataFrame({"toi": [1]}))
