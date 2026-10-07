"""Download and clean the NASA Exoplanet Archive TOI (TESS Object of Interest) catalog.

Labels are catalog dispositions, not ground truth: CP/KP are positive, FP/FA are negative, and
unresolved PC/APC rows get a missing label so they never leak into training by accident.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

# Archive column -> our column name.
COLUMNS = {
    "toi": "toi",
    "tid": "tic_id",
    "tfopwg_disp": "disposition",
    "pl_orbper": "period_days",
    "pl_tranmid": "epoch_bjd",
    "pl_trandurh": "duration_hours",
    "pl_trandep": "depth_ppm",
    "st_tmag": "tmag",
}

POSITIVE = {"CP", "KP"}
NEGATIVE = {"FP", "FA"}

DEFAULT_RAW_PATH = Path("data/raw/toi_catalog.csv")


def build_query_url() -> str:
    """TAP query URL returning the selected TOI columns as CSV."""
    cols = ",".join(COLUMNS)
    return f"{TAP_URL}?query=select+{cols}+from+toi&format=csv"


def fetch_toi_catalog(path: Path = DEFAULT_RAW_PATH, refresh: bool = False) -> pd.DataFrame:
    """Return the raw TOI table, downloading it only if not cached at ``path``."""
    if path.exists() and not refresh:
        logger.info("Using cached TOI catalog at %s", path)
        return pd.read_csv(path)
    logger.info("Downloading TOI catalog from the NASA Exoplanet Archive")
    df = pd.read_csv(build_query_url())
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def clean_toi_catalog(raw: pd.DataFrame) -> pd.DataFrame:
    """Rename columns, drop rows we cannot fold, and add a binary ``label``.

    ``label`` is 1 for CP/KP, 0 for FP/FA, and NaN for PC/APC or anything unrecognized.
    """
    missing = set(COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f"TOI table is missing expected columns: {sorted(missing)}")

    df = raw[list(COLUMNS)].rename(columns=COLUMNS).copy()
    df["disposition"] = df["disposition"].str.strip().str.upper()

    # Without a positive period and an epoch we cannot phase-fold the signal.
    foldable = (df["period_days"] > 0) & df["epoch_bjd"].notna() & df["tic_id"].notna()
    logger.info("Dropping %d rows without a usable period/epoch/TIC", int((~foldable).sum()))
    df = df[foldable].copy()

    df["tic_id"] = df["tic_id"].astype("int64")
    df["label"] = np.select(
        [df["disposition"].isin(POSITIVE), df["disposition"].isin(NEGATIVE)],
        [1.0, 0.0],
        default=np.nan,
    )
    return df.reset_index(drop=True)
