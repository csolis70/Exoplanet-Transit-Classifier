"""Download TESS SPOC 2-minute PDCSAP light curves (one per sector) for a star."""

from __future__ import annotations

import logging
from pathlib import Path

import lightkurve as lk

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path("data/raw/lightcurves")


class NoLightCurveError(Exception):
    """Raised when a star has no SPOC 2-minute light curves."""


def download_star(
    tic_id: int, cache_dir: Path = DEFAULT_CACHE_DIR
) -> lk.LightCurveCollection:
    """Return one light curve per available sector for ``tic_id``.

    Uses SPOC 2-minute cadence and the PDCSAP flux (instrument-corrected). Lightkurve's default
    quality mask drops flagged cadences. FITS files are cached under ``cache_dir``.
    """
    results = lk.search_lightcurve(
        f"TIC {tic_id}", mission="TESS", author="SPOC", exptime=120
    )
    if len(results) == 0:
        raise NoLightCurveError(f"No SPOC 2-minute light curves for TIC {tic_id}")
    logger.info("TIC %d: downloading %d sectors", tic_id, len(results))
    cache_dir.mkdir(parents=True, exist_ok=True)
    return results.download_all(flux_column="pdcsap_flux", download_dir=str(cache_dir))
