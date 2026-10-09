"""Turn raw light curves into fixed-length global and local phase-folded views.

Pipeline: stitch sectors -> mask known transits -> detrend -> fold -> bin.
Everything works on plain numpy arrays so it can be tested without the network.
"""

from __future__ import annotations

import numpy as np

GLOBAL_BINS = 201
LOCAL_BINS = 61
LOCAL_HALF_WIDTH_DURATIONS = 2.0  # local view spans +/- 2 transit durations around mid-transit


def stitch(sectors: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """Median-normalize each sector separately, then concatenate (time-sorted, NaNs dropped).

    Sectors have different absolute flux levels, so each is divided by its own median.
    """
    times, fluxes = [], []
    for time, flux in sectors:
        good = np.isfinite(time) & np.isfinite(flux)
        if not good.any():
            continue
        times.append(time[good])
        fluxes.append(flux[good] / np.median(flux[good]))
    if not times:
        raise ValueError("No finite data in any sector")
    time, flux = np.concatenate(times), np.concatenate(fluxes)
    order = np.argsort(time)
    return time[order], flux[order]


def phase_time(time: np.ndarray, period: float, epoch: float) -> np.ndarray:
    """Time from the nearest mid-transit, in days, in [-period/2, period/2)."""
    return (time - epoch + period / 2) % period - period / 2


def transit_mask(
    time: np.ndarray, period: float, epoch: float, duration: float, pad: float = 1.5
) -> np.ndarray:
    """True for points within ``pad`` x the transit duration of any mid-transit."""
    return np.abs(phase_time(time, period, epoch)) < pad * duration / 2


def detrend(
    time: np.ndarray, flux: np.ndarray, in_transit: np.ndarray, window: float
) -> np.ndarray:
    """Divide out a running median (width ``window`` days) fit only to out-of-transit points.

    The trend is evaluated at out-of-transit times and linearly interpolated across the masked
    transits, so the filter never sees (and never follows) the dip. ``time`` must be sorted.
    """
    oot_t, oot_f = time[~in_transit], flux[~in_transit]
    if len(oot_t) == 0:
        raise ValueError("Every point is in transit; cannot estimate a trend")
    lo = np.searchsorted(oot_t, oot_t - window / 2, side="left")
    hi = np.searchsorted(oot_t, oot_t + window / 2, side="right")
    trend_oot = np.array([np.median(oot_f[a:b]) for a, b in zip(lo, hi)])
    return flux / np.interp(time, oot_t, trend_oot)


def bin_view(
    phase: np.ndarray, flux: np.ndarray, half_width: float, n_bins: int
) -> np.ndarray:
    """Median flux in ``n_bins`` equal bins over [-half_width, half_width].

    Empty bins (gaps in phase coverage) are linearly interpolated from neighbors.
    """
    edges = np.linspace(-half_width, half_width, n_bins + 1)
    idx = np.digitize(phase, edges) - 1
    keep = (idx >= 0) & (idx < n_bins)
    view = np.full(n_bins, np.nan)
    for b in np.unique(idx[keep]):
        view[b] = np.median(flux[keep & (idx == b)])
    filled = np.isfinite(view)
    if not filled.any():
        raise ValueError("No data fell inside the view window")
    centers = (edges[:-1] + edges[1:]) / 2
    return np.interp(centers, centers[filled], view[filled])


def make_views(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    epoch: float,
    duration: float,
    detrend_window: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (global_view, local_view) as ``flux - 1`` (0 = baseline, negative = dip).

    ``duration`` is the catalog transit duration in days. The default detrending window is
    3x the duration, floored at 0.75 day. Depth scaling is left to the model stage so any
    normalization statistics can be fit on the training set only.
    """
    window = detrend_window if detrend_window is not None else max(3 * duration, 0.75)
    flat = detrend(time, flux, transit_mask(time, period, epoch, duration), window)
    phase = phase_time(time, period, epoch)
    global_view = bin_view(phase, flat, period / 2, GLOBAL_BINS) - 1.0
    local_view = bin_view(
        phase, flat, LOCAL_HALF_WIDTH_DURATIONS * duration, LOCAL_BINS
    ) - 1.0
    return global_view, local_view
