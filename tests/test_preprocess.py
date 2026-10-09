import numpy as np
import pytest

from transit_classifier import preprocess as pp

PERIOD, EPOCH, DURATION, DEPTH = 3.0, 1.0, 0.15, 0.01


def _synthetic(seed=0, trend=True):
    """27 days of 2-minute data: slow sinusoidal trend + box transit + small noise."""
    rng = np.random.default_rng(seed)
    time = np.arange(0, 27, 2 / 1440)
    flux = np.ones_like(time)
    if trend:
        flux += 0.02 * np.sin(2 * np.pi * time / 10)
    in_tr = pp.transit_mask(time, PERIOD, EPOCH, DURATION, pad=1.0)
    flux[in_tr] -= DEPTH
    flux += rng.normal(0, 0.0005, len(time))
    return time, flux


def test_stitch_normalizes_each_sector_and_drops_nans():
    a = (np.array([0.0, 1.0, 2.0]), np.array([100.0, 100.0, np.nan]))
    b = (np.array([3.0, 4.0]), np.array([5000.0, 5000.0]))
    time, flux = pp.stitch([b, a])
    assert list(time) == [0.0, 1.0, 3.0, 4.0]
    assert np.allclose(flux, 1.0)


def test_transit_mask_flags_only_points_near_mid_transit():
    time = np.array([EPOCH, EPOCH + PERIOD, EPOCH + 1.5])
    mask = pp.transit_mask(time, PERIOD, EPOCH, DURATION)
    assert list(mask) == [True, True, False]


def test_detrending_preserves_transit_depth():
    time, flux = _synthetic()
    mask = pp.transit_mask(time, PERIOD, EPOCH, DURATION)
    flat = pp.detrend(time, flux, mask, window=0.75)
    in_tr = pp.transit_mask(time, PERIOD, EPOCH, DURATION, pad=0.8)
    assert 1 - np.median(flat[in_tr]) == pytest.approx(DEPTH, rel=0.1)
    assert np.std(flat[~mask]) < 0.002  # the 2% sinusoid is gone


def test_unmasked_detrending_eats_the_transit():
    """The reason for the mask: a short filter that sees the dip follows it down."""
    time, flux = _synthetic()
    no_mask = np.zeros_like(time, dtype=bool)
    flat = pp.detrend(time, flux, no_mask, window=0.3)
    in_tr = pp.transit_mask(time, PERIOD, EPOCH, DURATION, pad=0.8)
    assert 1 - np.median(flat[in_tr]) < 0.5 * DEPTH


def test_make_views_shapes_and_dip_at_center():
    time, flux = _synthetic()
    g, loc = pp.make_views(time, flux, PERIOD, EPOCH, DURATION)
    assert g.shape == (pp.GLOBAL_BINS,) and loc.shape == (pp.LOCAL_BINS,)
    assert np.isfinite(g).all() and np.isfinite(loc).all()
    assert loc[pp.LOCAL_BINS // 2] == pytest.approx(-DEPTH, rel=0.15)
    assert abs(loc[0]) < 0.002  # edges of the local view are baseline
    assert g[pp.GLOBAL_BINS // 2] < -DEPTH / 2


def test_bin_view_fills_empty_bins():
    phase = np.array([-0.9, 0.9])
    view = pp.bin_view(phase, np.array([1.0, 3.0]), half_width=1.0, n_bins=10)
    assert np.isfinite(view).all()
    assert view[0] == 1.0 and view[-1] == 3.0
