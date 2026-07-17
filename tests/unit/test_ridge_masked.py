import numpy as np
import pytest
import wavesst
from wavesst.analysis.ridge import extract_ridges_masked, Ridge
from wavesst.transforms.sst import sst

FS = 256.0
N = 512


def _make_sst(f0: float, cfg):
    t = np.arange(N) / FS
    x = np.cos(2 * np.pi * f0 * t).astype(np.float32)
    return sst(x, fs=FS, nv=32, gamma='auto', cfg=cfg)


def test_masked_ridge_returns_ridge_list(cfg):
    result = _make_sst(40.0, cfg)
    n_freqs = len(result.freqs)
    n_times = len(result.times)
    mask = np.ones((n_freqs, n_times), dtype=bool)
    ridges = extract_ridges_masked(result, mask, n=1)
    assert isinstance(ridges, list)
    assert len(ridges) == 1
    assert isinstance(ridges[0], Ridge)


def test_masked_ridge_shape(cfg):
    result = _make_sst(40.0, cfg)
    n_freqs = len(result.freqs)
    n_times = len(result.times)
    mask = np.ones((n_freqs, n_times), dtype=bool)
    ridges = extract_ridges_masked(result, mask, n=1)
    assert ridges[0].freq_path.shape == (N,)
    assert ridges[0].bin_path.shape == (N,)
    assert ridges[0].energy_path.shape == (N,)


def test_masked_ridge_restricts_to_unmasked_bins(cfg):
    """Mask out the top half of frequency bins — ridge must stay in bottom half."""
    result = _make_sst(20.0, cfg)  # low frequency: 20 Hz
    n_freqs = len(result.freqs)
    n_times = len(result.times)
    mask = np.zeros((n_freqs, n_times), dtype=bool)
    # Allow only the bottom half (low frequency bins — high index in scale-ordered array)
    half = n_freqs // 2
    mask[half:, :] = True  # bottom half = higher bin indices = lower frequencies
    ridges = extract_ridges_masked(result, mask, n=1)
    # All bin_path values must be in [half, n_freqs)
    assert np.all(ridges[0].bin_path >= half)


def test_masked_all_false_raises(cfg):
    result = _make_sst(40.0, cfg)
    n_freqs = len(result.freqs)
    n_times = len(result.times)
    mask = np.zeros((n_freqs, n_times), dtype=bool)
    with pytest.raises(ValueError, match="mask"):
        extract_ridges_masked(result, mask, n=1)


def test_masked_wrong_shape_raises(cfg):
    result = _make_sst(40.0, cfg)
    bad_mask = np.ones((10, 10), dtype=bool)
    with pytest.raises(ValueError):
        extract_ridges_masked(result, bad_mask, n=1)


def test_masked_energy_path_populated(cfg):
    result = _make_sst(40.0, cfg)
    n_freqs = len(result.freqs)
    n_times = len(result.times)
    mask = np.ones((n_freqs, n_times), dtype=bool)
    ridges = extract_ridges_masked(result, mask, n=1)
    r = ridges[0]
    assert r.energy_path.dtype == np.float64
    np.testing.assert_allclose(r.energy_path.sum(), r.energy, rtol=1e-10)
