import numpy as np
import pytest
import wavesst
from wavesst.analysis.parallel import extract_ridges_parallel
from wavesst.analysis.ridge import Ridge
from wavesst.transforms.sst import sst

FS = 256.0
N = 512


def _make_sst(f0: float, cfg):
    t = np.arange(N) / FS
    x = np.cos(2 * np.pi * f0 * t).astype(np.float32)
    return sst(x, fs=FS, nv=32, gamma='auto', cfg=cfg)


def test_parallel_returns_ridge_list(cfg):
    result = _make_sst(40.0, cfg)
    ridges = extract_ridges_parallel(result, n=2, penalty=1.0)
    assert isinstance(ridges, list)
    assert len(ridges) == 2


def test_parallel_returns_ridge_objects(cfg):
    result = _make_sst(40.0, cfg)
    ridges = extract_ridges_parallel(result, n=2, penalty=1.0)
    for r in ridges:
        assert isinstance(r, Ridge)


def test_parallel_ridge_shapes(cfg):
    result = _make_sst(40.0, cfg)
    ridges = extract_ridges_parallel(result, n=2, penalty=1.0)
    for r in ridges:
        assert r.freq_path.shape == (N,)
        assert r.bin_path.shape == (N,)
        assert r.energy_path.shape == (N,)


def test_parallel_sorted_ascending_frequency(cfg):
    """Ridges from parallel extraction are ordered by ascending median frequency."""
    result = _make_sst(40.0, cfg)
    ridges = extract_ridges_parallel(result, n=3, penalty=1.0)
    medians = [float(np.median(r.freq_path)) for r in ridges]
    # Verify sorted ascending and genuinely distinct (3 different bands)
    assert medians == sorted(medians), f"Expected sorted medians, got {medians}"
    # Verify bands are distinct (each ridge covers a different frequency region)
    assert len(set(round(m) for m in medians)) == 3, \
        f"Expected 3 distinct frequency regions, got {medians}"


def test_parallel_n_ge_n_freqs_raises(cfg):
    result = _make_sst(40.0, cfg)
    n_freqs = len(result.freqs)
    with pytest.raises(ValueError, match="frequency bins"):
        extract_ridges_parallel(result, n=n_freqs, penalty=1.0)


def test_parallel_n1_returns_one_ridge(cfg):
    result = _make_sst(40.0, cfg)
    ridges = extract_ridges_parallel(result, n=1, penalty=1.0)
    assert len(ridges) == 1


def test_parallel_energy_path_populated(cfg):
    result = _make_sst(40.0, cfg)
    ridges = extract_ridges_parallel(result, n=2, penalty=1.0)
    for r in ridges:
        assert r.energy_path.dtype == np.float64
        np.testing.assert_allclose(r.energy_path.sum(), r.energy, rtol=1e-10)


def test_parallel_exported_from_wavesst():
    assert hasattr(wavesst, 'extract_ridges_parallel')
