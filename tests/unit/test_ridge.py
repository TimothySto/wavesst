import numpy as np
import pytest
from wavesst.transforms.sst import sst, SSTResult
from wavesst.analysis.ridge import extract_ridges, Ridge
from wavesst.backends import get_backend

FS = 256.0
N = 512


@pytest.fixture
def backend():
    return get_backend("numpy")


def _tone_sst(freq_hz, backend):
    t = np.arange(N) / FS
    x = np.cos(2 * np.pi * freq_hz * t).astype(np.float64)
    return sst(x, wavelet="morlet", scales="auto", fs=FS, gamma="auto", backend=backend)


def test_extract_ridges_returns_list(backend):
    result = _tone_sst(32.0, backend)
    ridges = extract_ridges(result, n=1, penalty=1.0)
    assert isinstance(ridges, list)


def test_extract_ridges_returns_ridge_objects(backend):
    result = _tone_sst(32.0, backend)
    ridges = extract_ridges(result, n=1, penalty=1.0)
    assert len(ridges) == 1
    assert isinstance(ridges[0], Ridge)


def test_ridge_has_correct_shape(backend):
    result = _tone_sst(32.0, backend)
    ridges = extract_ridges(result, n=1, penalty=1.0)
    r = ridges[0]
    assert r.freq_path.shape == (N,)
    assert r.bin_path.shape == (N,)


def test_ridge_freq_path_in_range(backend):
    result = _tone_sst(32.0, backend)
    ridges = extract_ridges(result, n=1, penalty=1.0)
    r = ridges[0]
    f_min, f_max = result.freqs.min(), result.freqs.max()
    assert np.all(r.freq_path >= f_min)
    assert np.all(r.freq_path <= f_max)


def test_ridge_tracks_correct_freq(backend):
    """Ridge should track near the true tone frequency."""
    f0 = 32.0
    result = _tone_sst(f0, backend)
    ridges = extract_ridges(result, n=1, penalty=1.0)
    median_freq = float(np.median(ridges[0].freq_path))
    assert abs(median_freq - f0) / f0 < 0.10, (
        f"Ridge median freq {median_freq:.2f} Hz should be near {f0} Hz"
    )


def test_extract_two_ridges(backend):
    """Two-tone signal should yield two ridges tracking the right frequencies."""
    f1, f2 = 20.0, 80.0
    t = np.arange(N) / FS
    x = (np.cos(2 * np.pi * f1 * t) + np.cos(2 * np.pi * f2 * t)).astype(np.float64)
    result = sst(x, wavelet="morlet", scales="auto", fs=FS, gamma="auto", backend=backend)
    ridges = extract_ridges(result, n=2, penalty=1.0)
    assert len(ridges) == 2
    median_freqs = sorted([float(np.median(r.freq_path)) for r in ridges])
    assert median_freqs[0] < 50.0 and median_freqs[1] > 50.0


def test_high_penalty_smoother_ridge(backend):
    """Higher penalty -> ridge frequency path should have smaller total variation."""
    result = _tone_sst(32.0, backend)
    r_low = extract_ridges(result, n=1, penalty=0.01)[0]
    r_high = extract_ridges(result, n=1, penalty=100.0)[0]
    tv_low = float(np.sum(np.abs(np.diff(r_low.bin_path.astype(float)))))
    tv_high = float(np.sum(np.abs(np.diff(r_high.bin_path.astype(float)))))
    assert tv_high <= tv_low + 1, (
        f"Higher penalty should give smoother ridge: tv_low={tv_low}, tv_high={tv_high}"
    )
