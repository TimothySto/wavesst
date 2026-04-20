import numpy as np
import pytest
from wavesst.transforms.cwt import cwt, CWTResult
from wavesst.backends import get_backend


@pytest.fixture
def signal_256():
    rng = np.random.default_rng(0)
    return rng.standard_normal(256).astype(np.float64)


@pytest.fixture
def backend():
    return get_backend("numpy")


# --- Shape and dtype ---

def test_cwt_returns_cwt_result(signal_256, backend):
    result = cwt(signal_256, wavelet="morlet", scales="auto", fs=1.0, backend=backend)
    assert isinstance(result, CWTResult)


def test_cwt_w_shape(signal_256, backend):
    result = cwt(signal_256, wavelet="morlet", scales="auto", fs=1.0, backend=backend)
    n_scales, n_samples = result.W.shape
    assert n_samples == len(signal_256)
    assert n_scales > 0


def test_cwt_w_dtype_complex(signal_256, backend):
    result = cwt(signal_256, wavelet="morlet", scales="auto", fs=1.0, backend=backend)
    assert np.iscomplexobj(result.W)


def test_cwt_scales_length(signal_256, backend):
    result = cwt(signal_256, wavelet="morlet", scales="auto", fs=1.0, backend=backend)
    assert len(result.scales) == result.W.shape[0]


def test_cwt_freqs_length(signal_256, backend):
    result = cwt(signal_256, wavelet="morlet", scales="auto", fs=1.0, backend=backend)
    assert len(result.freqs) == result.W.shape[0]


def test_cwt_times_length(signal_256, backend):
    result = cwt(signal_256, wavelet="morlet", scales="auto", fs=1.0, backend=backend)
    assert len(result.times) == len(signal_256)


def test_cwt_scales_positive(signal_256, backend):
    result = cwt(signal_256, wavelet="morlet", scales="auto", fs=1.0, backend=backend)
    assert np.all(result.scales > 0)


def test_cwt_freqs_positive(signal_256, backend):
    result = cwt(signal_256, wavelet="morlet", scales="auto", fs=1.0, backend=backend)
    assert np.all(result.freqs > 0)


def test_cwt_scales_log_spaced(signal_256, backend):
    """Auto scales should be geometrically (log) spaced."""
    result = cwt(signal_256, wavelet="morlet", scales="auto", fs=1.0, backend=backend)
    log_scales = np.log2(result.scales)
    diffs = np.diff(log_scales)
    np.testing.assert_allclose(diffs, diffs[0], rtol=1e-6)


def test_cwt_freqs_inversely_proportional_to_scales(signal_256, backend):
    """freqs = w0 / (2*pi * scale) * fs — so freqs * scales should be constant."""
    result = cwt(signal_256, wavelet="morlet", scales="auto", fs=1.0, backend=backend)
    products = result.freqs * result.scales
    np.testing.assert_allclose(products, products[0], rtol=1e-6)


def test_cwt_explicit_scale_count(signal_256, backend):
    result = cwt(signal_256, wavelet="morlet", scales=64, fs=1.0, backend=backend)
    assert result.W.shape[0] == 64


def test_cwt_explicit_scale_array(signal_256, backend):
    scales = np.array([1.0, 2.0, 4.0, 8.0])
    result = cwt(signal_256, wavelet="morlet", scales=scales, fs=1.0, backend=backend)
    assert result.W.shape[0] == 4
    np.testing.assert_array_equal(result.scales, scales)


def test_cwt_fs_scales_freqs(signal_256, backend):
    """With fixed scale array, doubling fs doubles all freqs."""
    fixed_scales = np.array([1.0, 2.0, 4.0, 8.0])
    r1 = cwt(signal_256, wavelet="morlet", scales=fixed_scales, fs=1.0, backend=backend)
    r2 = cwt(signal_256, wavelet="morlet", scales=fixed_scales, fs=2.0, backend=backend)
    np.testing.assert_allclose(r2.freqs, r1.freqs * 2.0, rtol=1e-6)
