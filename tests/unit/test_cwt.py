import numpy as np
import pytest
import torch
import wavesst
from wavesst.transforms.cwt import cwt, CWTResult


@pytest.fixture
def signal_256():
    rng = np.random.default_rng(0)
    return rng.standard_normal(256).astype(np.float64)


# cfg fixture is provided by conftest.py (device='cpu', dtype=torch.complex128)

# --- Shape and dtype ---

def test_cwt_returns_cwt_result(signal_256, cfg):
    result = cwt(signal_256, wavelet="morlet", scales="auto", fs=1.0, cfg=cfg)
    assert isinstance(result, CWTResult)


def test_cwt_w_is_tensor(signal_256, cfg):
    result = cwt(signal_256, wavelet="morlet", scales="auto", fs=1.0, cfg=cfg)
    assert isinstance(result.W, torch.Tensor)


def test_cwt_w_shape(signal_256, cfg):
    result = cwt(signal_256, wavelet="morlet", scales="auto", fs=1.0, cfg=cfg)
    n_scales, n_samples = result.W.shape
    assert n_samples == len(signal_256)
    assert n_scales > 0


def test_cwt_w_dtype_complex(signal_256, cfg):
    result = cwt(signal_256, wavelet="morlet", scales="auto", fs=1.0, cfg=cfg)
    assert result.W.is_complex()


def test_cwt_scales_length(signal_256, cfg):
    result = cwt(signal_256, wavelet="morlet", scales="auto", fs=1.0, cfg=cfg)
    assert len(result.scales) == result.W.shape[0]


def test_cwt_freqs_length(signal_256, cfg):
    result = cwt(signal_256, wavelet="morlet", scales="auto", fs=1.0, cfg=cfg)
    assert len(result.freqs) == result.W.shape[0]


def test_cwt_times_length(signal_256, cfg):
    result = cwt(signal_256, wavelet="morlet", scales="auto", fs=1.0, cfg=cfg)
    assert len(result.times) == len(signal_256)


def test_cwt_scales_positive(signal_256, cfg):
    result = cwt(signal_256, wavelet="morlet", scales="auto", fs=1.0, cfg=cfg)
    assert np.all(result.scales > 0)


def test_cwt_freqs_positive(signal_256, cfg):
    result = cwt(signal_256, wavelet="morlet", scales="auto", fs=1.0, cfg=cfg)
    assert np.all(result.freqs > 0)


def test_cwt_scales_log_spaced(signal_256, cfg):
    """Auto scales should be geometrically (log) spaced."""
    result = cwt(signal_256, wavelet="morlet", scales="auto", fs=1.0, cfg=cfg)
    log_scales = np.log2(result.scales)
    diffs = np.diff(log_scales)
    np.testing.assert_allclose(diffs, diffs[0], rtol=1e-6)


def test_cwt_freqs_inversely_proportional_to_scales(signal_256, cfg):
    """freqs = w0 / (2*pi * scale) — so freqs * scales should be constant."""
    result = cwt(signal_256, wavelet="morlet", scales="auto", fs=1.0, cfg=cfg)
    products = result.freqs * result.scales
    np.testing.assert_allclose(products, products[0], rtol=1e-6)


def test_cwt_explicit_scale_count(signal_256, cfg):
    result = cwt(signal_256, wavelet="morlet", scales=64, fs=1.0, cfg=cfg)
    assert result.W.shape[0] == 64


def test_cwt_explicit_scale_array(signal_256, cfg):
    scales = np.array([1.0, 2.0, 4.0, 8.0])
    result = cwt(signal_256, wavelet="morlet", scales=scales, fs=1.0, cfg=cfg)
    assert result.W.shape[0] == 4
    np.testing.assert_array_equal(result.scales, scales)


def test_cwt_auto_highest_freq_near_nyquist(cfg):
    """With auto scales, the highest resolved freq should be near fs/2."""
    fs = 256.0
    x = np.random.default_rng(7).standard_normal(256)
    result = cwt(x, wavelet="morlet", scales="auto", fs=fs, cfg=cfg)
    nyquist = fs / 2.0
    highest_freq = result.freqs.max()
    assert abs(highest_freq - nyquist) / nyquist < 0.1, (
        f"Highest freq {highest_freq:.1f} should be near Nyquist {nyquist:.1f}"
    )


# --- f_low / f_high band-limiting ---

def test_cwt_f_high_limits_max_freq(cfg):
    """f_high should cap the maximum centre frequency of returned scales."""
    fs = 256.0
    x = np.random.default_rng(3).standard_normal(512)
    f_cut = 60.0
    result = cwt(x, wavelet="morlet", fs=fs, f_high=f_cut, cfg=cfg)
    assert result.freqs.max() <= f_cut + 1e-6, (
        f"max freq {result.freqs.max():.2f} exceeds f_high={f_cut}"
    )


def test_cwt_f_low_limits_min_freq(cfg):
    """f_low should raise the minimum centre frequency of returned scales."""
    fs = 256.0
    x = np.random.default_rng(5).standard_normal(512)
    f_cut = 20.0
    result = cwt(x, wavelet="morlet", fs=fs, f_low=f_cut, cfg=cfg)
    assert result.freqs.min() >= f_cut - 1e-6, (
        f"min freq {result.freqs.min():.2f} below f_low={f_cut}"
    )


def test_cwt_f_low_f_high_combined(cfg):
    """f_low and f_high together restrict to a subband."""
    fs = 256.0
    x = np.random.default_rng(7).standard_normal(512)
    result = cwt(x, wavelet="morlet", fs=fs, f_low=20.0, f_high=60.0, cfg=cfg)
    assert result.freqs.min() >= 20.0 - 1e-6
    assert result.freqs.max() <= 60.0 + 1e-6


def test_cwt_f_band_reduces_scale_count(cfg):
    """Band-limiting should produce fewer scales than unrestricted."""
    fs = 256.0
    x = np.random.default_rng(9).standard_normal(512)
    r_full = cwt(x, wavelet="morlet", fs=fs, cfg=cfg)
    r_band = cwt(x, wavelet="morlet", fs=fs, f_low=20.0, f_high=80.0, cfg=cfg)
    assert r_band.W.shape[0] < r_full.W.shape[0]


def test_cwt_f_high_too_low_raises(cfg):
    """f_high below all scales should raise ValueError."""
    fs = 256.0
    x = np.random.default_rng(11).standard_normal(512)
    with pytest.raises(ValueError, match="No scales remain"):
        cwt(x, wavelet="morlet", fs=fs, f_high=0.001, cfg=cfg)


def test_sst_f_low_f_high_passthrough(cfg):
    """sst() should pass f_low/f_high to cwt() and return restricted freqs."""
    from wavesst.transforms.sst import sst
    fs = 256.0
    x = np.random.default_rng(13).standard_normal(512)
    result = sst(x, wavelet="morlet", fs=fs, f_low=20.0, f_high=80.0, cfg=cfg)
    assert result.freqs.min() >= 20.0 - 1e-6
    assert result.freqs.max() <= 80.0 + 1e-6
