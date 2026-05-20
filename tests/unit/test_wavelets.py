import math
import numpy as np
import pytest
import torch
import wavesst
from wavesst.transforms.cwt import cwt, CWTResult


FS = 256.0
N = 512


@pytest.fixture
def tone():
    t = np.arange(N) / FS
    return np.cos(2 * np.pi * 32.0 * t).astype(np.float64)


# --- Bump ---

def test_cwt_bump_returns_result(tone, cfg):
    result = cwt(tone, wavelet="bump", fs=FS, cfg=cfg)
    assert isinstance(result, CWTResult)


def test_cwt_bump_shape(tone, cfg):
    result = cwt(tone, wavelet="bump", fs=FS, cfg=cfg)
    assert result.W.shape[1] == N
    assert result.W.shape[0] > 0


def test_cwt_bump_is_complex(tone, cfg):
    result = cwt(tone, wavelet="bump", fs=FS, cfg=cfg)
    assert result.W.is_complex()


def test_cwt_bump_wavelet_field(tone, cfg):
    result = cwt(tone, wavelet="bump", fs=FS, cfg=cfg)
    assert result.wavelet == "bump"


def test_cwt_bump_energy_peaks_near_tone(tone, cfg):
    """Bump CWT marginal energy should peak at f0=32 Hz."""
    f0 = 32.0
    result = cwt(tone, wavelet="bump", fs=FS, cfg=cfg)
    energy = result.W.abs().pow(2).sum(dim=1).numpy()
    peak_idx = int(np.argmax(energy))
    peak_freq = result.freqs[peak_idx]
    assert abs(peak_freq - f0) / f0 < 0.15, (
        f"Bump energy peak at {peak_freq:.1f} Hz, expected near {f0} Hz"
    )


# --- Paul ---

def test_cwt_paul_default_order(tone, cfg):
    result = cwt(tone, wavelet="paul", fs=FS, cfg=cfg)
    assert result.wavelet == "paul"
    assert result.wavelet_order == 4


def test_cwt_paul_custom_order(tone, cfg):
    result = cwt(tone, wavelet="paul", wavelet_order=6, fs=FS, cfg=cfg)
    assert result.wavelet_order == 6


def test_cwt_paul_shape(tone, cfg):
    result = cwt(tone, wavelet="paul", fs=FS, cfg=cfg)
    assert result.W.shape[1] == N


def test_cwt_paul_energy_peaks_near_tone(tone, cfg):
    f0 = 32.0
    result = cwt(tone, wavelet="paul", fs=FS, cfg=cfg)
    energy = result.W.abs().pow(2).sum(dim=1).numpy()
    peak_idx = int(np.argmax(energy))
    peak_freq = result.freqs[peak_idx]
    assert abs(peak_freq - f0) / f0 < 0.20, (
        f"Paul energy peak at {peak_freq:.1f} Hz, expected near {f0} Hz"
    )


# --- DOG ---

def test_cwt_dog_default_order(tone, cfg):
    result = cwt(tone, wavelet="dog", fs=FS, cfg=cfg)
    assert result.wavelet == "dog"
    assert result.wavelet_order == 2


def test_cwt_dog_shape(tone, cfg):
    result = cwt(tone, wavelet="dog", fs=FS, cfg=cfg)
    assert result.W.shape[1] == N


def test_cwt_dog_energy_peaks_near_tone(tone, cfg):
    f0 = 32.0
    result = cwt(tone, wavelet="dog", fs=FS, cfg=cfg)
    energy = result.W.abs().pow(2).sum(dim=1).numpy()
    peak_idx = int(np.argmax(energy))
    peak_freq = result.freqs[peak_idx]
    assert abs(peak_freq - f0) / f0 < 0.25, (
        f"DOG energy peak at {peak_freq:.1f} Hz, expected near {f0} Hz"
    )


# --- SST also accepts new wavelets ---

def test_sst_bump(tone, cfg):
    from wavesst.transforms.sst import sst
    result = sst(tone, wavelet="bump", fs=FS, gamma="auto", cfg=cfg)
    assert result.Tx.shape[1] == N


def test_sst_paul(tone, cfg):
    from wavesst.transforms.sst import sst
    result = sst(tone, wavelet="paul", fs=FS, gamma="auto", cfg=cfg)
    assert result.Tx.shape[1] == N


def test_sst_dog(tone, cfg):
    from wavesst.transforms.sst import sst
    result = sst(tone, wavelet="dog", fs=FS, gamma="auto", cfg=cfg)
    assert result.Tx.shape[1] == N


def test_cwt_unsupported_wavelet_raises(tone, cfg):
    with pytest.raises(ValueError, match="Unsupported wavelet"):
        cwt(tone, wavelet="haar", fs=FS, cfg=cfg)


def test_cwt_paul_wavelet_order_zero_raises(tone, cfg):
    """wavelet_order=0 for Paul should raise ValueError, not ZeroDivisionError."""
    with pytest.raises(ValueError, match="wavelet_order"):
        cwt(tone, wavelet="paul", wavelet_order=0, fs=FS, cfg=cfg)


def test_cwt_dog_wavelet_order_zero_raises(tone, cfg):
    """wavelet_order=0 for DOG should raise ValueError, not ZeroDivisionError."""
    with pytest.raises(ValueError, match="wavelet_order"):
        cwt(tone, wavelet="dog", wavelet_order=0, fs=FS, cfg=cfg)


def test_cwt_f_high_too_low_raises_with_diagnostic(tone, cfg):
    """f_high below all scales raises ValueError with range info."""
    with pytest.raises(ValueError, match="No scales remain"):
        cwt(tone, wavelet="morlet", fs=FS, f_high=0.001, cfg=cfg)


def test_cwt_paul_f_low_f_high(tone, cfg):
    """f_low/f_high filtering works for Paul wavelet too."""
    result = cwt(tone, wavelet="paul", fs=FS, f_low=20.0, f_high=80.0, cfg=cfg)
    assert result.freqs.min() >= 20.0 - 1e-6
    assert result.freqs.max() <= 80.0 + 1e-6
