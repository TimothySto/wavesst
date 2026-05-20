import numpy as np
import pytest
import torch
import wavesst
from wavesst.transforms.cwt import cwt
from wavesst.transforms.icwt import icwt


FS = 256.0
N = 512


@pytest.fixture
def tone():
    t = np.arange(N) / FS
    return np.cos(2 * np.pi * 32.0 * t).astype(np.float64)


@pytest.fixture
def cwt_morlet(tone, cfg):
    return cwt(tone, wavelet="morlet", fs=FS, nv=32, cfg=cfg)


def test_icwt_returns_array(cwt_morlet):
    out = icwt(cwt_morlet)
    assert isinstance(out, np.ndarray)


def test_icwt_shape(cwt_morlet, tone):
    out = icwt(cwt_morlet)
    assert out.shape == (N,)


def test_icwt_dtype_float(cwt_morlet):
    out = icwt(cwt_morlet)
    assert np.issubdtype(out.dtype, np.floating)


def test_icwt_morlet_reconstructs_tone(tone, cwt_morlet):
    """Morlet icwt should reconstruct a pure tone to within 1% RMS error."""
    out = icwt(cwt_morlet)
    rms_err = float(np.sqrt(np.mean((out - tone) ** 2)))
    rms_sig = float(np.sqrt(np.mean(tone ** 2)))
    assert rms_err / rms_sig < 0.01, (
        f"Reconstruction error {rms_err/rms_sig:.4f} exceeds 1%"
    )


def test_icwt_bump_reconstructs_tone(tone, cfg):
    """Bump icwt should reconstruct a pure tone reasonably (< 10% RMS)."""
    r = cwt(tone, wavelet="bump", fs=FS, nv=32, cfg=cfg)
    out = icwt(r)
    rms_err = float(np.sqrt(np.mean((out - tone) ** 2)))
    rms_sig = float(np.sqrt(np.mean(tone ** 2)))
    assert rms_err / rms_sig < 0.10, (
        f"Bump reconstruction error {rms_err/rms_sig:.4f} exceeds 10%"
    )


def test_icwt_paul_shape(tone, cfg):
    r = cwt(tone, wavelet="paul", fs=FS, nv=32, cfg=cfg)
    out = icwt(r)
    assert out.shape == (N,)


def test_icwt_dog_shape(tone, cfg):
    r = cwt(tone, wavelet="dog", fs=FS, nv=32, cfg=cfg)
    out = icwt(r)
    assert out.shape == (N,)


def test_icwt_f_low_zeros_low_scales(cwt_morlet):
    """Setting f_low removes low-frequency energy."""
    out_full = icwt(cwt_morlet)
    out_hi = icwt(cwt_morlet, f_low=60.0)  # only above 60 Hz (tone is at 32 Hz)
    rms_full = float(np.sqrt(np.mean(out_full ** 2)))
    rms_hi = float(np.sqrt(np.mean(out_hi ** 2)))
    assert rms_hi < rms_full * 0.5, (
        f"f_low=60 should remove most energy; rms_hi={rms_hi:.4f} vs rms_full={rms_full:.4f}"
    )


def test_icwt_f_high_zeros_high_scales(cwt_morlet):
    """f_high below tone frequency should produce near-zero output."""
    out = icwt(cwt_morlet, f_high=10.0)  # tone at 32 Hz; keep only below 10 Hz
    rms = float(np.sqrt(np.mean(out ** 2)))
    assert rms < 0.1, f"f_high=10 should yield ~zero for f0=32Hz tone; rms={rms:.4f}"


def test_icwt_exported_from_wavesst():
    assert hasattr(wavesst, "icwt")
