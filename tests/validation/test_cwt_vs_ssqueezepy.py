"""
Cross-validation of wavesst CWT against pywt (PyWavelets).

Note: Despite the filename, this file validates against pywt (not ssqueezepy).
Real ssqueezepy comparison is in test_sst_vs_ssqueezepy.py.

Checks:
1. Energy profile (sum over time for each scale) correlates > 0.95 with pywt
2. Peak frequency from each implementation agrees within 10%
3. White noise CWT has roughly uniform energy profile (no spurious peaks)
"""
import numpy as np
import pytest
import torch
import wavesst

pywt = pytest.importorskip("pywt", reason="pywt not installed")

from wavesst.transforms.cwt import cwt


FS = 256.0
N = 512
RNG = np.random.default_rng(42)
NV = 32


@pytest.fixture(scope="module")
def module_cfg():
    return wavesst.Config(device='cpu', dtype=torch.complex128)


def _pure_tone(freq_hz, n=N, fs=FS):
    t = np.arange(n) / fs
    return np.cos(2 * np.pi * freq_hz * t).astype(np.float64)


def _pywt_cwt(x, scales, fs=FS):
    """Run pywt CWT using cmor (complex Morlet) wavelet."""
    coeffs, freqs = pywt.cwt(x, scales, "cmor1.5-1.0", sampling_period=1.0 / fs)
    return coeffs, freqs


@pytest.fixture(scope="module")
def tone_32():
    return _pure_tone(32.0)


@pytest.fixture(scope="module")
def white_noise():
    return RNG.standard_normal(N).astype(np.float64)


# --- Energy profile correlation ---

def test_energy_profile_correlation_tone(tone_32, module_cfg):
    """Energy profile of wavesst CWT should correlate with pywt CWT > 0.95."""
    our_result = cwt(tone_32, wavelet="morlet", scales="auto", fs=FS, nv=NV, cfg=module_cfg)
    our_energy = our_result.W.abs().pow(2).sum(dim=1).numpy()

    our_scales_samples = our_result.scales * FS
    pywt_coeffs, pywt_freqs = _pywt_cwt(tone_32, our_scales_samples[::-1])
    pywt_energy = np.sum(np.abs(pywt_coeffs) ** 2, axis=1)[::-1]

    ours_norm = our_energy / our_energy.sum()
    pywts_norm = pywt_energy / pywt_energy.sum()

    corr = np.corrcoef(ours_norm, pywts_norm)[0, 1]
    assert corr > 0.95, f"Energy profile correlation {corr:.4f} < 0.95 for pure tone"


def test_energy_profile_correlation_noise(white_noise, module_cfg):
    """Energy profile correlation on white noise."""
    our_result = cwt(white_noise, wavelet="morlet", scales="auto", fs=FS, nv=NV, cfg=module_cfg)
    our_energy = our_result.W.abs().pow(2).sum(dim=1).numpy()

    our_scales_samples = our_result.scales * FS
    pywt_coeffs, pywt_freqs = _pywt_cwt(white_noise, our_scales_samples[::-1])
    pywt_energy = np.sum(np.abs(pywt_coeffs) ** 2, axis=1)[::-1]

    ours_norm = our_energy / our_energy.sum()
    pywts_norm = pywt_energy / pywt_energy.sum()

    corr = np.corrcoef(ours_norm, pywts_norm)[0, 1]
    assert corr > 0.60, f"Energy profile correlation {corr:.4f} < 0.60 for white noise"


# --- Peak frequency agreement ---

def test_peak_frequency_agreement(tone_32, module_cfg):
    """Peak frequency of tone should match pywt's CWT peak within 10%."""
    our_result = cwt(tone_32, wavelet="morlet", scales="auto", fs=FS, nv=NV, cfg=module_cfg)
    our_energy = our_result.W.abs().pow(2).sum(dim=1).numpy()
    our_peak_freq = our_result.freqs[np.argmax(our_energy)]

    our_scales_samples = our_result.scales * FS
    pywt_coeffs, pywt_freqs = _pywt_cwt(tone_32, our_scales_samples[::-1])
    pywt_energy = np.sum(np.abs(pywt_coeffs) ** 2, axis=1)[::-1]
    pywt_peak_freq = our_result.freqs[np.argmax(pywt_energy)]

    rel_diff = abs(our_peak_freq - pywt_peak_freq) / pywt_peak_freq
    assert rel_diff < 0.10, (
        f"Peak freq: ours={our_peak_freq:.2f} Hz, pywt={pywt_peak_freq:.2f} Hz, "
        f"relative diff={rel_diff:.3f}"
    )


# --- White noise uniformity ---

def test_white_noise_uniform_energy(white_noise, module_cfg):
    """CWT of white noise should have roughly flat energy profile."""
    result = cwt(white_noise, wavelet="morlet", scales="auto", fs=FS, nv=NV, cfg=module_cfg)
    energy = result.W.abs().pow(2).sum(dim=1).numpy()
    energy_norm = energy / energy.mean()

    cv = energy_norm.std() / energy_norm.mean()
    assert cv < 0.5, f"White noise CWT energy CV={cv:.3f} is too high (expected < 0.5)"
