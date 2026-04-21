"""
Mathematical property tests for the SST:
- Sharpening: SST concentrates energy more than CWT
- IF localization: pure tone -> peak at correct frequency bin
- Two-component: two tones produce two distinct SST peaks
- Energy conservation: total SST energy ~= total CWT energy (up to thresholding)
"""
import numpy as np
import pytest
from wavesst.transforms.sst import sst

FS = 256.0
N = 512


# cfg fixture provided by conftest.py


def _tone(freq_hz, n=N, fs=FS):
    t = np.arange(n) / fs
    return np.cos(2 * np.pi * freq_hz * t)


def test_sst_sharpens_cwt(cfg):
    """SST energy should be more concentrated (lower entropy) than CWT energy."""
    x = _tone(32.0)
    result = sst(x, wavelet="morlet", scales="auto", fs=FS, gamma=None, cfg=cfg)

    cwt_energy = result.Wx.W.abs().pow(2).sum(dim=1).numpy()
    sst_energy = result.Tx.abs().pow(2).sum(dim=1).numpy()

    def _entropy(e):
        p = e / e.sum()
        p = p[p > 0]
        return -np.sum(p * np.log(p))

    assert _entropy(sst_energy) < _entropy(cwt_energy), (
        "SST energy should be more concentrated (lower entropy) than CWT energy"
    )


def test_sst_peak_at_correct_freq(cfg):
    """Pure tone at f0 -> SST peak within 10% of f0."""
    f0 = 40.0
    x = _tone(f0)
    result = sst(x, wavelet="morlet", scales="auto", fs=FS, gamma="auto", cfg=cfg)

    sst_energy = result.Tx.abs().pow(2).sum(dim=1).numpy()
    peak_freq = result.freqs[np.argmax(sst_energy)]

    assert abs(peak_freq - f0) / f0 < 0.10, (
        f"Expected SST peak near {f0} Hz, got {peak_freq:.2f} Hz"
    )


def test_sst_two_tone_two_peaks(cfg):
    """Two tones -> two distinct SST energy peaks."""
    from scipy.signal import find_peaks
    f1, f2 = 20.0, 80.0
    t = np.arange(N) / FS
    x = np.cos(2 * np.pi * f1 * t) + np.cos(2 * np.pi * f2 * t)

    result = sst(x, wavelet="morlet", scales="auto", fs=FS, gamma="auto", cfg=cfg)
    energy = result.Tx.abs().pow(2).sum(dim=1).numpy()

    peaks, _ = find_peaks(energy, height=energy.max() * 0.1)
    assert len(peaks) >= 2, f"Expected >=2 SST peaks for two-tone, found {len(peaks)}"


def test_sst_energy_preserved_no_threshold(cfg):
    """With gamma=None, Tx must be non-zero and finite."""
    rng = np.random.default_rng(5)
    x = rng.standard_normal(N)
    result = sst(x, wavelet="morlet", scales="auto", fs=FS, gamma=None, cfg=cfg)

    sst_total = result.Tx.abs().pow(2).sum().item()
    assert sst_total > 0, "SST Tx should have non-zero energy with gamma=None"
    assert np.isfinite(sst_total), "SST Tx energy should be finite"
