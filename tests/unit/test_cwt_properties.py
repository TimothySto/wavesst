"""
Mathematical property tests for the CWT:
- Linearity
- Frequency localization (pure tone → energy peaks at correct scale)
- Two-component separation
- nv parameter controls scale density
- Parseval-like energy conservation (approximate)
"""
import numpy as np
import pytest
from wavesst.transforms.cwt import cwt, _scales_to_freqs, MORLET_W0

FS = 256.0
N = 512


# cfg fixture provided by conftest.py


def _pure_tone(freq_hz, n=N, fs=FS):
    t = np.arange(n) / fs
    return np.cos(2 * np.pi * freq_hz * t)


# --- Linearity ---

def test_linearity_superposition(cfg):
    """CWT(a*x + b*y) == a*CWT(x) + b*CWT(y)."""
    rng = np.random.default_rng(1)
    x = rng.standard_normal(N)
    y = rng.standard_normal(N)
    a, b = 2.5, -1.3
    scales = np.geomspace(2.0 / FS, 64.0 / FS, 32)

    Wx = cwt(x, wavelet="morlet", scales=scales, fs=FS, cfg=cfg).W
    Wy = cwt(y, wavelet="morlet", scales=scales, fs=FS, cfg=cfg).W
    Wxy = cwt(a * x + b * y, wavelet="morlet", scales=scales, fs=FS, cfg=cfg).W

    np.testing.assert_allclose(Wxy.numpy(), (a * Wx + b * Wy).numpy(), atol=1e-10)


# --- Frequency localization ---

def test_energy_peak_near_correct_scale(cfg):
    """For a pure cosine at f0, the CWT magnitude should peak at the scale corresponding to f0."""
    f0 = 32.0  # Hz
    x = _pure_tone(f0)
    result = cwt(x, wavelet="morlet", scales="auto", fs=FS, cfg=cfg)

    energy = result.W.abs().pow(2).sum(dim=1).numpy()
    peak_freq = result.freqs[np.argmax(energy)]

    assert abs(peak_freq - f0) / f0 < 0.15, (
        f"Expected peak near {f0} Hz, got {peak_freq:.2f} Hz"
    )


def test_energy_peak_shifts_with_frequency(cfg):
    """Higher frequency pure tone → peak at higher frequency scale."""
    f1, f2 = 16.0, 64.0
    x1 = _pure_tone(f1)
    x2 = _pure_tone(f2)
    scales = np.geomspace(2.0 / FS, 128.0 / FS, 128)

    r1 = cwt(x1, wavelet="morlet", scales=scales, fs=FS, cfg=cfg)
    r2 = cwt(x2, wavelet="morlet", scales=scales, fs=FS, cfg=cfg)

    energy1 = r1.W.abs().pow(2).sum(dim=1).numpy()
    energy2 = r2.W.abs().pow(2).sum(dim=1).numpy()

    peak_freq1 = r1.freqs[np.argmax(energy1)]
    peak_freq2 = r2.freqs[np.argmax(energy2)]

    assert peak_freq2 > peak_freq1, (
        f"Expected peak of {f2} Hz > peak of {f1} Hz, got {peak_freq2:.2f} and {peak_freq1:.2f}"
    )


# --- Two-component separation ---

def test_two_component_peaks(cfg):
    """A two-tone signal should produce two distinct energy peaks."""
    f1, f2 = 20.0, 80.0
    t = np.arange(N) / FS
    x = np.cos(2 * np.pi * f1 * t) + np.cos(2 * np.pi * f2 * t)

    result = cwt(x, wavelet="morlet", scales="auto", fs=FS, cfg=cfg)
    energy = result.W.abs().pow(2).sum(dim=1).numpy()

    from scipy.signal import find_peaks
    peaks, _ = find_peaks(energy, height=energy.max() * 0.2)

    assert len(peaks) >= 2, (
        f"Expected at least 2 energy peaks for two-tone signal, found {len(peaks)}"
    )
    peak_freqs = sorted(result.freqs[peaks])
    assert peak_freqs[0] < f2 and peak_freqs[-1] > f1


# --- nv parameter ---

def test_nv_controls_scale_count(cfg):
    """Higher nv should produce more scales."""
    x = _pure_tone(32.0)
    r16 = cwt(x, wavelet="morlet", scales="auto", fs=FS, nv=16, cfg=cfg)
    r32 = cwt(x, wavelet="morlet", scales="auto", fs=FS, nv=32, cfg=cfg)
    assert r32.W.shape[0] > r16.W.shape[0]


def test_nv_32_has_expected_density(cfg):
    """With nv=32, scales should step by factor 2^(1/32) ≈ 1.022 between neighbors."""
    x = _pure_tone(32.0)
    result = cwt(x, wavelet="morlet", scales="auto", fs=FS, nv=32, cfg=cfg)
    ratios = result.scales[1:] / result.scales[:-1]
    expected_ratio = 2.0 ** (1.0 / 32)
    np.testing.assert_allclose(ratios, expected_ratio, rtol=1e-6)
