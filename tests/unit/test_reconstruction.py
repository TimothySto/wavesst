import numpy as np
import pytest
from wavesst.transforms.sst import sst
from wavesst.analysis.ridge import extract_ridges
from wavesst.analysis.reconstruction import reconstruct, Component

FS = 256.0
N = 512


# cfg fixture provided by conftest.py


def _make_sst_ridges(freq_hz, cfg):
    t = np.arange(N) / FS
    x = np.cos(2 * np.pi * freq_hz * t).astype(np.float64)
    sst_result = sst(x, wavelet="morlet", scales="auto", fs=FS, gamma="auto", cfg=cfg)
    ridges = extract_ridges(sst_result, n=1, penalty=1.0)
    return x, sst_result, ridges


def test_reconstruct_returns_list(cfg):
    _, sst_result, ridges = _make_sst_ridges(32.0, cfg)
    components = reconstruct(sst_result, ridges)
    assert isinstance(components, list)


def test_reconstruct_returns_component_objects(cfg):
    _, sst_result, ridges = _make_sst_ridges(32.0, cfg)
    components = reconstruct(sst_result, ridges)
    assert len(components) == 1
    assert isinstance(components[0], Component)


def test_component_signal_shape(cfg):
    _, sst_result, ridges = _make_sst_ridges(32.0, cfg)
    components = reconstruct(sst_result, ridges)
    assert components[0].signal.shape == (N,)


def test_component_amplitude_shape(cfg):
    _, sst_result, ridges = _make_sst_ridges(32.0, cfg)
    components = reconstruct(sst_result, ridges)
    assert components[0].amplitude.shape == (N,)


def test_component_phase_shape(cfg):
    _, sst_result, ridges = _make_sst_ridges(32.0, cfg)
    components = reconstruct(sst_result, ridges)
    assert components[0].phase.shape == (N,)


def test_component_amplitude_nonnegative(cfg):
    _, sst_result, ridges = _make_sst_ridges(32.0, cfg)
    components = reconstruct(sst_result, ridges)
    assert np.all(components[0].amplitude >= 0)


def test_reconstruction_snr_pure_tone(cfg):
    """Reconstructed single tone should have SNR > 20 dB vs original.

    No amplitude rescaling — tests absolute fidelity.  The reconstruction
    formula (2/Css)·Re[∑T_x] is amplitude-accurate: a unit cosine should
    come back as a unit cosine within edge-effect limits.
    """
    f0 = 32.0
    x, sst_result, ridges = _make_sst_ridges(f0, cfg)
    components = reconstruct(sst_result, ridges)
    x_hat = components[0].signal

    # Use central half to avoid edge effects
    mid = slice(N // 4, 3 * N // 4)
    signal_power = np.mean(x[mid] ** 2)
    noise_power  = np.mean((x[mid] - x_hat[mid]) ** 2)
    snr_db = 10 * np.log10(signal_power / (noise_power + 1e-12))
    assert snr_db > 20.0, f"Reconstruction SNR {snr_db:.1f} dB < 20 dB threshold"


def test_reconstruction_amplitude_accuracy(cfg):
    """Reconstructed RMS amplitude should match original within 5%.

    Verifies that the (2/Css) normalization is correct — catches the
    250× amplitude bug that existed before Session 5.
    """
    f0 = 32.0
    A = 1.0
    t = np.arange(N) / FS
    x = A * np.cos(2 * np.pi * f0 * t).astype(np.float64)
    sst_result = sst(x, wavelet="morlet", scales="auto", fs=FS, gamma="auto", cfg=cfg)
    ridges = extract_ridges(sst_result, n=1, penalty=1.0)
    components = reconstruct(sst_result, ridges)
    x_hat = components[0].signal

    mid = slice(N // 4, 3 * N // 4)
    rms_expected = A / np.sqrt(2)          # RMS of A·cos(...)
    rms_got      = np.sqrt(np.mean(x_hat[mid] ** 2))
    error_factor = rms_got / rms_expected

    assert abs(error_factor - 1.0) < 0.05, (
        f"Reconstruction RMS amplitude error factor = {error_factor:.4f} "
        f"(expected 1.00 ± 0.05)"
    )


def test_component_ridge_attached(cfg):
    _, sst_result, ridges = _make_sst_ridges(32.0, cfg)
    components = reconstruct(sst_result, ridges)
    assert components[0].ridge is ridges[0]


def test_reconstruct_msst_result(cfg):
    """reconstruct() should accept MSSTResult (duck-types through _reconstruct_cwt_sst)."""
    from wavesst.transforms.msst import msst, MSSTResult

    f0 = 32.0
    t = np.arange(N) / FS
    x = np.cos(2 * np.pi * f0 * t).astype(np.float64)
    msst_result = msst(x, wavelet="morlet", scales="auto", fs=FS,
                       n_iter=2, gamma="auto", cfg=cfg)
    assert isinstance(msst_result, MSSTResult)

    ridges = extract_ridges(msst_result, n=1, penalty=1.0)
    components = reconstruct(msst_result, ridges)

    assert isinstance(components, list)
    assert len(components) == 1
    assert isinstance(components[0], Component)
    assert components[0].signal.shape == (N,)
