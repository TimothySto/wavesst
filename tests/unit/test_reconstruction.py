import numpy as np
import pytest
from wavesst.transforms.sst import sst
from wavesst.analysis.ridge import extract_ridges
from wavesst.analysis.reconstruction import reconstruct, Component
from wavesst.backends import get_backend

FS = 256.0
N = 512


@pytest.fixture
def backend():
    return get_backend("numpy")


def _make_sst_ridges(freq_hz, backend):
    t = np.arange(N) / FS
    x = np.cos(2 * np.pi * freq_hz * t).astype(np.float64)
    sst_result = sst(x, wavelet="morlet", scales="auto", fs=FS, gamma="auto", backend=backend)
    ridges = extract_ridges(sst_result, n=1, penalty=1.0)
    return x, sst_result, ridges


def test_reconstruct_returns_list(backend):
    _, sst_result, ridges = _make_sst_ridges(32.0, backend)
    components = reconstruct(sst_result, ridges)
    assert isinstance(components, list)


def test_reconstruct_returns_component_objects(backend):
    _, sst_result, ridges = _make_sst_ridges(32.0, backend)
    components = reconstruct(sst_result, ridges)
    assert len(components) == 1
    assert isinstance(components[0], Component)


def test_component_signal_shape(backend):
    _, sst_result, ridges = _make_sst_ridges(32.0, backend)
    components = reconstruct(sst_result, ridges)
    assert components[0].signal.shape == (N,)


def test_component_amplitude_shape(backend):
    _, sst_result, ridges = _make_sst_ridges(32.0, backend)
    components = reconstruct(sst_result, ridges)
    assert components[0].amplitude.shape == (N,)


def test_component_phase_shape(backend):
    _, sst_result, ridges = _make_sst_ridges(32.0, backend)
    components = reconstruct(sst_result, ridges)
    assert components[0].phase.shape == (N,)


def test_component_amplitude_nonnegative(backend):
    _, sst_result, ridges = _make_sst_ridges(32.0, backend)
    components = reconstruct(sst_result, ridges)
    assert np.all(components[0].amplitude >= 0)


def test_reconstruction_snr_pure_tone(backend):
    """Reconstructed single tone should have SNR > 10 dB vs original."""
    f0 = 32.0
    x, sst_result, ridges = _make_sst_ridges(f0, backend)
    components = reconstruct(sst_result, ridges)
    x_hat = components[0].signal

    # Align scale (reconstruction normalization may differ by a constant factor)
    # Use least-squares scaling
    scale = np.dot(x, x_hat) / np.dot(x_hat, x_hat)
    x_hat_scaled = scale * x_hat

    signal_power = np.mean(x ** 2)
    noise_power = np.mean((x - x_hat_scaled) ** 2)
    snr_db = 10 * np.log10(signal_power / (noise_power + 1e-12))
    assert snr_db > 10.0, f"Reconstruction SNR {snr_db:.1f} dB < 10 dB threshold"


def test_component_ridge_attached(backend):
    _, sst_result, ridges = _make_sst_ridges(32.0, backend)
    components = reconstruct(sst_result, ridges)
    assert components[0].ridge is ridges[0]
