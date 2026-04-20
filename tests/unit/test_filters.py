import numpy as np
import pytest


def _import_morlet():
    from wavesst._core.filters import morlet_freq_response
    return morlet_freq_response


def test_morlet_import():
    morlet_freq_response = _import_morlet()
    assert callable(morlet_freq_response)


def test_morlet_shape():
    morlet_freq_response = _import_morlet()
    omega = np.linspace(0, 20.0, 512)
    result = morlet_freq_response(omega, scale=1.0)
    assert result.shape == (512,)
    assert result.dtype == np.float64


def test_morlet_nonnegative():
    morlet_freq_response = _import_morlet()
    omega = np.linspace(-10.0, 20.0, 512)
    result = morlet_freq_response(omega, scale=1.0)
    assert np.all(result >= 0.0)


def test_morlet_zero_at_negative_freq():
    """Analytic wavelet: response must be exactly 0 for omega <= 0."""
    morlet_freq_response = _import_morlet()
    omega = np.linspace(-5.0, 0.0, 64)
    result = morlet_freq_response(omega, scale=1.0)
    np.testing.assert_array_equal(result, 0.0)


def test_morlet_peak_at_w0_over_scale():
    """Peak of ψ̂(aω) should occur at ω = w0/scale."""
    morlet_freq_response = _import_morlet()
    w0 = 6.0
    scale = 2.0
    omega = np.linspace(0.0, 20.0, 10000)
    result = morlet_freq_response(omega, scale=scale, w0=w0)
    peak_omega = omega[np.argmax(result)]
    expected = w0 / scale
    assert abs(peak_omega - expected) < 0.01


def test_morlet_scale_shifts_peak():
    """Doubling the scale halves the peak frequency."""
    morlet_freq_response = _import_morlet()
    omega = np.linspace(0.0, 20.0, 10000)
    r1 = morlet_freq_response(omega, scale=1.0)
    r2 = morlet_freq_response(omega, scale=2.0)
    peak1 = omega[np.argmax(r1)]
    peak2 = omega[np.argmax(r2)]
    assert abs(peak1 / peak2 - 2.0) < 0.05


def test_morlet_admissibility_constant():
    """C_psi = integral of |psi_hat(omega)|^2 / omega dω should be finite and positive."""
    morlet_freq_response = _import_morlet()
    omega = np.linspace(1e-6, 30.0, 100000)
    dw = omega[1] - omega[0]
    psi_hat = morlet_freq_response(omega, scale=1.0)
    c_psi = np.sum(psi_hat**2 / omega) * dw
    assert c_psi > 0.0
    assert np.isfinite(c_psi)


def test_morlet_w0_parameter():
    """Different w0 values should shift the peak accordingly."""
    morlet_freq_response = _import_morlet()
    omega = np.linspace(0.0, 20.0, 10000)
    for w0 in [4.0, 6.0, 8.0]:
        result = morlet_freq_response(omega, scale=1.0, w0=w0)
        peak_omega = omega[np.argmax(result)]
        assert abs(peak_omega - w0) < 0.1, f"Expected peak at {w0}, got {peak_omega}"
