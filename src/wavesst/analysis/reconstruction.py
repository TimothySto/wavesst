from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from wavesst.transforms.sst import SSTResult
from wavesst.analysis.ridge import Ridge
from wavesst._core.filters import morlet_freq_response


def _admissibility_constant(w0: float = 6.0, n_pts: int = 100_000) -> float:
    """Numerically compute C_psi = integral_0^inf |psi_hat(omega)|^2 / omega d_omega."""
    omega = np.linspace(1e-6, 30.0, n_pts)
    dw = omega[1] - omega[0]
    psi_hat = morlet_freq_response(omega, scale=1.0, w0=w0)
    return float(np.sum(psi_hat ** 2 / omega) * dw)


_C_PSI = _admissibility_constant()   # computed once at import time


@dataclass
class Component:
    signal: np.ndarray      # float64, shape (n_samples,) -- reconstructed time-domain signal
    amplitude: np.ndarray   # float64, shape (n_samples,) -- instantaneous amplitude envelope
    phase: np.ndarray       # float64, shape (n_samples,) -- instantaneous phase (radians)
    ridge: Ridge            # the ridge this component was extracted along


def reconstruct(
    sst_result: SSTResult,
    ridges: list[Ridge],
    bandwidth: float | None = None,
) -> list[Component]:
    """
    Reconstruct time-domain components from SST output.

    Inverse SST formula:
      x_k(t) ~= (2 / C_psi) * Re[ sum_{f in band_k(t)} T_x(f, t) * df ]

    We integrate T_x within a frequency band of width `bandwidth` around each
    ridge's instantaneous frequency path, then apply the inverse normalization.

    Parameters
    ----------
    sst_result : SSTResult
    ridges     : list of Ridge from extract_ridges()
    bandwidth  : Hz half-width of integration band around each ridge.
                 Defaults to df * 3 (plus/minus 3 frequency bins).

    Returns
    -------
    list of Component, same length as ridges
    """
    Tx = sst_result.Tx              # (n_freqs, n_samples) complex
    freqs = sst_result.freqs        # (n_freqs,) Hz, uniform ascending
    df = freqs[1] - freqs[0]
    n_freqs, n_samples = Tx.shape

    if bandwidth is None:
        bandwidth = df * 3.0        # +/-3 bins around ridge

    components: list[Component] = []

    for ridge in ridges:
        # Integrate Tx within band around ridge path
        analytic = np.zeros(n_samples, dtype=np.complex128)

        for t in range(n_samples):
            f_ridge = ridge.freq_path[t]
            f_lo = f_ridge - bandwidth
            f_hi = f_ridge + bandwidth
            mask = (freqs >= f_lo) & (freqs <= f_hi)
            analytic[t] = np.sum(Tx[mask, t]) * df

        # Apply inverse normalization: (2/C_psi)
        analytic *= (2.0 / _C_PSI)

        # Real part -> reconstructed signal
        signal = np.real(analytic)

        # Amplitude envelope = |analytic signal|
        amplitude = np.abs(analytic)

        # Instantaneous phase = angle of analytic signal
        phase = np.angle(analytic)

        components.append(Component(
            signal=signal,
            amplitude=amplitude,
            phase=phase,
            ridge=ridge,
        ))

    return components
