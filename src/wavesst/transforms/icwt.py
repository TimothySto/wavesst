from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from wavesst.transforms.cwt import CWTResult

from wavesst.transforms.cwt import MORLET_W0


def _reconstruction_constant(wavelet: str, order: int | None, n_pts: int = 200_000) -> float:
    """
    Compute the reconstruction constant K_psi for the L1-normalised CWT.

    This CWT uses W(a,b) = IFFT[X_hat(omega) * psi_hat(a*omega)] (no sqrt(a) factor).
    For a cosine input at frequency omega_0 the CWT satisfies:
        Re[W(a,t)] = (1/2) * psi_hat(a*omega_0) * cos(omega_0*t)
    and the discrete inversion sum satisfies:
        (da_ratio) * sum_j Re[W(a_j,t)] -> (1/2) * cos(omega_0*t) * int_0^inf psi_hat(u)/u du

    Perfect reconstruction therefore requires:
        K_psi = (1/2) * integral_0^inf psi_hat(omega) / omega domega

    This differs from the classical admissibility constant C_psi = int |psi_hat|^2/omega domega;
    the two coincide only for wavelets where psi_hat is its own |.|^2 (up to scaling).

    n_pts: number of quadrature points on [1e-6, 100] for the numerical integral.
    """
    omega = np.linspace(1e-6, 100.0, n_pts)
    dw = omega[1] - omega[0]

    if wavelet == "morlet":
        w0 = MORLET_W0
        # One-sided Morlet: psi_hat(omega) = pi^(-1/4) exp(-1/2*(omega-w0)^2)
        psi = np.pi ** (-0.25) * np.exp(-0.5 * (omega - w0) ** 2)
    elif wavelet == "bump":
        w0 = MORLET_W0
        u = omega / w0 - 1.0
        support = np.abs(u) < 1.0
        denom = np.where(support, u ** 2 - 1.0, -1.0)
        psi = np.where(support, np.exp(1.0 / denom), 0.0)
    elif wavelet == "paul":
        m = order if order is not None else 4
        norm = (2 ** m) / math.sqrt(m * math.factorial(2 * m))
        psi = norm * omega ** m * np.exp(-omega)
    elif wavelet == "dog":
        m = order if order is not None else 2
        norm = 1.0 / math.sqrt(math.gamma(m + 0.5))
        psi = norm * omega ** m * np.exp(-0.5 * omega ** 2)
    else:
        raise ValueError(f"Unknown wavelet '{wavelet}' for reconstruction constant")

    # K_psi = (1/2) * int_0^inf psi_hat(omega) / omega domega
    integrand = psi / omega
    return 0.5 * float(np.sum(integrand) * dw)


def icwt(
    result: "CWTResult",
    f_low: float | None = None,
    f_high: float | None = None,
) -> np.ndarray:
    """
    Inverse CWT via the admissibility formula (log-scale version).

    For geometrically-spaced scales a_j = a_0 * 2^(j/nv):
        da_j / a_j = log(a[1]/a[0])  (constant)

    The inversion formula simplifies to:
        x(t) = (da_ratio / C_psi) * Re[ sum_j W(a_j, t) ]

    (Factor is 1/C_psi not 2/C_psi because the wavelet is analytic — the
    admissibility integral is one-sided, cancelling the usual factor of 2.)

    Optional f_low / f_high zero out scales whose centre frequency falls
    outside [f_low, f_high] before summing — enables bandpass denoising.

    Parameters
    ----------
    result  : CWTResult from cwt()
    f_low   : lower frequency bound in Hz (None = no limit)
    f_high  : upper frequency bound in Hz (None = no limit)

    Returns
    -------
    np.ndarray, shape (N,), float64 — reconstructed time-domain signal
    """
    W = result.W                    # (n_scales, N) complex, CPU
    scales = result.scales          # (n_scales,) float64
    freqs = result.freqs            # (n_scales,) Hz

    if len(scales) < 2:
        raise ValueError("icwt requires at least 2 scales (cannot compute da_ratio).")

    # Apply frequency band mask
    if f_low is not None or f_high is not None:
        W = W.clone()
        if f_low is not None:
            W[freqs < f_low, :] = 0.0
        if f_high is not None:
            W[freqs > f_high, :] = 0.0

    # da / a (constant for log-uniform scales)
    da_ratio = float(np.log(scales[1] / scales[0]))

    # Reconstruction constant K_psi = (1/2) int psi_hat(omega)/omega domega
    c_psi = _reconstruction_constant(result.wavelet, result.wavelet_order)

    # Sum over scales and take real part.
    # Factor is 1/C_psi (not 2/C_psi) because this CWT is analytic (one-sided
    # spectrum): the admissibility integral is already one-sided, matching the
    # filter bank which has psi_hat(omega)=0 for omega<=0.
    W_sum = W.sum(dim=0)                             # (N,) complex
    x_reconstructed = (da_ratio / c_psi) * torch.real(W_sum)

    return x_reconstructed.numpy().astype(np.float64)
