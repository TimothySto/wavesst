from __future__ import annotations

import math
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from wavesst.transforms.cwt import CWTResult

from wavesst.transforms.cwt import MORLET_W0


@lru_cache(maxsize=16)
def _reconstruction_constant(wavelet: str, order: int | None, n_pts: int = 200_000) -> float:
    """
    Compute the reconstruction constant K_ψ = (1/2) · ∫_0^∞ ψ̂(ω)/ω dω.

    The CWT in this library uses the convention:
        W(a,t) = IFFT[ X̂(ω) · ψ̂(aω) ]   (no √a normalisation)

    For a cosine input x(t) = cos(ω₀t):
        Re[W(a,t)] ≈ (1/2) · ψ̂(aω₀) · cos(ω₀t)

    Summing over log-uniform scales:
        Re[Σ_a W(a,t) · da/a] ≈ (1/2) · [∫_0^∞ ψ̂(u)/u du] · x(t)

    So perfect reconstruction uses:
        x(t) = (da_ratio / K_ψ) · Re[W.sum(dim=0)]
    where K_ψ = (1/2) · ∫ψ̂/ω dω — the factor 1/2 comes from the cosine
    representation, not from any analytic-signal cancellation.

    Parameters
    ----------
    wavelet : str
        Wavelet name ('morlet', 'bump', 'paul', 'dog')
    order : int | None
        Wavelet order (used for 'paul' and 'dog'; ignored for 'morlet' and 'bump')
    n_pts : int
        Number of quadrature points on [1e-6, 100] for the numerical integral (default 200000)

    Returns
    -------
    float
        The reconstruction constant K_ψ
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

    For the CWT convention used here (no √a normalisation):
        W(a,t) = IFFT[ X̂(ω) · ψ̂(aω) ]

    The reconstruction formula is:
        x(t) = (da_ratio / K_ψ) · Re[ Σ_j W(a_j, t) ]

    where:
        da_ratio = log(a[1]/a[0])  (constant for log-uniform scales)
        K_ψ = (1/2) · ∫_0^∞ ψ̂(ω)/ω dω  (wavelet-specific reconstruction constant)

    This is equivalent to the standard formula (2/C_ψ)·∫W·da/a with the
    conventional C_ψ = ∫|ψ̂|²/ω dω only for wavelets where |ψ̂| = ψ̂
    (i.e., real, positive filter banks as used here).

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
    # The reconstruction constant K_ψ already accounts for the 1/2 factor
    # from the cosine representation; see _reconstruction_constant() docstring.
    W_sum = W.sum(dim=0)                             # (N,) complex
    x_reconstructed = (da_ratio / c_psi) * torch.real(W_sum)

    return x_reconstructed.numpy().astype(np.float64)
