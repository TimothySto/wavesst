from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from wavesst.backends._protocol import ArrayBackend
from wavesst.backends import get_backend
from wavesst.transforms.cwt import cwt, CWTResult, MORLET_W0
from wavesst._core.filters import morlet_freq_response, deriv_morlet_freq_response


@dataclass
class SSTResult:
    Tx: np.ndarray      # complex, shape (n_freqs, n_samples) — synchrosqueezed TF plane
    Wx: CWTResult       # underlying CWT (needed for reconstruction)
    freqs: np.ndarray   # float64, shape (n_freqs,) — uniform frequency grid, ascending
    times: np.ndarray   # float64, shape (n_samples,)


def _compute_gamma(gamma, Wx_abs: np.ndarray, N: int) -> float:
    """Resolve gamma to a float threshold value."""
    if gamma is None:
        return 0.0
    if callable(gamma):
        return float(gamma(Wx_abs))
    if isinstance(gamma, (int, float)):
        return float(gamma)
    if gamma == "auto":
        # MAD estimator on finest-scale (highest-freq) coefficients
        finest = Wx_abs[-1]
        sigma_hat = float(np.median(finest)) / 0.6745
        return sigma_hat
    if gamma == "universal":
        # Donoho-Johnstone universal threshold
        finest = Wx_abs[-1]
        sigma_hat = float(np.median(finest)) / 0.6745
        return sigma_hat * math.sqrt(2.0 * math.log(N))
    raise ValueError(
        f"gamma must be 'auto', 'universal', float, callable, or None; got {gamma!r}"
    )


def sst(
    x: np.ndarray,
    *,
    wavelet: str = "morlet",
    scales: str | int | np.ndarray = "auto",
    fs: float = 1.0,
    nv: int = 32,
    gamma: str | float | None = "auto",
    backend: ArrayBackend | None = None,
) -> SSTResult:
    """
    CWT-based Synchrosqueezing Transform.

    Computes the IF estimator analytically via a derivative wavelet filter
    (not numerical differentiation), then reassigns CWT energy onto a
    uniform frequency grid.

    IF estimator:   omega_hat(a,b) = -i * (d_b W_x(a,b)) / W_x(a,b)
    Reassignment:   T_x(f, b) += W_x(a,b) * (1/a) for each a where
                    |W_x(a,b)| > gamma and round(f_hat) == f bin

    Parameters
    ----------
    x       : 1-D float64 signal, length N
    wavelet : "morlet" (only supported in v1)
    scales  : "auto" | int | ndarray
    fs      : sample rate in Hz
    nv      : voices per octave (used with scales="auto")
    gamma   : threshold — "auto" | "universal" | float | callable | None
    backend : ArrayBackend; defaults to get_backend()

    Returns
    -------
    SSTResult with fields Tx, Wx, freqs, times
    """
    if wavelet != "morlet":
        raise ValueError(f"Unsupported wavelet '{wavelet}'")
    if backend is None:
        backend = get_backend()

    x = np.asarray(x, dtype=np.float64)
    N = len(x)

    # --- Step 1: CWT ---
    wx = cwt(x, wavelet=wavelet, scales=scales, fs=fs, nv=nv, backend=backend)
    W = wx.W                    # (n_scales, N) complex
    scale_arr = wx.scales       # (n_scales,) seconds
    n_scales = len(scale_arr)

    # --- Step 2: Derivative CWT via iω·ψ̂ filter ---
    # Full freq axis so ifft gives complex analytic output (matching the CWT above)
    omega = np.fft.fftfreq(N, d=1.0 / fs) * (2.0 * math.pi)  # rad/s, length N
    X_hat = np.fft.fft(x)  # complex, length N

    dW = np.empty((n_scales, N), dtype=np.complex128)
    for i, a in enumerate(scale_arr):
        _re, im_part = deriv_morlet_freq_response(omega, scale=a)
        # filter = iω·ψ̂(aω);  im_part = ω·ψ̂(aω) (real), so filter = 1j*im_part
        deriv_filter = im_part * 1j
        dW[i] = np.fft.ifft(X_hat * (math.sqrt(a) * deriv_filter))

    # --- Step 3: IF estimator ω̂(a,b) = -i · (∂_b W_x) / W_x ---
    W_abs = np.abs(W)
    gamma_val = _compute_gamma(gamma, W_abs, N)

    mask = W_abs > gamma_val
    W_safe = np.where(mask, W, 1.0 + 0j)
    omega_hat = np.real(-1j * dW / W_safe)   # (n_scales, N) rad/s

    # --- Step 4: Uniform frequency grid ---
    freqs_cwt = wx.freqs            # Hz, may be descending (small scale → high freq)
    f_min = float(freqs_cwt.min())
    f_max = float(freqs_cwt.max())
    n_freqs = n_scales
    freq_grid = np.linspace(f_min, f_max, n_freqs)   # uniform, ascending
    df = freq_grid[1] - freq_grid[0]

    # --- Step 5: Reassignment ---
    Tx = np.zeros((n_freqs, N), dtype=np.complex128)

    for i, a in enumerate(scale_arr):
        da_over_a = 1.0 / a
        for b in range(N):
            if not mask[i, b]:
                continue
            f_hat = omega_hat[i, b] / (2.0 * math.pi)   # rad/s → Hz
            k = int(round((f_hat - f_min) / df))
            if 0 <= k < n_freqs:
                Tx[k, b] += W[i, b] * da_over_a

    return SSTResult(Tx=Tx, Wx=wx, freqs=freq_grid, times=wx.times)
