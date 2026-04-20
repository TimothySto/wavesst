from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from wavesst.backends._protocol import ArrayBackend
from wavesst.backends import get_backend
from wavesst._core.filters import morlet_freq_response

MORLET_W0 = 6.0


@dataclass
class CWTResult:
    W: np.ndarray       # complex, shape (n_scales, n_samples)
    scales: np.ndarray  # float64, shape (n_scales,)
    freqs: np.ndarray   # float64, shape (n_scales,)
    times: np.ndarray   # float64, shape (n_samples,)


def _auto_scales(N: int, fs: float, nv: int = 32) -> np.ndarray:
    """Geometric scale array covering s0 to N/2 samples."""
    s0 = 2.0 / fs                           # smallest scale: 2 samples
    J = int(nv * math.log2(N / 2))         # number of scales
    j = np.arange(J, dtype=np.float64)
    return s0 * (2.0 ** (j / nv))


def _scales_to_freqs(scales: np.ndarray, w0: float = MORLET_W0) -> np.ndarray:
    # scales in seconds, omega in rad/s: peak at a*omega=w0 → f = w0/(2π·a)
    return w0 / (2.0 * math.pi * scales)


def cwt(
    x: np.ndarray,
    *,
    wavelet: str = "morlet",
    scales: str | int | np.ndarray = "auto",
    fs: float = 1.0,
    nv: int = 32,
    backend: ArrayBackend | None = None,
) -> CWTResult:
    """
    Continuous Wavelet Transform via FFT-based convolution.

    W_x(a, b) = IFFT[ X_hat(omega) * sqrt(a) * psi_hat*(a*omega) ]

    Parameters
    ----------
    x : 1-D float64 array, length N
    wavelet : "morlet" (only supported value in v1)
    scales : "auto" | int | ndarray
    fs : sample rate in Hz
    nv : voices per octave (used when scales="auto")
    backend : ArrayBackend instance; defaults to get_backend()

    Returns
    -------
    CWTResult
    """
    if wavelet != "morlet":
        raise ValueError(f"Unsupported wavelet '{wavelet}'; only 'morlet' is supported in v1")

    if backend is None:
        backend = get_backend()

    x = np.asarray(x, dtype=np.float64)
    N = len(x)

    # Build scale array
    if isinstance(scales, str) and scales == "auto":
        scale_arr = _auto_scales(N, fs, nv)
    elif isinstance(scales, (int, np.integer)):
        s0 = 2.0 / fs
        s_max = (N / 2) / fs
        scale_arr = np.geomspace(s0, s_max, int(scales))
    elif isinstance(scales, np.ndarray):
        scale_arr = scales.astype(np.float64)
    else:
        raise ValueError(f"scales must be 'auto', int, or ndarray; got {type(scales)}")

    n_scales = len(scale_arr)

    # Angular frequency axis for rfft output
    omega = np.fft.rfftfreq(N, d=1.0 / fs) * (2.0 * math.pi)  # rad/s, length N//2+1

    # Forward FFT of signal (once)
    X_hat = np.fft.rfft(x)  # complex, length N//2+1

    # Allocate output
    W = np.empty((n_scales, N), dtype=np.complex128)

    for i, a in enumerate(scale_arr):
        psi_hat = morlet_freq_response(omega, scale=a)     # real, (N//2+1,)
        # Multiply: X_hat * sqrt(a) * conj(psi_hat)
        # psi_hat is real so conj is identity; sqrt(a) normalises energy across scales
        product = X_hat * (math.sqrt(a) * psi_hat)
        W[i] = np.fft.irfft(product, n=N)

    freqs = _scales_to_freqs(scale_arr)
    times = np.arange(N, dtype=np.float64) / fs

    return CWTResult(W=W, scales=scale_arr, freqs=freqs, times=times)
