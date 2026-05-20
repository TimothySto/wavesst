from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from wavesst.config import Config, config as _global_config
from wavesst.transforms.wavelets import (
    _bump_filter_bank,
    _paul_filter_bank,
    _dog_filter_bank,
)

MORLET_W0 = 6.0
_PI_NEG_QUARTER = 0.7511255444649425  # pi^(-1/4)

_SUPPORTED_WAVELETS = ("morlet", "bump", "paul", "dog")


@dataclass
class CWTResult:
    W: torch.Tensor          # complex, shape (n_scales, n_samples), on CPU
    scales: np.ndarray       # float64, shape (n_scales,)
    freqs: np.ndarray        # float64, shape (n_scales,)
    times: np.ndarray        # float64, shape (n_samples,)
    cfg: Config = field(default_factory=Config, repr=False)
    wavelet: str = "morlet"
    wavelet_order: int | None = None


def _auto_scales(N: int, fs: float, nv: int = 32) -> np.ndarray:
    """Geometric scale array: s0=2/fs up to N/2 samples, nv voices per octave."""
    s0 = 2.0 / fs
    J = int(nv * math.log2(N / 2))
    j = np.arange(J, dtype=np.float64)
    return s0 * (2.0 ** (j / nv))


def _scales_to_freqs(scales: np.ndarray, w0: float = MORLET_W0) -> np.ndarray:
    """f = w0 / (2pi * a)  — scales in seconds, freqs in Hz."""
    return w0 / (2.0 * math.pi * scales)


def _get_wavelet_center(wavelet: str, wavelet_order: int | None) -> float:
    """Return the effective center frequency (w0 equivalent) for the wavelet.

    This value enters the scale-to-frequency formula: f_c = w0 / (2pi*a).
    """
    if wavelet in ("morlet", "bump"):
        return MORLET_W0
    elif wavelet == "paul":
        m = wavelet_order if wavelet_order is not None else 4
        if m < 1:
            raise ValueError(f"wavelet_order for Paul wavelet must be >= 1, got {m}")
        return float(m)
    elif wavelet == "dog":
        m = wavelet_order if wavelet_order is not None else 2
        if m < 1:
            raise ValueError(f"wavelet_order for DOG wavelet must be >= 1, got {m}")
        return math.sqrt(float(m))
    else:
        raise ValueError(
            f"Unsupported wavelet '{wavelet}'. Choose from: {_SUPPORTED_WAVELETS}"
        )


def _morlet_filter_bank(
    omega: torch.Tensor,
    scales: torch.Tensor,
    w0: float,
    real_dtype: torch.dtype,
) -> torch.Tensor:
    """Morlet filter bank. Returns psi_hat (n_chunk, N) real."""
    scaled_omega = scales[:, None] * omega[None, :]
    gaussian = _PI_NEG_QUARTER * torch.exp(
        -0.5 * (scaled_omega - w0) ** 2
    ).to(real_dtype)
    psi_hat = torch.where(
        scaled_omega > 0.0,
        gaussian,
        torch.zeros(1, dtype=real_dtype, device=omega.device),
    )
    return psi_hat


def _get_filter_bank(
    wavelet: str,
    wavelet_order: int | None,
    omega: torch.Tensor,
    scales: torch.Tensor,
    real_dtype: torch.dtype,
) -> torch.Tensor:
    """Dispatch to the correct filter bank. Returns psi_hat (n_scales, N) real."""
    if wavelet == "morlet":
        return _morlet_filter_bank(omega, scales, MORLET_W0, real_dtype)
    elif wavelet == "bump":
        return _bump_filter_bank(omega, scales, MORLET_W0, real_dtype)
    elif wavelet == "paul":
        m = wavelet_order if wavelet_order is not None else 4
        return _paul_filter_bank(omega, scales, m, real_dtype)
    elif wavelet == "dog":
        m = wavelet_order if wavelet_order is not None else 2
        return _dog_filter_bank(omega, scales, m, real_dtype)
    else:
        raise ValueError(
            f"Unsupported wavelet '{wavelet}'. Choose from: {_SUPPORTED_WAVELETS}"
        )


def cwt(
    x,
    *,
    wavelet: str = "morlet",
    wavelet_order: int | None = None,
    scales="auto",
    fs: float = 1.0,
    nv: int = 32,
    f_low: float | None = None,
    f_high: float | None = None,
    cfg: Config | None = None,
) -> CWTResult:
    """
    Continuous Wavelet Transform via batched FFT (torch-native).

    W_x(a, b) = IFFT[ X_hat(omega) * psi_hat*(a*omega) ]

    Supports wavelets: "morlet" (default), "bump", "paul", "dog".
    Paul and DOG accept an optional wavelet_order (defaults: Paul=4, DOG=2).
    Optional f_low/f_high (Hz) restrict computed scales to those whose
    centre frequency falls within [f_low, f_high].

    Parameters
    ----------
    x            : 1-D array-like or torch.Tensor, length N
    wavelet      : one of "morlet", "bump", "paul", "dog"
    wavelet_order: order parameter for Paul/DOG (int or None)
    scales       : "auto" | int | np.ndarray
    fs           : sample rate in Hz
    nv           : voices per octave (used when scales="auto")
    f_low        : lower frequency limit in Hz (optional)
    f_high       : upper frequency limit in Hz (optional)
    cfg          : Config; defaults to wavesst.config module singleton

    Returns
    -------
    CWTResult with wavelet and wavelet_order fields populated
    """
    # Validate wavelet early (raises ValueError if unknown)
    w0 = _get_wavelet_center(wavelet, wavelet_order)

    if cfg is None:
        cfg = _global_config

    device = cfg.resolve_device()
    real_dtype = cfg.real_dtype

    # --- Convert input ---
    if isinstance(x, torch.Tensor):
        x_dev = x.to(device=device, dtype=real_dtype)
    else:
        np_float = np.float32 if real_dtype == torch.float32 else np.float64
        arr = np.asarray(x, dtype=np_float)
        x_dev = torch.from_numpy(arr).to(device)
    N = x_dev.shape[0]

    # --- Build scale array ---
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

    # --- Apply f_low / f_high band-limiting ---
    if f_low is not None or f_high is not None:
        freqs_candidate = _scales_to_freqs(scale_arr, w0)
        mask = np.ones(len(scale_arr), dtype=bool)
        if f_low is not None:
            mask &= freqs_candidate >= f_low
        if f_high is not None:
            mask &= freqs_candidate <= f_high
        scale_arr = scale_arr[mask]
        if len(scale_arr) == 0:
            raise ValueError(
                f"No scales remain after frequency band-limiting "
                f"(f_low={f_low}, f_high={f_high}). "
                f"Available frequency range: "
                f"[{freqs_candidate.min():.3g}, {freqs_candidate.max():.3g}] Hz."
            )

    n_scales = len(scale_arr)
    np_float = np.float32 if real_dtype == torch.float32 else np.float64

    omega = torch.fft.fftfreq(N, d=1.0 / fs, device=device) * (2.0 * math.pi)
    X_hat = torch.fft.fft(x_dev)
    W_cpu = torch.empty(n_scales, N, dtype=cfg.dtype, device='cpu')
    chunk_size = cfg.resolve_chunk_scales(N)
    scales_dev = torch.from_numpy(scale_arr.astype(np_float)).to(device)

    for start in range(0, n_scales, chunk_size):
        end = min(start + chunk_size, n_scales)
        scales_chunk = scales_dev[start:end]
        psi_hat = _get_filter_bank(wavelet, wavelet_order, omega, scales_chunk, real_dtype)
        product = X_hat[None, :] * psi_hat
        W_chunk = torch.fft.ifft(product, dim=-1)
        W_cpu[start:end] = W_chunk.to(device='cpu', dtype=cfg.dtype)

    freqs = _scales_to_freqs(scale_arr, w0)
    times = np.arange(N, dtype=np.float64) / fs

    # Resolve defaults for wavelet_order
    resolved_order: int | None = None
    if wavelet == "paul":
        resolved_order = wavelet_order if wavelet_order is not None else 4
    elif wavelet == "dog":
        resolved_order = wavelet_order if wavelet_order is not None else 2

    return CWTResult(
        W=W_cpu, scales=scale_arr, freqs=freqs, times=times, cfg=cfg,
        wavelet=wavelet, wavelet_order=resolved_order,
    )
