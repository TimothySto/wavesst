from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from wavesst.config import Config, config as _global_config

MORLET_W0 = 6.0
_PI_NEG_QUARTER = 0.7511255444649425  # pi^(-1/4)


@dataclass
class CWTResult:
    W: torch.Tensor     # complex, shape (n_scales, n_samples), on CPU
    scales: np.ndarray  # float64, shape (n_scales,)
    freqs: np.ndarray   # float64, shape (n_scales,)
    times: np.ndarray   # float64, shape (n_samples,)
    cfg: Config = field(default_factory=Config, repr=False)


def _auto_scales(N: int, fs: float, nv: int = 32) -> np.ndarray:
    """Geometric scale array: s0=2/fs up to N/2 samples, nv voices per octave."""
    s0 = 2.0 / fs
    J = int(nv * math.log2(N / 2))
    j = np.arange(J, dtype=np.float64)
    return s0 * (2.0 ** (j / nv))


def _scales_to_freqs(scales: np.ndarray, w0: float = MORLET_W0) -> np.ndarray:
    """f = w0 / (2π · a)  — scales in seconds, freqs in Hz."""
    return w0 / (2.0 * math.pi * scales)


def _morlet_filter_bank(
    omega: torch.Tensor,    # (N,) float rad/s, on device
    scales: torch.Tensor,   # (n_chunk,) float, on device
    w0: float,
    real_dtype: torch.dtype,
) -> torch.Tensor:
    """
    Build 2D Morlet filter bank for a chunk of scales via broadcasting.

    Returns psi_hat of shape (n_chunk, N), real dtype.
    psi_hat[i,j] = PI^(-1/4) * exp(-0.5*(a_i*omega_j - w0)^2)  for a_i*omega_j > 0
                 = 0                                              otherwise
    """
    scaled_omega = scales[:, None] * omega[None, :]          # (n_chunk, N)
    gaussian = _PI_NEG_QUARTER * torch.exp(
        -0.5 * (scaled_omega - w0) ** 2
    ).to(real_dtype)
    psi_hat = torch.where(
        scaled_omega > 0.0,
        gaussian,
        torch.zeros(1, dtype=real_dtype, device=omega.device),
    )
    return psi_hat  # (n_chunk, N) real


def cwt(
    x,
    *,
    wavelet: str = "morlet",
    scales="auto",
    fs: float = 1.0,
    nv: int = 32,
    cfg: Config | None = None,
) -> CWTResult:
    """
    Continuous Wavelet Transform via batched FFT (torch-native).

    W_x(a, b) = IFFT[ X_hat(ω) · √a · ψ̂*(a·ω) ]

    Uses full-spectrum fft/ifft (NOT rfft) so W is the complex analytic CWT.
    The Morlet wavelet is analytic (ψ̂(ω)=0 for ω≤0), so the complex-valued
    output contains the instantaneous amplitude and phase directly.

    Large W matrix stages in CPU RAM as a torch CPU tensor; only one chunk
    at a time occupies VRAM. Use cfg.resolve_chunk_scales(N) for sizing.

    Parameters
    ----------
    x       : 1-D array-like or torch.Tensor, length N
    wavelet : "morlet" (only supported in v1)
    scales  : "auto" | int | np.ndarray  (scales in seconds)
    fs      : sample rate in Hz
    nv      : voices per octave (used when scales="auto")
    cfg     : Config; defaults to wavesst.config module singleton

    Returns
    -------
    CWTResult
        W      — torch.Tensor, complex, (n_scales, N), on CPU
        scales — np.ndarray float64, (n_scales,)
        freqs  — np.ndarray float64, (n_scales,) Hz; f = w0/(2π·a)
        times  — np.ndarray float64, (N,) seconds
        cfg    — the Config used
    """
    if wavelet != "morlet":
        raise ValueError(f"Unsupported wavelet '{wavelet}'; only 'morlet' supported in v1")

    if cfg is None:
        cfg = _global_config

    device = cfg.resolve_device()
    real_dtype = cfg.real_dtype

    # --- Convert input to real tensor on device ---
    if isinstance(x, torch.Tensor):
        x_dev = x.to(device=device, dtype=real_dtype)
    else:
        np_float = np.float32 if real_dtype == torch.float32 else np.float64
        arr = np.asarray(x, dtype=np_float)
        x_dev = torch.from_numpy(arr).to(device)
    N = x_dev.shape[0]

    # --- Build scale array (numpy, stays on CPU for Cython compatibility) ---
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
    np_float = np.float32 if real_dtype == torch.float32 else np.float64

    # --- Angular frequency axis on device (rad/s), full spectrum ---
    omega = torch.fft.fftfreq(N, d=1.0 / fs, device=device) * (2.0 * math.pi)

    # --- Forward FFT of signal once ---
    X_hat = torch.fft.fft(x_dev)   # (N,) complex

    # --- Allocate W in CPU RAM (torch CPU tensor) ---
    W_cpu = torch.empty(n_scales, N, dtype=cfg.dtype, device='cpu')

    # --- Chunked scale loop: each chunk processed on device, then offloaded ---
    chunk_size = cfg.resolve_chunk_scales(N)
    scales_dev = torch.from_numpy(scale_arr.astype(np_float)).to(device)

    for start in range(0, n_scales, chunk_size):
        end = min(start + chunk_size, n_scales)
        scales_chunk = scales_dev[start:end]          # (chunk,)
        sqrt_a = scales_chunk[:, None] ** 0.5         # (chunk, 1)

        psi_hat = _morlet_filter_bank(omega, scales_chunk, MORLET_W0, real_dtype)
        # product shape: (chunk, N) complex
        product = X_hat[None, :] * (sqrt_a * psi_hat)
        W_chunk = torch.fft.ifft(product, dim=-1)     # (chunk, N) complex

        # Offload to CPU RAM, cast to target complex dtype
        W_cpu[start:end] = W_chunk.to(device='cpu', dtype=cfg.dtype)

    freqs = _scales_to_freqs(scale_arr)
    times = np.arange(N, dtype=np.float64) / fs

    return CWTResult(W=W_cpu, scales=scale_arr, freqs=freqs, times=times, cfg=cfg)
