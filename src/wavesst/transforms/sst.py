from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from wavesst.config import Config, config as _global_config
from wavesst.transforms.cwt import (
    cwt, CWTResult, MORLET_W0, _morlet_filter_bank,
)


@dataclass
class SSTResult:
    Tx: torch.Tensor    # complex, shape (n_freqs, n_samples), on CPU
    Wx: CWTResult       # underlying CWT
    freqs: np.ndarray   # float64, shape (n_freqs,) — uniform, ascending
    times: np.ndarray   # float64, shape (n_samples,)


def _compute_gamma(gamma, W_abs: torch.Tensor, N: int) -> float:
    """Resolve gamma to a float threshold value.

    W_abs is a CPU torch Tensor (real). If gamma is callable, it receives
    W_abs as a numpy array with dtype matching cfg.real_dtype.
    """
    if gamma is None:
        return 0.0
    if callable(gamma):
        return float(gamma(W_abs.numpy()))
    if isinstance(gamma, (int, float)):
        return float(gamma)
    if gamma == "auto":
        # MAD estimator on finest-scale (highest-freq) coefficients
        finest = W_abs[-1]
        sigma_hat = float(finest.median()) / 0.6745
        return sigma_hat
    if gamma == "universal":
        finest = W_abs[-1]
        sigma_hat = float(finest.median()) / 0.6745
        return sigma_hat * math.sqrt(2.0 * math.log(N))
    raise ValueError(
        f"gamma must be 'auto', 'universal', float, callable, or None; got {gamma!r}"
    )


def sst(
    x,
    *,
    wavelet: str = "morlet",
    scales="auto",
    fs: float = 1.0,
    nv: int = 32,
    gamma="auto",
    cfg: Config | None = None,
) -> SSTResult:
    """
    CWT-based Synchrosqueezing Transform (torch-native).

    IF estimator:  ω̂(a,b) = Re[ -i · (∂_b W_x) / W_x ]
    Reassignment:  T_x(f,b) += W_x(a,b)/a  where round(ω̂/2π) == f bin

    Uses vectorised scatter_add_ (real+imag split) — no Python loop over time.
    W streams from CPU RAM back to device in chunks; Tx accumulates on device.

    Parameters
    ----------
    x       : 1-D array-like or torch.Tensor, length N
    wavelet : "morlet"
    scales  : "auto" | int | np.ndarray
    fs      : sample rate in Hz
    nv      : voices per octave
    gamma   : "auto" | "universal" | float | callable | None
    cfg     : Config; defaults to wavesst.config module singleton

    Returns
    -------
    SSTResult — Tx is torch.Tensor on CPU
    """
    if wavelet != "morlet":
        raise ValueError(f"Unsupported wavelet '{wavelet}'")

    if cfg is None:
        cfg = _global_config

    device = cfg.resolve_device()
    real_dtype = cfg.real_dtype

    # --- Step 1: CWT (W stages in CPU RAM) ---
    wx = cwt(x, wavelet=wavelet, scales=scales, fs=fs, nv=nv, cfg=cfg)
    scale_arr = wx.scales   # (n_scales,) float64 numpy
    n_scales = len(scale_arr)
    N = wx.W.shape[1]

    # --- Step 2: Threshold ---
    W_abs_cpu = wx.W.abs()   # (n_scales, N) real, CPU
    gamma_val = _compute_gamma(gamma, W_abs_cpu, N)

    # --- Step 3: Uniform frequency grid ---
    freqs_cwt = wx.freqs
    f_min = float(freqs_cwt.min())
    f_max = float(freqs_cwt.max())
    n_freqs = n_scales
    freq_grid = np.linspace(f_min, f_max, n_freqs)
    df = float(freq_grid[1] - freq_grid[0])

    # --- Step 4: Allocate Tx accumulators on device ---
    Tx_real = torch.zeros(n_freqs, N, dtype=real_dtype, device=device)
    Tx_imag = torch.zeros(n_freqs, N, dtype=real_dtype, device=device)

    # Pre-compute on device: omega axis and X_hat (for derivative filter)
    omega = torch.fft.fftfreq(N, d=1.0 / fs, device=device) * (2.0 * math.pi)

    if isinstance(x, torch.Tensor):
        x_dev = x.to(device=device, dtype=real_dtype)
    else:
        np_float = np.float32 if real_dtype == torch.float32 else np.float64
        x_dev = torch.from_numpy(np.asarray(x, dtype=np_float)).to(device)
    X_hat = torch.fft.fft(x_dev)   # (N,) complex

    # Time index broadcast template
    b_indices = torch.arange(N, device=device, dtype=torch.long)  # (N,)

    # --- Step 5: Chunked reassignment ---
    chunk_size = cfg.resolve_chunk_scales(N)
    np_float = np.float32 if real_dtype == torch.float32 else np.float64
    scales_dev = torch.from_numpy(scale_arr.astype(np_float)).to(device)

    for start in range(0, n_scales, chunk_size):
        end = min(start + chunk_size, n_scales)
        chunk = end - start
        scales_chunk = scales_dev[start:end]      # (chunk,)
        sqrt_a = scales_chunk[:, None] ** 0.5     # (chunk, 1)

        # Load W_chunk from CPU RAM onto device
        W_chunk = wx.W[start:end].to(device=device, dtype=cfg.dtype)  # (chunk, N)

        # Build derivative filter: i·ω·ψ̂(aω) = complex(0, ω·ψ̂)
        psi_hat = _morlet_filter_bank(omega, scales_chunk, MORLET_W0, real_dtype)
        deriv_psi = torch.complex(
            torch.zeros_like(psi_hat),
            omega[None, :] * psi_hat,
        )  # (chunk, N) complex

        # dW = IFFT[ X_hat · √a · i·ω·ψ̂(aω) ]
        dW_chunk = torch.fft.ifft(
            X_hat[None, :] * (sqrt_a * deriv_psi), dim=-1
        )  # (chunk, N) complex

        # IF estimator: ω̂ = Re(-i · dW / W)
        W_abs_chunk = W_chunk.abs()
        mask_chunk = W_abs_chunk > gamma_val
        W_safe = torch.where(
            mask_chunk,
            W_chunk,
            torch.ones(1, dtype=cfg.dtype, device=device),
        )
        omega_hat = torch.real(-1j * dW_chunk / W_safe)   # (chunk, N) real rad/s

        # --- Vectorised scatter ---
        f_hat = omega_hat / (2.0 * math.pi)               # (chunk, N) Hz
        k = torch.round((f_hat - f_min) / df).long()      # (chunk, N) int64
        valid = mask_chunk & (k >= 0) & (k < n_freqs)     # (chunk, N) bool

        weights = W_chunk / scales_chunk[:, None]          # (chunk, N) complex

        # Expand time indices and flatten valid entries
        b_exp = b_indices.unsqueeze(0).expand(chunk, N)   # (chunk, N)
        k_flat = k[valid]                                  # (M,)
        b_flat = b_exp[valid]                              # (M,)
        w_real = weights.real[valid]                       # (M,)
        w_imag = weights.imag[valid]                       # (M,)

        # Row-major linear index into flattened (n_freqs, N) Tx
        lin_idx = k_flat * N + b_flat                      # (M,)

        Tx_real.view(-1).scatter_add_(0, lin_idx, w_real)
        Tx_imag.view(-1).scatter_add_(0, lin_idx, w_imag)

    # Combine and move to CPU
    Tx = torch.complex(Tx_real, Tx_imag).cpu()

    return SSTResult(Tx=Tx, Wx=wx, freqs=freq_grid, times=wx.times)
