from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from wavesst.config import Config, config as _global_config
from wavesst.transforms.cwt import cwt, CWTResult, MORLET_W0, _morlet_filter_bank
from wavesst.transforms.sst import sst, SSTResult, _compute_gamma


@dataclass
class MSSTResult:
    Tx: torch.Tensor    # complex, shape (n_freqs, n_samples), on CPU
    Wx: CWTResult       # underlying CWT (from first pass)
    freqs: np.ndarray   # float64, shape (n_freqs,) — uniform, ascending
    times: np.ndarray   # float64, shape (n_samples,)
    n_iter: int         # number of squeezing iterations applied


def msst(
    x,
    *,
    wavelet: str = "morlet",
    scales="auto",
    fs: float = 1.0,
    nv: int = 32,
    n_iter: int = 2,
    gamma="auto",
    cfg: Config | None = None,
) -> MSSTResult:
    """
    Multi-Synchrosqueezing Transform — iterative squeezing.

    n_iter=1 is equivalent to standard SST (identical output).
    For n_iter > 1, squeezing is applied n_iter times; each additional pass
    uses a progressively tighter gamma derived from the previous Tx energy,
    focusing energy onto the dominant ridges.

    Note: uses the W-derived IF estimator at each pass (not the Tx-derived
    estimator of Pham-Meignen 2017). A future session may add the full
    Pham-Meignen formulation.

    Parameters
    ----------
    x       : 1-D array-like or torch.Tensor, length N
    wavelet : "morlet"
    scales  : "auto" | int | np.ndarray
    fs      : sample rate in Hz
    nv      : voices per octave
    n_iter  : number of squeezing passes (1 = standard SST)
    gamma   : "auto" | "universal" | float | callable | None
    cfg     : Config; defaults to wavesst.config module singleton

    Returns
    -------
    MSSTResult — Tx is torch.Tensor on CPU
    """
    if n_iter < 1:
        raise ValueError(f"n_iter must be >= 1, got {n_iter}")
    if cfg is None:
        cfg = _global_config

    # First pass: standard SST
    result = sst(x, wavelet=wavelet, scales=scales, fs=fs, nv=nv, gamma=gamma, cfg=cfg)

    if n_iter == 1:
        return MSSTResult(
            Tx=result.Tx, Wx=result.Wx,
            freqs=result.freqs, times=result.times, n_iter=1,
        )

    # --- Additional passes use tighter gamma from previous Tx ---
    device = cfg.resolve_device()
    real_dtype = cfg.real_dtype
    wx = result.Wx
    scale_arr = wx.scales
    n_scales = len(scale_arr)
    N = wx.W.shape[1]
    freq_grid = result.freqs
    f_min = float(freq_grid.min())
    n_freqs = len(freq_grid)
    df = float(freq_grid[1] - freq_grid[0])

    omega = torch.fft.fftfreq(N, d=1.0 / fs, device=device) * (2.0 * math.pi)
    np_float = np.float32 if real_dtype == torch.float32 else np.float64
    if isinstance(x, torch.Tensor):
        x_dev = x.to(device=device, dtype=real_dtype)
    else:
        x_dev = torch.from_numpy(np.asarray(x, dtype=np_float)).to(device)
    X_hat = torch.fft.fft(x_dev)
    scales_dev = torch.from_numpy(scale_arr.astype(np_float)).to(device)
    b_indices = torch.arange(N, device=device, dtype=torch.long)
    chunk_size = cfg.resolve_chunk_scales(N)

    Tx_prev = result.Tx   # CPU tensor

    for _iteration in range(1, n_iter):
        # Tighten gamma: MAD estimator on non-zero Tx entries
        Tx_abs = Tx_prev.abs()
        nonzero = Tx_abs[Tx_abs > 0]
        gamma_val = float(nonzero.median()) / 0.6745 if nonzero.numel() > 0 else 0.0

        Tx_real = torch.zeros(n_freqs, N, dtype=real_dtype, device=device)
        Tx_imag = torch.zeros(n_freqs, N, dtype=real_dtype, device=device)

        for start in range(0, n_scales, chunk_size):
            end = min(start + chunk_size, n_scales)
            chunk = end - start
            scales_chunk = scales_dev[start:end]

            W_chunk = wx.W[start:end].to(device=device, dtype=cfg.dtype)

            psi_hat = _morlet_filter_bank(omega, scales_chunk, MORLET_W0, real_dtype)
            deriv_psi = torch.complex(
                torch.zeros_like(psi_hat),
                omega[None, :] * psi_hat,
            )
            dW_chunk = torch.fft.ifft(
                X_hat[None, :] * deriv_psi, dim=-1
            )

            W_abs_chunk = W_chunk.abs()
            mask_chunk = W_abs_chunk > gamma_val
            W_safe = torch.where(
                mask_chunk, W_chunk,
                torch.ones(1, dtype=cfg.dtype, device=device),
            )
            omega_hat = torch.real(-1j * dW_chunk / W_safe)

            f_hat = omega_hat / (2.0 * math.pi)
            k = torch.round((f_hat - f_min) / df).long()
            valid = mask_chunk & (k >= 0) & (k < n_freqs)
            da_ratio = float(math.log(scale_arr[1] / scale_arr[0]))
            weights = W_chunk * da_ratio
            b_exp = b_indices.unsqueeze(0).expand(chunk, N)
            k_flat = k[valid]
            b_flat = b_exp[valid]
            lin_idx = k_flat * N + b_flat
            Tx_real.view(-1).scatter_add_(0, lin_idx, weights.real[valid])
            Tx_imag.view(-1).scatter_add_(0, lin_idx, weights.imag[valid])

        Tx_prev = torch.complex(Tx_real, Tx_imag).cpu()

    return MSSTResult(
        Tx=Tx_prev, Wx=wx,
        freqs=freq_grid, times=wx.times, n_iter=n_iter,
    )
