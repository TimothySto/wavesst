from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from wavesst.config import Config, config as _global_config
from wavesst.transforms.cwt import cwt, CWTResult
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
    wavelet_order: int | None = None,
    cfg: Config | None = None,
) -> MSSTResult:
    """
    Multi-Synchrosqueezing Transform — Pham-Meignen (2017) iterative squeezing.

    Pass 1: standard SST (W-derived IF).
    Passes 2+: Tx-derived IF estimator (true Pham-Meignen algorithm):
        dTx = IFFT[iω · FFT[Tx_{k-1}]]  along time axis
        ω̂(ξ,t) = Re(-i · dTx / Tx)      where |Tx| > gamma_tx
    For each scale a, the IF is looked up at the nearest freq-grid row ξ ≈ f_a,
    and W(a,t) is scattered into the new Tx using that IF estimate.

    n_iter=1 is equivalent to standard SST (identical output).

    Parameters
    ----------
    x            : 1-D array-like or torch.Tensor, length N
    wavelet      : "morlet"
    scales       : "auto" | int | np.ndarray
    fs           : sample rate in Hz
    nv           : voices per octave
    n_iter       : number of squeezing passes (1 = standard SST)
    gamma        : "auto" | "universal" | float | callable | None
    wavelet_order: reserved for future use
    cfg          : Config; defaults to wavesst.config module singleton

    Returns
    -------
    MSSTResult — Tx is torch.Tensor on CPU
    """
    if n_iter < 1:
        raise ValueError(f"n_iter must be >= 1, got {n_iter}")
    if cfg is None:
        cfg = _global_config

    # Pass 1: standard SST (W-derived IF) — unchanged
    result = sst(x, wavelet=wavelet, scales=scales, fs=fs, nv=nv, gamma=gamma, cfg=cfg)

    if n_iter == 1:
        return MSSTResult(
            Tx=result.Tx, Wx=result.Wx,
            freqs=result.freqs, times=result.times, n_iter=1,
        )

    # --- Passes 2+: true Pham-Meignen Tx-derived IF estimator ---
    device = cfg.resolve_device()
    real_dtype = cfg.real_dtype
    wx = result.Wx
    scale_arr = wx.scales       # (n_scales,)
    n_scales = len(scale_arr)
    N = wx.W.shape[1]
    freq_grid = result.freqs    # (n_freqs,) Hz, uniform, ascending
    f_min = float(freq_grid.min())
    n_freqs = len(freq_grid)
    df = float(freq_grid[1] - freq_grid[0])

    # Angular-frequency axis for time-derivative
    omega = torch.fft.fftfreq(N, d=1.0 / fs, device=device) * (2.0 * math.pi)

    np_float = np.float32 if real_dtype == torch.float32 else np.float64
    b_indices = torch.arange(N, device=device, dtype=torch.long)
    chunk_size = cfg.resolve_chunk_scales(N)

    # Precompute nearest freq-grid row for each scale (outside the iteration loop)
    scale_freqs = wx.freqs.astype(np_float)    # (n_scales,) Hz
    freq_arr = freq_grid.astype(np_float)       # (n_freqs,) Hz
    nearest_bin = np.argmin(
        np.abs(freq_arr[None, :] - scale_freqs[:, None]), axis=1
    )  # (n_scales,) int
    nearest_bin_dev = torch.from_numpy(nearest_bin.astype(np.int64)).to(device)

    Tx_prev = result.Tx   # CPU tensor, complex

    for _iteration in range(1, n_iter):
        # Compute time-derivative of Tx_{k-1} via FFT along time axis
        Tx_dev = Tx_prev.to(device=device, dtype=cfg.dtype)   # (n_freqs, N)
        Tx_hat = torch.fft.fft(Tx_dev, dim=-1)                # (n_freqs, N)
        dTx = torch.fft.ifft(1j * omega[None, :] * Tx_hat, dim=-1)  # (n_freqs, N)

        # Tx-derived IF field
        Tx_abs = Tx_dev.abs()
        tx_abs_nz = Tx_abs[Tx_abs > 0]
        gamma_tx = float(tx_abs_nz.median()) / 0.6745 if tx_abs_nz.numel() > 0 else 0.0
        Tx_safe = torch.where(
            Tx_abs > gamma_tx,
            Tx_dev,
            torch.ones(1, dtype=cfg.dtype, device=device),
        )
        omega_hat_tx = torch.real(-1j * dTx / Tx_safe)   # (n_freqs, N) rad/s
        mask_tx = Tx_abs > gamma_tx                        # (n_freqs, N)

        Tx_new_real = torch.zeros(n_freqs, N, dtype=real_dtype, device=device)
        Tx_new_imag = torch.zeros(n_freqs, N, dtype=real_dtype, device=device)

        da_ratio = float(math.log(scale_arr[1] / scale_arr[0]))

        for start in range(0, n_scales, chunk_size):
            end = min(start + chunk_size, n_scales)
            chunk = end - start

            W_chunk = wx.W[start:end].to(device=device, dtype=cfg.dtype)
            bins_chunk = nearest_bin_dev[start:end]        # (chunk,)
            omega_hat_chunk = omega_hat_tx[bins_chunk, :]  # (chunk, N)
            mask_chunk = mask_tx[bins_chunk, :]            # (chunk, N)

            f_hat = omega_hat_chunk / (2.0 * math.pi)
            k = torch.round((f_hat - f_min) / df).long()
            valid = mask_chunk & (k >= 0) & (k < n_freqs)

            weights = W_chunk * da_ratio

            b_exp = b_indices.unsqueeze(0).expand(chunk, N)
            k_flat = k[valid]
            b_flat = b_exp[valid]
            lin_idx = k_flat * N + b_flat
            Tx_new_real.view(-1).scatter_add_(0, lin_idx, weights.real[valid])
            Tx_new_imag.view(-1).scatter_add_(0, lin_idx, weights.imag[valid])

        Tx_prev = torch.complex(Tx_new_real, Tx_new_imag).cpu()

    return MSSTResult(
        Tx=Tx_prev, Wx=wx,
        freqs=freq_grid, times=wx.times, n_iter=n_iter,
    )
