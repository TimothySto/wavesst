from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from wavesst.config import Config, config as _global_config
from wavesst.transforms.stft import stft, STFTResult


@dataclass
class STFTSSTResult:
    Tx: torch.Tensor    # complex, shape (n_freqs, n_frames)
    Vx: STFTResult      # underlying STFT
    freqs: torch.Tensor # float, shape (n_freqs,) Hz — same grid as Vx.freqs
    times: torch.Tensor # float, shape (n_frames,) seconds — same as Vx.times


def stft_sst(
    x,
    *,
    fs: float = 1.0,
    window: str = "hann",
    nperseg: int = 256,
    noverlap: int | None = None,
    gamma="auto",
    cfg: Config | None = None,
) -> STFTSSTResult:
    """
    STFT-based Synchrosqueezing Transform.

    IF estimator: ω̂(η,τ) = η - Im(∂_τV_x / V_x) / (2π)

    ∂_τV_x is computed via central finite difference on adjacent frames
    (forward/backward at signal boundaries). Reassignment onto the STFT's
    own uniform frequency grid via scatter_add_.

    Parameters
    ----------
    x        : 1-D array-like or torch.Tensor, length N
    fs       : sample rate in Hz
    window   : "hann" | "hamming"
    nperseg  : FFT window size (samples)
    noverlap : overlap in samples (default: nperseg // 2)
    gamma    : "auto" | float | None — magnitude threshold on |V|
    cfg      : Config; defaults to wavesst.config module singleton

    Returns
    -------
    STFTSSTResult — Tx on device (same device as STFT output)
    """
    if cfg is None:
        cfg = _global_config

    vx = stft(x, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap, cfg=cfg)
    V = vx.V            # (n_freqs, n_frames) complex
    n_freqs, n_frames = V.shape
    device = V.device
    dt = vx.hop / fs    # seconds per frame

    # --- Threshold on |V| ---
    V_abs = V.abs()
    if gamma == "auto":
        gamma_val = float(V_abs[-1].median()) / 0.6745
    elif gamma is None:
        gamma_val = 0.0
    elif isinstance(gamma, (int, float)):
        gamma_val = float(gamma)
    else:
        raise ValueError(f"gamma must be 'auto', float, or None; got {gamma!r}")

    mask = V_abs > gamma_val   # (n_freqs, n_frames) bool

    # --- Time derivative of V via central finite difference ---
    # ∂_τV ≈ (V[:,τ+1] - V[:,τ-1]) / (2·dt)
    dV = torch.empty_like(V)
    if n_frames >= 3:
        dV[:, 1:-1] = (V[:, 2:] - V[:, :-2]) / (2.0 * dt)
        dV[:, 0]    = (V[:, 1]  - V[:, 0])   / dt
        dV[:, -1]   = (V[:, -1] - V[:, -2])  / dt
    elif n_frames == 2:
        dV[:, 0] = (V[:, 1] - V[:, 0]) / dt
        dV[:, 1] = (V[:, 1] - V[:, 0]) / dt
    else:  # single frame
        dV[:] = 0.0

    # --- IF estimator ω̂(η,τ) = η - Im(∂_τV / V) / (2π) ---
    V_safe = torch.where(mask, V, torch.ones(1, dtype=V.dtype, device=device))
    eta = vx.freqs                                                     # (n_freqs,) Hz
    omega_hat = eta[:, None] - (dV / V_safe).imag / (2.0 * math.pi)  # (n_freqs, n_frames) Hz

    # --- Vectorised scatter onto same uniform freq grid ---
    f_min = float(eta[0])
    df = float(eta[1] - eta[0]) if n_freqs > 1 else 1.0

    real_dtype = cfg.real_dtype
    Tx_real = torch.zeros(n_freqs, n_frames, dtype=real_dtype, device=device)
    Tx_imag = torch.zeros(n_freqs, n_frames, dtype=real_dtype, device=device)

    k = torch.round((omega_hat - f_min) / df).long()  # (n_freqs, n_frames)
    valid = mask & (k >= 0) & (k < n_freqs)           # (n_freqs, n_frames)

    t_idx = torch.arange(n_frames, device=device).unsqueeze(0).expand(n_freqs, n_frames)
    k_flat = k[valid]
    t_flat = t_idx[valid]
    lin_idx = k_flat * n_frames + t_flat

    Tx_real.view(-1).scatter_add_(0, lin_idx, V.real[valid])
    Tx_imag.view(-1).scatter_add_(0, lin_idx, V.imag[valid])

    Tx = torch.complex(Tx_real, Tx_imag)

    return STFTSSTResult(Tx=Tx, Vx=vx, freqs=eta, times=vx.times)
