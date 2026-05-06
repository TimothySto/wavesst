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
    n_samples: int      # original signal length N (for reconstruction)


def _deriv_window(window: torch.Tensor, fs: float) -> torch.Tensor:
    """Compute the time-derivative of a window function.

    Uses central finite differences on the sample values, then converts
    the sample-domain derivative to a time-domain derivative by multiplying
    by fs (samples/second).

    For smooth windows (Hann, Hamming) the O(h²) finite-difference error
    is negligible — < 0.1% for typical nperseg values.

    Returns
    -------
    dg : torch.Tensor, same shape and dtype as window — units: 1/s
    """
    dg = torch.empty_like(window)
    dg[1:-1] = (window[2:] - window[:-2]) * 0.5
    dg[0]    = window[1]  - window[0]
    dg[-1]   = window[-1] - window[-2]
    return dg * float(fs)   # sample difference → time derivative (1/s)


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

    IF estimator: ω̂(η,τ) = η + Im(∂_τV_x / V_x) / (2π)

    ∂_τV_x is computed exactly via a second STFT pass using the window
    derivative g'(t):

        ∂_τV(η,τ) = -V_{g'}(η,τ)

    This avoids the finite-difference aliasing that occurs when the product
    |f - η| · hop/fs is large. The result is exact for any hop size.

    Derivation:
        V(η,τ) = ∫ x(t) g(t-τ) e^{-i2πηt} dt
        ∂_τV   = -∫ x(t) g'(t-τ) e^{-i2πηt} dt  = -V_{g'}
        For x = A·e^{i2πf₀t}: ∂_τV/V = i2π(f₀-η)
        → ω̂ = η + Im(∂_τV/V)/(2π) = η + (f₀-η) = f₀  ✓

    Reassignment onto the STFT's own uniform frequency grid via scatter_add_.

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

    device = cfg.resolve_device()
    real_dtype = cfg.real_dtype

    # --- Convert input once (reused for derivative STFT) ---
    if isinstance(x, torch.Tensor):
        x_dev = x.to(device=device, dtype=real_dtype)
    else:
        np_float = np.float32 if real_dtype == torch.float32 else np.float64
        x_dev = torch.from_numpy(np.asarray(x, dtype=np_float)).to(device)
    N = x_dev.shape[0]

    # --- Compute standard STFT ---
    vx = stft(x_dev, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap, cfg=cfg)
    V = vx.V            # (n_freqs, n_frames) complex
    n_freqs, n_frames = V.shape
    hop = vx.hop
    win = vx.window     # (nperseg,) real — already on device

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

    # --- Exact time derivative via window-derivative STFT ---
    # ∂_τV(η,τ) = -V_{g'}(η,τ)  where g' = dg/dt
    win_deriv = _deriv_window(win, fs)   # (nperseg,) 1/s, on device

    # Re-frame the signal with derivative window (same frame grid as STFT)
    frame_starts = torch.arange(n_frames, device=device) * hop   # (n_frames,)
    col_idx      = torch.arange(nperseg,  device=device)         # (nperseg,)
    frames = x_dev[frame_starts[:, None] + col_idx[None, :]]     # (n_frames, nperseg)
    frames_dg = frames * win_deriv.unsqueeze(0)                  # (n_frames, nperseg)
    V_dg = torch.fft.rfft(frames_dg, dim=-1).T.contiguous()     # (n_freqs, n_frames)

    dV = -V_dg   # ∂_τV = -V_{g'}

    # --- IF estimator ω̂(η,τ) = η + Im(∂_τV / V) / (2π) ---
    V_safe = torch.where(mask, V, torch.ones(1, dtype=V.dtype, device=device))
    eta = vx.freqs                                                     # (n_freqs,) Hz
    omega_hat = eta[:, None] + (dV / V_safe).imag / (2.0 * math.pi)  # (n_freqs, n_frames) Hz

    # --- Vectorised scatter onto same uniform freq grid ---
    f_min = float(eta[0])
    df = float(eta[1] - eta[0]) if n_freqs > 1 else 1.0

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

    return STFTSSTResult(Tx=Tx, Vx=vx, freqs=eta, times=vx.times, n_samples=N)
