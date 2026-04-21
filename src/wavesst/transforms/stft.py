from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from wavesst.config import Config, config as _global_config


@dataclass
class STFTResult:
    V: torch.Tensor       # complex, shape (n_freqs, n_frames)
    freqs: torch.Tensor   # float, shape (n_freqs,) Hz — 0 .. fs/2
    times: torch.Tensor   # float, shape (n_frames,) seconds — frame centres
    hop: int              # hop length in samples
    window: torch.Tensor  # float, shape (nperseg,)


def stft(
    x,
    *,
    fs: float = 1.0,
    window: str = "hann",
    nperseg: int = 256,
    noverlap: int | None = None,
    cfg: Config | None = None,
) -> STFTResult:
    """
    Short-Time Fourier Transform (GPU-native via torch.fft.rfft).

    Uses rfft (one-sided spectrum, correct for real-signal STFT — unlike the
    CWT which requires full fft/ifft for the analytic Morlet filter).

    Parameters
    ----------
    x        : 1-D array-like or torch.Tensor, length N
    fs       : sample rate in Hz
    window   : "hann" | "hamming"
    nperseg  : FFT window size (samples)
    noverlap : overlap in samples (default: nperseg // 2)
    cfg      : Config; defaults to wavesst.config module singleton

    Returns
    -------
    STFTResult
        V      — torch.Tensor, complex, (n_freqs, n_frames), on device
        freqs  — torch.Tensor, float, (n_freqs,) Hz; 0 to fs/2
        times  — torch.Tensor, float, (n_frames,) seconds; frame centres
        hop    — hop length in samples
        window — the analysis window tensor
    """
    if cfg is None:
        cfg = _global_config

    device = cfg.resolve_device()
    real_dtype = cfg.real_dtype

    if noverlap is None:
        noverlap = nperseg // 2
    if noverlap >= nperseg:
        raise ValueError(f"noverlap ({noverlap}) must be < nperseg ({nperseg})")
    hop = nperseg - noverlap

    # --- Build window ---
    if window == "hann":
        win = torch.hann_window(nperseg, device=device, dtype=real_dtype)
    elif window == "hamming":
        win = torch.hamming_window(nperseg, device=device, dtype=real_dtype)
    else:
        raise ValueError(f"Unknown window '{window}'; supported: 'hann', 'hamming'")

    # --- Convert input ---
    if isinstance(x, torch.Tensor):
        x_dev = x.to(device=device, dtype=real_dtype)
    else:
        np_float = np.float32 if real_dtype == torch.float32 else np.float64
        x_dev = torch.from_numpy(np.asarray(x, dtype=np_float)).to(device)
    N = x_dev.shape[0]

    # --- Frame extraction via advanced indexing (no loops) ---
    n_frames = max(0, (N - nperseg) // hop + 1)
    frame_starts = torch.arange(n_frames, device=device) * hop   # (n_frames,)
    col_idx = torch.arange(nperseg, device=device)               # (nperseg,)
    # frames[i, j] = x[frame_starts[i] + j]
    frames = x_dev[frame_starts[:, None] + col_idx[None, :]]     # (n_frames, nperseg)
    frames = frames * win.unsqueeze(0)                            # apply window

    # --- One-sided rfft → (n_frames, n_freqs) ---
    V_frames = torch.fft.rfft(frames, dim=-1)   # (n_frames, n_freqs) complex
    V = V_frames.T.contiguous()                 # (n_freqs, n_frames)

    freqs = torch.fft.rfftfreq(nperseg, d=1.0 / fs, device=device)  # (n_freqs,) Hz
    # Frame centre times: mid-point of each window
    times = (frame_starts.to(real_dtype) + nperseg / 2.0) / fs      # (n_frames,) seconds

    return STFTResult(V=V, freqs=freqs, times=times, hop=hop, window=win)
