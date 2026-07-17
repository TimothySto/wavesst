from __future__ import annotations

import numpy as np


def make_chirp(
    duration: float,
    fs: float,
    f_start: float | None = None,
    f_end: float | None = None,
    method: str = "linear",
    f_inst=None,
    t_start: float = 0.0,
    t_end: float | None = None,
    segments: list[tuple[float, float]] | None = None,
) -> np.ndarray:
    """
    Generate a chirp signal.

    Parameters
    ----------
    duration  : total signal duration in seconds
    fs        : sample rate in Hz
    f_start   : start frequency in Hz (required for method='linear'/'quadratic')
    f_end     : end frequency in Hz (required for method='linear'/'quadratic')
    method    : 'linear', 'quadratic', or 'arbitrary'
    f_inst    : instantaneous frequency; callable(t_arr)->freq_arr or ndarray of
                shape (N,); used when method='arbitrary'
    t_start   : onset in seconds; samples before this are zero (default: 0.0)
    t_end     : offset in seconds; samples after this are zero (default: duration)
    segments  : list of (frequency_hz, duration_s) for piecewise-constant IF;
                overrides f_start/f_end/method/f_inst when provided; any
                samples past the end of the last segment are zero

    Returns
    -------
    float32 ndarray of shape (N,) where N = int(duration * fs)
    """
    N = int(duration * fs)
    t = np.arange(N, dtype=np.float64) / fs

    if segments is not None:
        f_arr = _segments_to_f_arr(segments, N, fs)
    elif method == "linear":
        if f_start is None or f_end is None:
            raise ValueError("f_start and f_end are required for method='linear'")
        f_arr = f_start + (f_end - f_start) * t / duration
    elif method == "quadratic":
        if f_start is None or f_end is None:
            raise ValueError("f_start and f_end are required for method='quadratic'")
        f_arr = f_start + (f_end - f_start) * (t / duration) ** 2
    elif method == "arbitrary":
        if f_inst is None:
            raise ValueError("f_inst is required for method='arbitrary'")
        if callable(f_inst):
            f_arr = np.asarray(f_inst(t), dtype=np.float64)
        else:
            f_arr = np.asarray(f_inst, dtype=np.float64)
        if f_arr.shape != (N,):
            raise ValueError(
                f"f_inst array must have shape ({N},), got {f_arr.shape}"
            )
    else:
        raise ValueError(
            f"Unknown method {method!r}. Use 'linear', 'quadratic', or 'arbitrary'."
        )

    phase = 2.0 * np.pi * np.cumsum(f_arr) / fs
    x = np.cos(phase).astype(np.float32)

    i_start = int(t_start * fs)
    i_end = N if t_end is None else min(int(t_end * fs), N)
    if i_start > 0:
        x[:i_start] = 0.0
    if i_end < N:
        x[i_end:] = 0.0

    return x


def _segments_to_f_arr(
    segments: list[tuple[float, float]], N: int, fs: float
) -> np.ndarray:
    f_arr = np.zeros(N, dtype=np.float64)
    pos = 0
    for freq_hz, seg_dur in segments:
        n_seg = int(seg_dur * fs)
        end = min(pos + n_seg, N)
        f_arr[pos:end] = freq_hz
        pos = end
        if pos >= N:
            break
    return f_arr


def make_amfm(
    duration: float,
    fs: float,
    f_carrier: float,
    am_func=None,
    fm_func=None,
    t_start: float = 0.0,
    t_end: float | None = None,
) -> np.ndarray:
    """Placeholder — implemented in Task 3."""
    raise NotImplementedError("make_amfm implemented in Task 3")
