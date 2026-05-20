from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _to_numpy(x):
    """Convert a torch.Tensor (any device) or array-like to a numpy array."""
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(x)


# ---------------------------------------------------------------------------
# Public plot functions
# ---------------------------------------------------------------------------

def plot_cwt(cwt_result, ax=None, **kwargs):
    """
    Plot CWT magnitude as a time-frequency heatmap.

    Parameters
    ----------
    cwt_result : CWTResult — W may be on CPU or GPU
    ax         : matplotlib.axes.Axes or None (creates new figure)
    **kwargs   : forwarded to ax.pcolormesh

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()

    W_mag = _to_numpy(cwt_result.W.abs())   # (n_scales, N)
    times = _to_numpy(cwt_result.times)
    freqs = _to_numpy(cwt_result.freqs)

    kwargs.setdefault("shading", "auto")
    kwargs.setdefault("cmap", "inferno")
    ax.pcolormesh(times, freqs, W_mag, **kwargs)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("CWT Magnitude")
    return ax


def plot_sst(sst_result, ax=None, **kwargs):
    """
    Plot SST (or STFT-SST) magnitude as a time-frequency heatmap.

    Parameters
    ----------
    sst_result : SSTResult or STFTSSTResult — Tx may be on CPU or GPU
    ax         : matplotlib.axes.Axes or None
    **kwargs   : forwarded to ax.pcolormesh

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()

    Tx_mag = _to_numpy(sst_result.Tx.abs())   # (n_freqs, N_or_frames)
    times  = _to_numpy(sst_result.times)
    freqs  = _to_numpy(sst_result.freqs)

    kwargs.setdefault("shading", "auto")
    kwargs.setdefault("cmap", "inferno")
    ax.pcolormesh(times, freqs, Tx_mag, **kwargs)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("SST Magnitude")
    return ax


def plot_ridges(sst_result, ridges, ax=None, colors=None, **kwargs):
    """
    Plot SST magnitude with ridge paths overlaid.

    Parameters
    ----------
    sst_result : SSTResult or STFTSSTResult
    ridges     : list[Ridge]
    ax         : matplotlib.axes.Axes or None
    colors     : list of color specs (one per ridge) or a single color string.
                 Defaults to cycling through ["cyan", "lime", "red", "yellow"].
    **kwargs   : forwarded to ax.plot for each ridge line

    Returns
    -------
    matplotlib.axes.Axes
    """
    _DEFAULT_COLORS = ["cyan", "lime", "red", "yellow"]

    ax = plot_sst(sst_result, ax=ax)

    kwargs.setdefault("linewidth", 1.5)

    for i, ridge in enumerate(ridges):
        times_np = _to_numpy(ridge.times)
        if colors is not None:
            # per-ridge list or single color
            c = colors[i] if isinstance(colors, (list, tuple)) else colors
        else:
            c = _DEFAULT_COLORS[i % len(_DEFAULT_COLORS)]
        ax.plot(times_np, ridge.freq_path, color=c, **kwargs)

    ax.set_title("SST with Ridges")
    return ax


def plot_components(components, times, ax=None, **kwargs):
    """
    Plot reconstructed component signals.

    Parameters
    ----------
    components : list[Component]
    times      : 1-D array-like of time values (seconds)
    ax         : matplotlib.axes.Axes or None
    **kwargs   : forwarded to ax.plot for each component

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()

    times_arr = np.asarray(times)
    labels = kwargs.pop("label", None)
    for i, comp in enumerate(components):
        lbl = labels[i] if isinstance(labels, (list, tuple)) else (
            labels if labels is not None else f"Component {i + 1}"
        )
        ax.plot(times_arr, comp.signal, label=lbl, **kwargs)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Reconstructed Components")
    ax.legend()
    return ax


def plot_stft(stft_result, ax=None, **kwargs):
    """
    Plot STFT magnitude as a time-frequency heatmap.

    Parameters
    ----------
    stft_result : STFTResult — V may be on CPU or GPU
    ax          : matplotlib.axes.Axes or None (creates new figure)
    **kwargs    : forwarded to ax.pcolormesh

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()

    V_mag = _to_numpy(stft_result.V.abs())   # (n_freqs, n_frames)
    times = _to_numpy(stft_result.times)
    freqs = _to_numpy(stft_result.freqs)

    kwargs.setdefault("shading", "auto")
    kwargs.setdefault("cmap", "inferno")
    ax.pcolormesh(times, freqs, V_mag, **kwargs)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("STFT Magnitude")
    return ax


def plot_stft_sst(stft_sst_result, ax=None, **kwargs):
    """
    Plot STFT-SST magnitude as a time-frequency heatmap.

    Parameters
    ----------
    stft_sst_result : STFTSSTResult — Tx may be on CPU or GPU
    ax              : matplotlib.axes.Axes or None (creates new figure)
    **kwargs        : forwarded to ax.pcolormesh

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()

    Tx_mag = _to_numpy(stft_sst_result.Tx.abs())  # (n_freqs, n_frames)
    times  = _to_numpy(stft_sst_result.times)
    freqs  = _to_numpy(stft_sst_result.freqs)

    kwargs.setdefault("shading", "auto")
    kwargs.setdefault("cmap", "inferno")
    ax.pcolormesh(times, freqs, Tx_mag, **kwargs)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("STFT-SST Magnitude")
    return ax
