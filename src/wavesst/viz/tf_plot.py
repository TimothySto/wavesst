from __future__ import annotations

import numpy as np


def plot_cwt(cwt_result, ax=None, **kwargs):
    """
    Plot CWT magnitude as a time-frequency heatmap.

    Parameters
    ----------
    cwt_result : CWTResult — W is torch.Tensor on CPU
    ax         : matplotlib.axes.Axes or None (creates new figure)
    **kwargs   : forwarded to ax.pcolormesh

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()

    W_mag = cwt_result.W.abs().numpy()   # (n_scales, N)
    times = cwt_result.times
    freqs = cwt_result.freqs

    kwargs.setdefault("shading", "auto")
    kwargs.setdefault("cmap", "inferno")
    ax.pcolormesh(times, freqs, W_mag, **kwargs)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("CWT Magnitude")
    return ax


def plot_sst(sst_result, ax=None, **kwargs):
    """
    Plot SST magnitude as a time-frequency heatmap.

    Parameters
    ----------
    sst_result : SSTResult — Tx is torch.Tensor on CPU
    ax         : matplotlib.axes.Axes or None
    **kwargs   : forwarded to ax.pcolormesh

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()

    Tx_mag = sst_result.Tx.abs().numpy()  # (n_freqs, N)
    times = sst_result.times
    freqs = sst_result.freqs

    kwargs.setdefault("shading", "auto")
    kwargs.setdefault("cmap", "inferno")
    ax.pcolormesh(times, freqs, Tx_mag, **kwargs)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("SST Magnitude")
    return ax


def plot_ridges(sst_result, ridges, ax=None, **kwargs):
    """
    Plot SST magnitude with ridge paths overlaid.

    Parameters
    ----------
    sst_result : SSTResult
    ridges     : list[Ridge]
    ax         : matplotlib.axes.Axes or None
    **kwargs   : forwarded to ax.plot for each ridge line

    Returns
    -------
    matplotlib.axes.Axes
    """
    ax = plot_sst(sst_result, ax=ax)
    times = sst_result.times

    kwargs.setdefault("color", "cyan")
    kwargs.setdefault("linewidth", 1.5)
    for ridge in ridges:
        ax.plot(times, ridge.freq_path, **kwargs)

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
    for i, comp in enumerate(components):
        label = kwargs.pop("label", f"Component {i + 1}")
        ax.plot(times_arr, comp.signal, label=label, **kwargs)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Reconstructed Components")
    ax.legend()
    return ax
