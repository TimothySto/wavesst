"""
Interactive ipywidgets wrappers for wavesst plot functions.

Each ``iplot_*`` function calls its ``plot_*`` counterpart and then
adds controls (sliders, dropdowns, toggles) via ``ipywidgets.interact``.

Import is deferred — ipywidgets is an optional dependency.  Importing
this module without ipywidgets installed raises ``ImportError`` with a
clear message.

Usage (inside a Jupyter notebook)::

    from wavesst.viz.interactive import iplot_cwt
    iplot_cwt(cwt_result)
"""
from __future__ import annotations

try:
    import ipywidgets as widgets
    from IPython.display import display
except ImportError as exc:
    raise ImportError(
        "ipywidgets is required for interactive plots.\n"
        "Install with:  pip install 'wavesst[viz]'"
    ) from exc

import numpy as np
import matplotlib.pyplot as plt

from wavesst.viz.tf_plot import (
    _to_numpy,
    plot_cwt,
    plot_sst,
    plot_ridges,
    plot_components,
    plot_stft,
    plot_stft_sst,
)


# ---------------------------------------------------------------------------
# Shared control builders
# ---------------------------------------------------------------------------

def _freq_slider(freqs: np.ndarray):
    f_min, f_max = float(freqs.min()), float(freqs.max())
    step = (f_max - f_min) / 100
    return widgets.FloatRangeSlider(
        value=[f_min, f_max], min=f_min, max=f_max, step=step,
        description="Freq (Hz):", continuous_update=False,
        layout=widgets.Layout(width="500px"),
    )


def _time_slider(times: np.ndarray):
    t_min, t_max = float(times.min()), float(times.max())
    step = (t_max - t_min) / 100
    return widgets.FloatRangeSlider(
        value=[t_min, t_max], min=t_min, max=t_max, step=step,
        description="Time (s):", continuous_update=False,
        layout=widgets.Layout(width="500px"),
    )


def _cmap_dropdown():
    return widgets.Dropdown(
        options=["inferno", "viridis", "plasma", "magma", "hot", "gray"],
        value="inferno",
        description="Colormap:",
    )


def _log_toggle():
    return widgets.ToggleButton(
        value=False, description="Log scale", icon="check",
    )


# ---------------------------------------------------------------------------
# Interactive plot wrappers
# ---------------------------------------------------------------------------

def iplot_cwt(cwt_result):
    """Interactive CWT magnitude plot with freq/time zoom, colormap, and log toggle."""
    freqs = _to_numpy(cwt_result.freqs)
    times = _to_numpy(cwt_result.times)

    freq_slider = _freq_slider(freqs)
    time_slider = _time_slider(times)
    cmap_dd     = _cmap_dropdown()
    log_toggle  = _log_toggle()

    out = widgets.Output()

    def _update(freq_range, time_range, cmap, log_scale):
        with out:
            out.clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(10, 4))
            W_mag = _to_numpy(cwt_result.W.abs())
            if log_scale:
                W_mag = np.log1p(W_mag)
            ax.pcolormesh(times, freqs, W_mag, shading="auto", cmap=cmap)
            ax.set_xlim(time_range)
            ax.set_ylim(freq_range)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Frequency (Hz)")
            ax.set_title("CWT Magnitude" + (" [log]" if log_scale else ""))
            plt.tight_layout()
            plt.show()

    ui = widgets.VBox([
        widgets.HBox([freq_slider, time_slider]),
        widgets.HBox([cmap_dd, log_toggle]),
    ])
    widgets.interactive_output(_update, {
        "freq_range": freq_slider,
        "time_range": time_slider,
        "cmap": cmap_dd,
        "log_scale": log_toggle,
    })
    display(ui, out)
    _update(freq_slider.value, time_slider.value, cmap_dd.value, log_toggle.value)


def iplot_sst(sst_result):
    """Interactive SST (or STFT-SST) magnitude plot."""
    freqs = _to_numpy(sst_result.freqs)
    times = _to_numpy(sst_result.times)

    freq_slider = _freq_slider(freqs)
    time_slider = _time_slider(times)
    cmap_dd     = _cmap_dropdown()
    log_toggle  = _log_toggle()

    out = widgets.Output()

    def _update(freq_range, time_range, cmap, log_scale):
        with out:
            out.clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(10, 4))
            Tx_mag = _to_numpy(sst_result.Tx.abs())
            if log_scale:
                Tx_mag = np.log1p(Tx_mag)
            ax.pcolormesh(times, freqs, Tx_mag, shading="auto", cmap=cmap)
            ax.set_xlim(time_range)
            ax.set_ylim(freq_range)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Frequency (Hz)")
            ax.set_title("SST Magnitude" + (" [log]" if log_scale else ""))
            plt.tight_layout()
            plt.show()

    ui = widgets.VBox([
        widgets.HBox([freq_slider, time_slider]),
        widgets.HBox([cmap_dd, log_toggle]),
    ])
    widgets.interactive_output(_update, {
        "freq_range": freq_slider,
        "time_range": time_slider,
        "cmap": cmap_dd,
        "log_scale": log_toggle,
    })
    display(ui, out)
    _update(freq_slider.value, time_slider.value, cmap_dd.value, log_toggle.value)


def iplot_ridges(sst_result, ridges):
    """Interactive SST + ridge overlay plot with freq/time zoom and component toggles."""
    _DEFAULT_COLORS = ["cyan", "lime", "red", "yellow"]
    freqs = _to_numpy(sst_result.freqs)
    times = _to_numpy(sst_result.times)

    freq_slider = _freq_slider(freqs)
    time_slider = _time_slider(times)
    cmap_dd     = _cmap_dropdown()
    log_toggle  = _log_toggle()

    # One toggle per ridge
    ridge_toggles = [
        widgets.ToggleButton(
            value=True,
            description=f"Ridge {i + 1}",
            button_style="info",
            layout=widgets.Layout(width="100px"),
        )
        for i in range(len(ridges))
    ]

    out = widgets.Output()

    def _update(freq_range, time_range, cmap, log_scale, **ridge_visible):
        with out:
            out.clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(10, 4))
            Tx_mag = _to_numpy(sst_result.Tx.abs())
            if log_scale:
                Tx_mag = np.log1p(Tx_mag)
            ax.pcolormesh(times, freqs, Tx_mag, shading="auto", cmap=cmap)
            for i, ridge in enumerate(ridges):
                if ridge_visible.get(f"r{i}", True):
                    t_np = _to_numpy(ridge.times)
                    ax.plot(t_np, ridge.freq_path,
                            color=_DEFAULT_COLORS[i % len(_DEFAULT_COLORS)],
                            linewidth=1.5, label=f"Ridge {i + 1}")
            ax.set_xlim(time_range)
            ax.set_ylim(freq_range)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Frequency (Hz)")
            ax.set_title("SST with Ridges" + (" [log]" if log_scale else ""))
            ax.legend(loc="upper right", fontsize=8)
            plt.tight_layout()
            plt.show()

    control_dict = {
        "freq_range": freq_slider,
        "time_range": time_slider,
        "cmap": cmap_dd,
        "log_scale": log_toggle,
    }
    for i, tog in enumerate(ridge_toggles):
        control_dict[f"r{i}"] = tog

    ui = widgets.VBox([
        widgets.HBox([freq_slider, time_slider]),
        widgets.HBox([cmap_dd, log_toggle]),
        widgets.HBox(ridge_toggles),
    ])
    widgets.interactive_output(_update, control_dict)
    display(ui, out)
    _update(freq_slider.value, time_slider.value, cmap_dd.value, log_toggle.value,
            **{f"r{i}": True for i in range(len(ridges))})


def iplot_stft(stft_result):
    """Interactive STFT magnitude plot."""
    freqs = _to_numpy(stft_result.freqs)
    times = _to_numpy(stft_result.times)

    freq_slider = _freq_slider(freqs)
    time_slider = _time_slider(times)
    cmap_dd     = _cmap_dropdown()
    log_toggle  = _log_toggle()

    out = widgets.Output()

    def _update(freq_range, time_range, cmap, log_scale):
        with out:
            out.clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(10, 4))
            V_mag = _to_numpy(stft_result.V.abs())
            if log_scale:
                V_mag = np.log1p(V_mag)
            ax.pcolormesh(times, freqs, V_mag, shading="auto", cmap=cmap)
            ax.set_xlim(time_range)
            ax.set_ylim(freq_range)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Frequency (Hz)")
            ax.set_title("STFT Magnitude" + (" [log]" if log_scale else ""))
            plt.tight_layout()
            plt.show()

    ui = widgets.VBox([
        widgets.HBox([freq_slider, time_slider]),
        widgets.HBox([cmap_dd, log_toggle]),
    ])
    widgets.interactive_output(_update, {
        "freq_range": freq_slider,
        "time_range": time_slider,
        "cmap": cmap_dd,
        "log_scale": log_toggle,
    })
    display(ui, out)
    _update(freq_slider.value, time_slider.value, cmap_dd.value, log_toggle.value)


def iplot_stft_sst(stft_sst_result):
    """Interactive STFT-SST magnitude plot."""
    freqs = _to_numpy(stft_sst_result.freqs)
    times = _to_numpy(stft_sst_result.times)

    freq_slider = _freq_slider(freqs)
    time_slider = _time_slider(times)
    cmap_dd     = _cmap_dropdown()
    log_toggle  = _log_toggle()

    out = widgets.Output()

    def _update(freq_range, time_range, cmap, log_scale):
        with out:
            out.clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(10, 4))
            Tx_mag = _to_numpy(stft_sst_result.Tx.abs())
            if log_scale:
                Tx_mag = np.log1p(Tx_mag)
            ax.pcolormesh(times, freqs, Tx_mag, shading="auto", cmap=cmap)
            ax.set_xlim(time_range)
            ax.set_ylim(freq_range)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Frequency (Hz)")
            ax.set_title("STFT-SST Magnitude" + (" [log]" if log_scale else ""))
            plt.tight_layout()
            plt.show()

    ui = widgets.VBox([
        widgets.HBox([freq_slider, time_slider]),
        widgets.HBox([cmap_dd, log_toggle]),
    ])
    widgets.interactive_output(_update, {
        "freq_range": freq_slider,
        "time_range": time_slider,
        "cmap": cmap_dd,
        "log_scale": log_toggle,
    })
    display(ui, out)
    _update(freq_slider.value, time_slider.value, cmap_dd.value, log_toggle.value)


def iplot_components(components, times):
    """Interactive reconstructed-components plot with per-component visibility toggles."""
    times_arr = np.asarray(times)
    t_min, t_max = float(times_arr.min()), float(times_arr.max())
    step = (t_max - t_min) / 100

    time_slider = widgets.FloatRangeSlider(
        value=[t_min, t_max], min=t_min, max=t_max, step=step,
        description="Time (s):", continuous_update=False,
        layout=widgets.Layout(width="500px"),
    )
    comp_toggles = [
        widgets.ToggleButton(
            value=True,
            description=f"Comp {i + 1}",
            button_style="info",
            layout=widgets.Layout(width="100px"),
        )
        for i in range(len(components))
    ]

    out = widgets.Output()

    def _update(time_range, **comp_visible):
        with out:
            out.clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(10, 3))
            for i, comp in enumerate(components):
                if comp_visible.get(f"c{i}", True):
                    ax.plot(times_arr, comp.signal, label=f"Component {i + 1}")
            ax.set_xlim(time_range)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")
            ax.set_title("Reconstructed Components")
            ax.legend(loc="upper right", fontsize=8)
            plt.tight_layout()
            plt.show()

    control_dict = {"time_range": time_slider}
    for i, tog in enumerate(comp_toggles):
        control_dict[f"c{i}"] = tog

    ui = widgets.VBox([
        time_slider,
        widgets.HBox(comp_toggles),
    ])
    widgets.interactive_output(_update, control_dict)
    display(ui, out)
    _update(time_slider.value, **{f"c{i}": True for i in range(len(components))})
