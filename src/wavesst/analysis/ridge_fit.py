from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wavesst.analysis.ridge import Ridge
from wavesst.analysis.onset import OnsetResult, detect_onset_segments


@dataclass
class RidgeFit:
    order:       int
    coeffs:      np.ndarray   # shape (order+1,), highest-power-first (np.polyfit convention)
    fitted_freq: np.ndarray   # Hz at each time step along the full ridge, shape (n_time,)
    residuals:   np.ndarray   # freq_path − fitted_freq, shape (n_time,)
    rmse:        float        # RMS of residuals within the fitting window (t_start to t_end)


@dataclass
class SegmentFit:
    onset: OnsetResult   # t_start and t_end of this active segment
    fit:   RidgeFit      # polynomial fit restricted to this segment's time window


def fit_ridge(
    ridge: Ridge,
    order: int = 1,
    t_start: float | None = None,
    t_end: float | None = None,
) -> RidgeFit:
    """
    Fit a polynomial of given order to a ridge's instantaneous frequency.

    Parameters
    ----------
    ridge   : Ridge with freq_path and times fields
    order   : polynomial degree (1=linear, 2=quadratic, ...)
    t_start : restrict the fit to samples at or after this time (seconds)
    t_end   : restrict the fit to samples at or before this time (seconds)

    Returns
    -------
    RidgeFit with coeffs, fitted_freq (over full ridge), residuals, rmse.
    fitted_freq is always shape (n_time,) — the polynomial is evaluated over
    the full time axis even when the fit was computed on a sub-window.
    """
    times = ridge.times
    freq_path = ridge.freq_path

    # Select the window for fitting
    mask = np.ones(len(times), dtype=bool)
    if t_start is not None:
        mask &= times >= t_start
    if t_end is not None:
        mask &= times <= t_end

    fit_times = times[mask]
    fit_freqs = freq_path[mask]

    if len(fit_times) < order + 1:
        raise ValueError(
            f"fit window contains {len(fit_times)} sample(s) but "
            f"order={order} requires at least {order + 1}"
        )

    coeffs = np.polyfit(fit_times, fit_freqs, deg=order)

    # Evaluate over the FULL ridge time axis
    fitted_freq = np.polyval(coeffs, times)
    residuals = freq_path - fitted_freq
    rmse = float(np.sqrt(np.mean(residuals[mask] ** 2)))

    return RidgeFit(
        order=order,
        coeffs=coeffs,
        fitted_freq=fitted_freq,
        residuals=residuals,
        rmse=rmse,
    )


def fit_ridge_segments(
    ridge: Ridge,
    threshold: float = 0.1,
    order: int = 1,
) -> list[SegmentFit]:
    """
    Detect active segments by energy thresholding and fit a polynomial to each.

    For a transient signal with multiple pulses, returns one SegmentFit per pulse.
    Returns an empty list when the ridge has zero energy everywhere.

    Parameters
    ----------
    ridge     : Ridge with energy_path, freq_path, and times fields
    threshold : energy threshold fraction for onset detection (see detect_onset_segments)
    order     : polynomial degree for the frequency fit within each segment

    Returns
    -------
    list of SegmentFit, one per active segment, in chronological order.
    """
    segments = detect_onset_segments(ridge, threshold=threshold)
    return [
        SegmentFit(
            onset=seg,
            fit=fit_ridge(ridge, order=order, t_start=seg.t_start, t_end=seg.t_end),
        )
        for seg in segments
    ]
