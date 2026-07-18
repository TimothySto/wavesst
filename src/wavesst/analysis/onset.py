from __future__ import annotations

from typing import NamedTuple

import numpy as np

from wavesst.analysis.ridge import Ridge


class OnsetResult(NamedTuple):
    t_start: float   # seconds -- first sample where energy >= threshold * max_energy
    t_end:   float   # seconds -- last  sample where energy >= threshold * max_energy


def detect_onsets(ridge: Ridge, threshold: float = 0.1) -> OnsetResult:
    """
    Detect the active region of a ridge by energy thresholding.

    Parameters
    ----------
    ridge     : Ridge with a populated energy_path field
    threshold : fraction of peak ridge energy; bins below this are considered
                inactive (default 0.1 = 10% of peak)

    Returns
    -------
    OnsetResult(t_start, t_end) in seconds.
    Returns (0.0, 0.0) when the ridge has zero energy everywhere.
    """
    max_e = float(ridge.energy_path.max())
    if max_e == 0.0:
        return OnsetResult(t_start=0.0, t_end=0.0)

    active = ridge.energy_path >= threshold * max_e
    indices = np.where(active)[0]
    if indices.size == 0:
        return OnsetResult(t_start=0.0, t_end=0.0)

    return OnsetResult(
        t_start=float(ridge.times[indices[0]]),
        t_end=float(ridge.times[indices[-1]]),
    )


def detect_onset_segments(ridge: Ridge, threshold: float = 0.1) -> list[OnsetResult]:
    """
    Detect all contiguous active segments in a ridge's energy profile.

    Each segment where energy >= threshold * max_energy becomes one OnsetResult.
    Returns an empty list when the ridge has zero energy everywhere.

    Parameters
    ----------
    ridge     : Ridge with a populated energy_path field
    threshold : fraction of peak ridge energy below which samples are inactive

    Returns
    -------
    list of OnsetResult, one per contiguous active segment, in chronological order.
    """
    max_e = float(ridge.energy_path.max())
    if max_e == 0.0:
        return []

    active = ridge.energy_path >= threshold * max_e  # (n_time,) bool

    # Find transitions: pad with False on both ends to catch edge segments
    padded = np.concatenate(([False], active, [False]))
    diff = np.diff(padded.astype(np.int8))
    starts = np.where(diff == 1)[0]   # indices where active begins
    ends   = np.where(diff == -1)[0]  # indices where active ends (exclusive)

    return [
        OnsetResult(
            t_start=float(ridge.times[s]),
            t_end=float(ridge.times[e - 1]),
        )
        for s, e in zip(starts, ends)
    ]
