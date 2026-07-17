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
