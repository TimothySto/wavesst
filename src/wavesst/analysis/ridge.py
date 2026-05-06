from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from wavesst.transforms.sst import SSTResult
from wavesst._core.ridge_dp import find_ridge


@dataclass
class Ridge:
    freq_path: np.ndarray   # float64, shape (n_time,) -- Hz at each time step
    bin_path: np.ndarray    # int32,   shape (n_time,) -- frequency bin index
    energy: float           # total energy along this ridge
    times: np.ndarray       # float64, shape (n_time,) -- time values in seconds


def extract_ridges(
    sst_result,
    n: int = 1,
    penalty: float = 1.0,
) -> list[Ridge]:
    """
    Extract n ridges from a synchrosqueezed TF plane using dynamic programming.

    Accepts either a SSTResult or a STFTSSTResult (duck-typed: any object with
    .Tx, .freqs, and .times attributes).

    Each ridge is found on the residual energy (previous ridges zeroed out)
    to allow extraction of multiple non-overlapping components.

    Parameters
    ----------
    sst_result : SSTResult or STFTSSTResult
    n          : number of ridges to extract
    penalty    : smoothness weight — higher → smoother ridge (less freq jumping)

    Returns
    -------
    list of Ridge, length n, ordered by decreasing energy.
    Ridge.times holds the time axis of the TF plane (frame centres for STFT-SST,
    sample times for CWT-SST).
    """
    # Support both SSTResult (numpy freqs/times) and STFTSSTResult (torch freqs/times)
    Tx = sst_result.Tx
    if isinstance(Tx, torch.Tensor):
        Tx = Tx.cpu()

    energy = Tx.abs().double().numpy()   # (n_freqs, n_time)

    freqs = sst_result.freqs
    if isinstance(freqs, torch.Tensor):
        freqs = freqs.cpu().numpy()

    times = sst_result.times
    if isinstance(times, torch.Tensor):
        times = times.cpu().numpy()

    residual = energy.copy()
    ridges: list[Ridge] = []
    n_time = energy.shape[1]

    for _ in range(n):
        bin_path = find_ridge(residual, penalty)                          # (n_time,) int32
        freq_path = freqs[bin_path]                                       # (n_time,) Hz
        ridge_energy = float(np.sum(energy[bin_path, np.arange(n_time)]))

        ridges.append(Ridge(
            freq_path=freq_path,
            bin_path=bin_path.astype(np.int32),
            energy=ridge_energy,
            times=times,
        ))

        # Zero out this ridge in the residual so the next call finds a different path
        for t in range(n_time):
            residual[bin_path[t], t] = 0.0

    return ridges
