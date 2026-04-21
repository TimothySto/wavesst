from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from wavesst.transforms.sst import SSTResult
from wavesst._core.ridge_dp import find_ridge


@dataclass
class Ridge:
    freq_path: np.ndarray   # float64, shape (n_samples,) -- Hz at each time step
    bin_path: np.ndarray    # int32,   shape (n_samples,) -- frequency bin index
    energy: float           # total energy along this ridge


def extract_ridges(
    sst_result: SSTResult,
    n: int = 1,
    penalty: float = 1.0,
) -> list[Ridge]:
    """
    Extract n ridges from the SST TF plane using dynamic programming.

    Each ridge is found on the residual energy (previous ridges zeroed out)
    to allow extraction of multiple non-overlapping components.

    Parameters
    ----------
    sst_result : SSTResult from wavesst.transforms.sst.sst()
    n          : number of ridges to extract
    penalty    : smoothness weight -- higher -> smoother ridge (less freq jumping)

    Returns
    -------
    list of Ridge, length n, ordered by decreasing energy
    """
    energy = sst_result.Tx.abs().double().numpy()  # (n_freqs, n_samples)
    freqs = sst_result.freqs                             # (n_freqs,)
    residual = energy.copy()
    ridges: list[Ridge] = []

    for _ in range(n):
        bin_path = find_ridge(residual, penalty)                      # (n_samples,) int32
        freq_path = freqs[bin_path]                                   # (n_samples,) Hz
        ridge_energy = float(np.sum(energy[bin_path, np.arange(energy.shape[1])]))

        ridges.append(Ridge(
            freq_path=freq_path,
            bin_path=bin_path.astype(np.int32),
            energy=ridge_energy,
        ))

        # Zero out this ridge in the residual so the next call finds a different path
        for t in range(residual.shape[1]):
            residual[bin_path[t], t] = 0.0

    return ridges
