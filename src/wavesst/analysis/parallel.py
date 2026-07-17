from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from wavesst.analysis.ridge import Ridge


def _band_worker(args):
    """Module-level worker function (required for ProcessPoolExecutor pickling)."""
    sub_energy, freq_offset, freqs_sub, times, penalty = args
    from wavesst._core.ridge_dp import find_ridge
    bin_path_local = find_ridge(sub_energy, penalty)          # (n_time,) int32
    bin_path_global = (bin_path_local + freq_offset).astype(np.int32)
    freq_path = freqs_sub[bin_path_local]                      # Hz values for this sub-band
    n_time = sub_energy.shape[1]
    energy_path = sub_energy[bin_path_local, np.arange(n_time)]
    ridge_energy = float(np.sum(energy_path))
    return Ridge(
        freq_path=freq_path,
        bin_path=bin_path_global,
        energy=ridge_energy,
        times=times,
        energy_path=energy_path,
    )


def extract_ridges_parallel(
    sst_result,
    n: int,
    penalty: float = 1.0,
    n_jobs: int = -1,
) -> list[Ridge]:
    """
    Extract n dominant ridges by frequency-band decomposition, one worker per band.

    The frequency axis is divided into n equal sub-bands. Each worker runs the
    ridge DP on its sub-band independently (no cross-band masking/peeling).
    Results are sorted by ascending median frequency.

    Parameters
    ----------
    sst_result : any result type with .Tx and .freqs/.times
    n          : number of bands and ridges
    penalty    : frequency-jump penalty for the DP
    n_jobs     : worker count (-1 = min(n, cpu_count))
    """
    import torch

    # --- Unpack result (same duck-typing as extract_ridges) ---
    Tx = sst_result.Tx
    if isinstance(Tx, torch.Tensor):
        Tx = Tx.cpu().numpy()
    energy = np.abs(Tx).astype(np.float64)       # (n_freqs, n_time)

    freqs = sst_result.freqs
    if isinstance(freqs, torch.Tensor):
        freqs = freqs.cpu().numpy()

    times = sst_result.times
    if isinstance(times, torch.Tensor):
        times = times.cpu().numpy()

    n_freqs, n_time = energy.shape
    workers = min(n, os.cpu_count() or 1) if n_jobs == -1 else n_jobs

    # --- Divide frequency axis into n equal sub-bands ---
    band_edges = np.linspace(0, n_freqs, n + 1, dtype=int)
    args_list = []
    for i in range(n):
        lo, hi = int(band_edges[i]), int(band_edges[i + 1])
        args_list.append((
            energy[lo:hi, :],       # sub-band energy
            lo,                     # freq bin offset for global bin_path
            freqs[lo:hi],           # Hz values for this sub-band
            times,
            penalty,
        ))

    # --- Run workers ---
    if workers == 1 or n == 1:
        ridges = [_band_worker(a) for a in args_list]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            ridges = list(pool.map(_band_worker, args_list))

    # --- Sort by ascending median frequency ---
    ridges.sort(key=lambda r: float(np.median(r.freq_path)))
    return ridges
