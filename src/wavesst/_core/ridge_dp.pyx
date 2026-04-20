# cython: language_level=3, boundscheck=False, wraparound=False
import numpy as np
cimport numpy as cnp

cnp.import_array()


def find_ridge(
    cnp.ndarray[cnp.float64_t, ndim=2] energy,
    double penalty,
):
    """
    Dynamic programming ridge extraction on a 2-D energy array.

    Finds the path k(b) through time steps b=0..n_samples-1 that maximizes:
      sum_b [ energy[k(b), b] - penalty * |k(b) - k(b-1)| ]

    Parameters
    ----------
    energy : float64 array, shape (n_freqs, n_samples)
        |T_x(freq, time)| — the TF energy plane.
    penalty : float
        Cost per unit frequency-bin jump between adjacent time steps.

    Returns
    -------
    bin_path : int32 array, shape (n_samples,)
        Frequency bin index of the ridge at each time step.
    """
    cdef int n_freqs = energy.shape[0]
    cdef int n_samples = energy.shape[1]
    cdef int t, k, k_prev, best_k
    cdef double best_val, val

    # dp[k] = best accumulated score ending at freq bin k at current time
    cdef cnp.ndarray[cnp.float64_t, ndim=1] dp = np.empty(n_freqs, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] dp_new = np.empty(n_freqs, dtype=np.float64)
    # back[t, k] = predecessor bin at time t-1 for path ending at bin k at time t
    cdef cnp.ndarray[cnp.int32_t, ndim=2] back = np.empty((n_samples, n_freqs), dtype=np.int32)
    cdef cnp.ndarray[cnp.int32_t, ndim=1] bin_path = np.empty(n_samples, dtype=np.int32)

    # Initialize at t=0
    for k in range(n_freqs):
        dp[k] = energy[k, 0]

    # Forward pass
    for t in range(1, n_samples):
        for k in range(n_freqs):
            best_val = -1e300
            best_k = 0
            for k_prev in range(n_freqs):
                val = dp[k_prev] - penalty * abs(k - k_prev)
                if val > best_val:
                    best_val = val
                    best_k = k_prev
            dp_new[k] = best_val + energy[k, t]
            back[t, k] = best_k
        # swap
        for k in range(n_freqs):
            dp[k] = dp_new[k]

    # Find best final bin
    best_val = -1e300
    best_k = 0
    for k in range(n_freqs):
        if dp[k] > best_val:
            best_val = dp[k]
            best_k = k

    # Backtrack
    bin_path[n_samples - 1] = best_k
    for t in range(n_samples - 2, -1, -1):
        bin_path[t] = back[t + 1, bin_path[t + 1]]

    return bin_path
