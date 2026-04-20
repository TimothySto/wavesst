# cython: language_level=3, boundscheck=False, wraparound=False
import numpy as np
cimport numpy as cnp
from libc.math cimport exp

cnp.import_array()

# Normalization constant: pi^(-1/4) so that the L2 norm of psi is 1
DEF PI_NEG_QUARTER = 0.7511255444649425


def morlet_freq_response(
    cnp.ndarray[cnp.float64_t, ndim=1] omega,
    double scale,
    double w0=6.0,
):
    """
    Frequency-domain Morlet wavelet filter bank at a single scale.

    Computes  psi_hat(a*omega) = pi^(-1/4) * exp(-0.5 * (a*omega - w0)^2)
    for omega > 0, zero otherwise (analytic wavelet — one-sided spectrum).

    Parameters
    ----------
    omega : 1-D float64 array
        Angular frequency axis (rad/sample or rad/s).
    scale : float
        Wavelet scale a > 0.
    w0 : float
        Center frequency of the Morlet wavelet (default 6.0).

    Returns
    -------
    result : 1-D float64 array, same length as omega
    """
    cdef int M = omega.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1] result = np.zeros(M, dtype=np.float64)
    cdef double scaled_w, diff
    cdef int i

    for i in range(M):
        scaled_w = scale * omega[i]
        if scaled_w > 0.0:
            diff = scaled_w - w0
            result[i] = PI_NEG_QUARTER * exp(-0.5 * diff * diff)

    return result


def deriv_morlet_freq_response(
    cnp.ndarray[cnp.float64_t, ndim=1] omega,
    double scale,
    double w0=6.0,
):
    """
    Frequency-domain derivative Morlet filter: iω · ψ̂(a·ω).

    Since ψ̂ is real and the factor iω is purely imaginary, the result has:
      real part = 0
      imaginary part = ω · ψ̂(a·ω)

    Returns (real_part, imag_part) as separate float64 arrays.
    Used by SST to compute ∂_b W_x without numerical differentiation.

    Parameters
    ----------
    omega : 1-D float64 array — angular frequency axis (rad/s)
    scale : float — wavelet scale a > 0 (seconds)
    w0    : float — Morlet center frequency (default 6.0)

    Returns
    -------
    (real_part, imag_part) : tuple of two 1-D float64 arrays, same length as omega
    """
    cdef int M = omega.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1] real_part = np.zeros(M, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] imag_part = np.zeros(M, dtype=np.float64)
    cdef double scaled_w, diff
    cdef int i

    for i in range(M):
        scaled_w = scale * omega[i]
        if scaled_w > 0.0:
            diff = scaled_w - w0
            imag_part[i] = omega[i] * PI_NEG_QUARTER * exp(-0.5 * diff * diff)

    return real_part, imag_part
