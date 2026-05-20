from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from wavesst.transforms.sst import SSTResult
from wavesst.analysis.ridge import Ridge
from wavesst._core.filters import morlet_freq_response


def _admissibility_constant(w0: float = 6.0, n_pts: int = 100_000) -> float:
    """Numerically compute Css = integral_0^inf psi_hat(omega) / omega d_omega."""
    omega = np.linspace(1e-6, 30.0, n_pts)
    dw = omega[1] - omega[0]
    psi_hat = morlet_freq_response(omega, scale=1.0, w0=w0)
    return float(np.sum(psi_hat / omega) * dw)


_C_PSI = _admissibility_constant()   # computed once at import time


@dataclass
class Component:
    signal: np.ndarray      # float64, shape (n_samples,) -- reconstructed time-domain signal
    amplitude: np.ndarray   # float64, shape (n_samples,) -- instantaneous amplitude envelope
    phase: np.ndarray       # float64, shape (n_samples,) -- instantaneous phase (radians)
    ridge: Ridge            # the ridge this component was extracted along


def _reconstruct_cwt_sst(
    sst_result: SSTResult,
    ridges: list[Ridge],
    bandwidth: float | None,
) -> list[Component]:
    """
    Inverse CWT-SST formula:
      x_k(t) = (2 / Css) * Re[ sum_{f in band_k(t)} T_x(f, t) ]

    No df factor — the log-scale weight log(2)/nv is already absorbed into T_x.
    """
    import torch
    Tx = sst_result.Tx
    if isinstance(Tx, torch.Tensor):
        Tx = Tx.cpu().numpy()
    freqs = sst_result.freqs        # (n_freqs,) Hz, uniform ascending
    df = freqs[1] - freqs[0]
    n_freqs, n_samples = Tx.shape

    if bandwidth is None:
        bandwidth = df * 3.0        # ±3 bins around ridge

    components: list[Component] = []

    for ridge in ridges:
        analytic = np.zeros(n_samples, dtype=np.complex128)

        for t in range(n_samples):
            f_ridge = ridge.freq_path[t]
            mask = (freqs >= f_ridge - bandwidth) & (freqs <= f_ridge + bandwidth)
            analytic[t] = np.sum(Tx[mask, t])

        analytic *= (2.0 / _C_PSI)

        components.append(Component(
            signal=np.real(analytic),
            amplitude=np.abs(analytic),
            phase=np.angle(analytic),
            ridge=ridge,
        ))

    return components


def _reconstruct_stft_sst(
    sst_result,
    ridges: list[Ridge],
    bandwidth: float | None,
    fs: float,
) -> list[Component]:
    """
    Inverse STFT-SST via overlap-add (OLA) synthesis at the ridge frequency.

    Background
    ----------
    Summing T_x(η,τ) over a frequency band gives ~0 for a pure tone because the
    Hann window's leakage at adjacent bins exactly cancels the main lobe
    (Poisson-sum cancellation).  The correct reconstruction uses V — the
    original STFT output — at the ridge frequency, synthesised via OLA:

        x_k[n] = (4/M) · Re[Σ_m V[k_m, m] · e^{i2πk_m·n/M} · g[n-n_m]]
                 ──────────────────────────────────────────────────────────
                        Σ_m g[n-n_m]

    Derivation of the (4/M) prefactor
    ----------------------------------
    For a pure tone A·cos(2πf₀t) exactly on bin k₀:
      V[k₀, m] = A/2 · G₀ · e^{iφ_m}   where G₀ = Σg[n], φ_m = 2πk₀·n_m/M
    Substituting into the OLA sum:
      Σ_m (4/M)·Re[V·e^{i2πk₀n/M}·g] = Σ_m (4/M)·(A/2·G₀)·cos(2πf₀t)·g[n-n_m]
                                       = (2A·G₀/M)·cos(2πf₀t)·Σ_m g[n-n_m]
    Dividing by norm = Σ_m g[n-n_m]:
      x_k[n] = (2·G₀/M)·A·cos(2πf₀t) = A·cos(2πf₀t)   [since G₀ = M/2 for Hann]

    Parameters
    ----------
    sst_result : STFTSSTResult
    ridges     : list of Ridge (freq_path at frame times, n_frames points)
    bandwidth  : unused (kept for API compatibility)
    fs         : sample rate in Hz
    """
    import torch

    # Use the original STFT output V (NOT T_x) to avoid Poisson-sum cancellation
    V = sst_result.Vx.V
    if isinstance(V, torch.Tensor):
        V = V.cpu().numpy()             # (n_freqs, n_frames) complex

    freqs = sst_result.freqs
    if isinstance(freqs, torch.Tensor):
        freqs = freqs.cpu().numpy()     # (n_freqs,) Hz

    win = sst_result.Vx.window
    if isinstance(win, torch.Tensor):
        win = win.cpu().numpy().astype(np.float64)   # (nperseg,)

    hop = int(sst_result.Vx.hop)
    nperseg = len(win)
    n_freqs, n_frames = V.shape
    N = sst_result.n_samples

    components: list[Component] = []

    for ridge in ridges:
        x_k   = np.zeros(N, dtype=np.float64)
        norm  = np.zeros(N, dtype=np.float64)

        for m in range(n_frames):
            # Closest STFT bin to the ridge frequency at this frame
            k_m = int(np.argmin(np.abs(freqs - ridge.freq_path[m])))

            n_start = m * hop
            n_end   = min(n_start + nperseg, N)
            length  = n_end - n_start
            local_n = np.arange(length, dtype=np.float64)

            # Global-phase reconstruction: e^{i2πk_m·n/M} where n is absolute
            phase = 2.0 * np.pi * k_m * (n_start + local_n) / nperseg
            contrib = (4.0 / nperseg) * np.real(
                complex(V[k_m, m]) * np.exp(1j * phase)
            ) * win[:length]

            x_k[n_start:n_end]  += contrib
            norm[n_start:n_end] += win[:length]

        norm = np.maximum(norm, 1e-8)
        x_k /= norm

        # Analytic envelope from V at ridge bins (interpolated to sample times)
        frame_times = sst_result.times
        if isinstance(frame_times, torch.Tensor):
            frame_times = frame_times.cpu().numpy()
        sample_times = np.arange(N) / fs

        v_ridge = np.array([V[int(np.argmin(np.abs(freqs - ridge.freq_path[m]))), m]
                             for m in range(n_frames)], dtype=np.complex128)
        # Normalise: V[k,m] ≈ A/2·G₀ → analytic amplitude = A
        G0 = float(win.sum())
        v_ridge_norm = v_ridge * (2.0 / G0)    # → magnitude A

        v_real = np.interp(sample_times, frame_times, v_ridge_norm.real)
        v_imag = np.interp(sample_times, frame_times, v_ridge_norm.imag)
        analytic = v_real + 1j * v_imag

        components.append(Component(
            signal=x_k,
            amplitude=np.abs(analytic),
            phase=np.angle(analytic),
            ridge=ridge,
        ))

    return components


def reconstruct(
    sst_result,
    ridges: list[Ridge],
    bandwidth: float | None = None,
    fs: float = 1.0,
) -> list[Component]:
    """
    Reconstruct time-domain components from SST output.

    Accepts either a SSTResult (CWT-based SST) or a STFTSSTResult (STFT-based SST).

    CWT-SST inverse formula:
      x_k(t) = (2 / Css) · Re[ Σ_{f ∈ band_k(t)} T_x(f, t) ]
      where Css = ∫ ψ̂(ω)/ω dω  (SST admissibility constant ≈ 0.323)

    STFT-SST inverse formula (overlap-add on V):
      For each frame m, collect V[k_m, m] at the ridge bin k_m.
      OLA synthesis: x_k[n] = (4/M) · Re[ Σ_m V[k_m,m] · e^{i2πk_m·n/M}
                                           · g[n−n_m] ] / Σ_m g²[n−n_m]
      where M = nperseg. Uses V (raw STFT), not T_x, to avoid Poisson-sum
      cancellation from Hann-window leakage.

    Parameters
    ----------
    sst_result : SSTResult or STFTSSTResult
    ridges     : list of Ridge from extract_ridges()
    bandwidth  : Hz half-width of integration band (default: df × 3)
    fs         : sample rate in Hz — required for STFT-SST to build the
                 sample time axis; ignored for CWT-SST.

    Returns
    -------
    list of Component, same length as ridges
    """
    # Import here to avoid circular dependency at module load time
    from wavesst.transforms.stft_sst import STFTSSTResult

    if isinstance(sst_result, STFTSSTResult):
        return _reconstruct_stft_sst(sst_result, ridges, bandwidth, fs)
    else:
        return _reconstruct_cwt_sst(sst_result, ridges, bandwidth)
