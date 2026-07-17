from __future__ import annotations

import numpy as np


def make_noise(
    duration: float,
    fs: float,
    color: str = "white",
    amplitude: float = 1.0,
    seed: int | None = None,
) -> np.ndarray:
    """
    Generate coloured noise.

    Parameters
    ----------
    duration  : signal duration in seconds
    fs        : sample rate in Hz
    color     : 'white' (flat spectrum), 'pink' (1/√f), 'brown' (1/f),
                or 'impulsive' (sparse random spikes, ~1% non-zero)
    amplitude : RMS amplitude scaling factor applied after normalisation
    seed      : random seed for reproducibility

    Returns
    -------
    float32 ndarray of shape (N,) where N = int(duration * fs)
    """
    _VALID_COLORS = ("white", "pink", "brown", "impulsive")
    if color not in _VALID_COLORS:
        raise ValueError(
            f"Unknown color {color!r}. Use one of: {_VALID_COLORS}"
        )

    N = int(duration * fs)
    rng = np.random.default_rng(seed)

    if color == "white":
        x = rng.standard_normal(N).astype(np.float32)

    elif color == "pink":
        white = rng.standard_normal(N)
        f = np.fft.rfftfreq(N)
        f[0] = 1.0          # avoid divide-by-zero at DC
        spectrum = np.fft.rfft(white) / np.sqrt(f)
        spectrum[0] = 0.0   # zero DC
        x = np.fft.irfft(spectrum, n=N).astype(np.float32)

    elif color == "brown":
        white = rng.standard_normal(N)
        f = np.fft.rfftfreq(N)
        f[0] = 1.0
        spectrum = np.fft.rfft(white) / f
        spectrum[0] = 0.0
        x = np.fft.irfft(spectrum, n=N).astype(np.float32)

    else:  # impulsive
        x = np.zeros(N, dtype=np.float32)
        mask = rng.random(N) < 0.01
        n_spikes = int(mask.sum())
        if n_spikes > 0:
            x[mask] = rng.standard_normal(n_spikes).astype(np.float32)

    # Normalise to unit RMS (where RMS > 0), then scale
    rms = float(np.sqrt(np.mean(x ** 2)))
    if rms > 0.0:
        x = x / rms * amplitude

    return x.astype(np.float32)
