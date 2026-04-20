"""
Validation: SST produces sharper TF representation than raw CWT.

Since ssqueezepy crashes on this Windows environment (fatal DLL error),
we validate mathematically:
1. SST energy is more concentrated than CWT energy for pure tones
2. SST peak frequency is closer to ground truth than CWT peak
3. SST peak frequency is within 5% of true frequency at multiple test points
"""
import numpy as np
import pytest
from wavesst.transforms.sst import sst
from wavesst.transforms.cwt import cwt
from wavesst.backends import get_backend

FS = 256.0
N = 1024   # longer signal for cleaner test


@pytest.fixture
def backend():
    return get_backend("numpy")


def _tone(freq_hz, n=N, fs=FS):
    t = np.arange(n) / fs
    return np.cos(2 * np.pi * freq_hz * t).astype(np.float64)


def test_sst_more_concentrated_than_cwt(backend):
    """For a pure tone, SST Renyi entropy < CWT Renyi entropy (sharpening guaranteed)."""
    f0 = 32.0
    x = _tone(f0)

    cwt_result = cwt(x, wavelet="morlet", scales="auto", fs=FS, backend=backend)
    sst_result = sst(x, wavelet="morlet", scales="auto", fs=FS, gamma="auto", backend=backend)

    cwt_energy = np.sum(np.abs(cwt_result.W) ** 2, axis=1)
    sst_energy = np.sum(np.abs(sst_result.Tx) ** 2, axis=1)

    def renyi_entropy(e, alpha=2):
        p = e / e.sum()
        p = p[p > 0]
        return (1.0 / (1.0 - alpha)) * np.log(np.sum(p ** alpha))

    assert renyi_entropy(sst_energy) < renyi_entropy(cwt_energy), (
        "SST Renyi entropy should be lower than CWT (more concentrated)"
    )


def test_sst_peak_closer_to_truth_than_cwt(backend):
    """SST peak frequency error <= CWT peak frequency error for a pure tone."""
    f0 = 45.0
    x = _tone(f0)

    cwt_result = cwt(x, wavelet="morlet", scales="auto", fs=FS, backend=backend)
    sst_result = sst(x, wavelet="morlet", scales="auto", fs=FS, gamma="auto", backend=backend)

    cwt_energy = np.sum(np.abs(cwt_result.W) ** 2, axis=1)
    cwt_peak_freq = cwt_result.freqs[np.argmax(cwt_energy)]

    sst_energy = np.sum(np.abs(sst_result.Tx) ** 2, axis=1)
    sst_peak_freq = sst_result.freqs[np.argmax(sst_energy)]

    cwt_err = abs(cwt_peak_freq - f0)
    sst_err = abs(sst_peak_freq - f0)

    assert sst_err <= cwt_err + 1.0, (   # +1 Hz tolerance for discretization
        f"SST error {sst_err:.2f} Hz should not exceed CWT error {cwt_err:.2f} Hz"
    )


def test_sst_absolute_peak_accuracy(backend):
    """SST peak frequency should be within 5% of true frequency."""
    for f0 in [16.0, 32.0, 64.0]:
        x = _tone(f0)
        result = sst(x, wavelet="morlet", scales="auto", fs=FS, gamma="auto", backend=backend)
        energy = np.sum(np.abs(result.Tx) ** 2, axis=1)
        peak_freq = result.freqs[np.argmax(energy)]
        rel_err = abs(peak_freq - f0) / f0
        assert rel_err < 0.05, f"f0={f0}: SST peak {peak_freq:.2f} Hz, rel err {rel_err:.3f}"
