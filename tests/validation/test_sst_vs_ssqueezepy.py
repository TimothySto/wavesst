"""
Cross-validation of wavesst SST against ssqueezepy.

ssqueezepy 0.6.6 is importable after the MKL→OpenBLAS fix (Session 3).
Skipped automatically if ssqueezepy is not installed.
"""
import numpy as np
import pytest
import torch
import wavesst

ssqueezepy = pytest.importorskip("ssqueezepy", reason="ssqueezepy not installed")

from wavesst.transforms.sst import sst

FS = 256.0
N = 512


@pytest.fixture(scope="module")
def module_cfg():
    return wavesst.Config(device='cpu', dtype=torch.complex128)


def _tone(freq_hz, n=N, fs=FS):
    t = np.arange(n) / fs
    return np.cos(2 * np.pi * freq_hz * t).astype(np.float64)


def _ssq_peak_freq(x, fs):
    """Run ssqueezepy SST and return peak frequency from energy profile."""
    Tx, _, ssq_freqs, _ = ssqueezepy.ssq_cwt(x, wavelet='morlet', fs=fs)
    energy = np.sum(np.abs(Tx) ** 2, axis=1)
    return float(ssq_freqs[np.argmax(energy)])


def test_sst_peak_agrees_with_ssqueezepy(module_cfg):
    """wavesst SST peak frequency should agree with ssqueezepy within 10%."""
    f0 = 32.0
    x = _tone(f0)

    our_result = sst(x, wavelet="morlet", scales="auto", fs=FS, gamma="auto", cfg=module_cfg)
    our_energy = our_result.Tx.abs().pow(2).sum(dim=1).numpy()
    our_peak = our_result.freqs[np.argmax(our_energy)]

    ssq_peak = _ssq_peak_freq(x, FS)

    rel_diff = abs(our_peak - ssq_peak) / ssq_peak
    assert rel_diff < 0.10, (
        f"wavesst peak {our_peak:.2f} Hz vs ssqueezepy {ssq_peak:.2f} Hz "
        f"(rel diff {rel_diff:.3f} > 0.10)"
    )


def test_sst_more_concentrated_than_cwt_matches_ssqueezepy_direction(module_cfg):
    """Both wavesst and ssqueezepy SST should be more concentrated than CWT.

    Note: direct entropy comparison across libraries is invalid because grid sizes
    and normalizations differ. We verify both SST outputs individually sharpen their
    respective CWT by checking that SST energy is more peaked.
    """
    from wavesst.transforms.cwt import cwt as wavesst_cwt

    f0 = 40.0
    x = _tone(f0)

    # wavesst: SST more concentrated than CWT
    our_cwt = wavesst_cwt(x, wavelet="morlet", scales="auto", fs=FS, cfg=module_cfg)
    our_sst = sst(x, wavelet="morlet", scales="auto", fs=FS, gamma="auto", cfg=module_cfg)

    cwt_energy = our_cwt.W.abs().pow(2).sum(dim=1).numpy()
    sst_energy = our_sst.Tx.abs().pow(2).sum(dim=1).numpy()

    # Gini coefficient: higher = more concentrated (0=uniform, 1=single spike)
    def gini(e):
        e = np.sort(e)
        n = len(e)
        return (2 * np.sum((np.arange(1, n + 1) * e)) / (n * e.sum()) - (n + 1) / n)

    assert gini(sst_energy) >= gini(cwt_energy), (
        "wavesst SST should be more concentrated (higher Gini) than CWT"
    )

    # ssqueezepy SST: independently verify it concentrates energy
    Tx_ssq, Wx_ssq, ssq_freqs, _ = ssqueezepy.ssq_cwt(x, wavelet='morlet', fs=FS)
    ssq_sst_energy = np.sum(np.abs(Tx_ssq) ** 2, axis=1)
    ssq_cwt_energy = np.sum(np.abs(Wx_ssq) ** 2, axis=1)

    assert gini(ssq_sst_energy) >= gini(ssq_cwt_energy), (
        "ssqueezepy SST should also be more concentrated than its CWT"
    )
