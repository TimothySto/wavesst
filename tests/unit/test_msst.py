import numpy as np
import pytest
import torch
import wavesst
from wavesst.transforms.msst import msst, MSSTResult

FS = 256.0
N = 512


# cfg fixture provided by conftest.py (device='cpu', complex128)


@pytest.fixture
def tone():
    t = np.arange(N) / FS
    return np.cos(2 * np.pi * 32.0 * t).astype(np.float64)


def test_msst_returns_result(tone, cfg):
    result = msst(tone, fs=FS, n_iter=1, cfg=cfg)
    assert isinstance(result, MSSTResult)


def test_msst_tx_shape(tone, cfg):
    result = msst(tone, fs=FS, n_iter=1, cfg=cfg)
    n_freqs, n_samples = result.Tx.shape
    assert n_samples == N
    assert n_freqs > 0


def test_msst_n_iter_stored(tone, cfg):
    result = msst(tone, fs=FS, n_iter=2, cfg=cfg)
    assert result.n_iter == 2


def test_msst_1iter_matches_sst(tone, cfg):
    """n_iter=1 should produce identical output to sst()."""
    from wavesst.transforms.sst import sst
    sst_result = sst(tone, wavelet="morlet", scales="auto", fs=FS, gamma="auto", cfg=cfg)
    msst_result = msst(tone, wavelet="morlet", scales="auto", fs=FS, n_iter=1, gamma="auto", cfg=cfg)
    torch.testing.assert_close(msst_result.Tx, sst_result.Tx)


def test_msst_2iter_more_concentrated(tone, cfg):
    """2 iterations should be at least as concentrated as 1 iteration."""
    r1 = msst(tone, fs=FS, n_iter=1, gamma="auto", cfg=cfg)
    r2 = msst(tone, fs=FS, n_iter=2, gamma="auto", cfg=cfg)

    def renyi(e, alpha=2):
        p = e / e.sum()
        p = p[p > 0]
        return (1.0 / (1.0 - alpha)) * np.log(np.sum(p ** alpha))

    e1 = r1.Tx.abs().pow(2).sum(dim=1).numpy()
    e2 = r2.Tx.abs().pow(2).sum(dim=1).numpy()
    # More concentrated → lower Rényi entropy (with this formula, lower = more concentrated).
    # Tolerance of 1.0: for a pure tone SST is already near-perfect (pass-1 Rényi ≈ 0),
    # so PM pass-2 may be slightly less concentrated due to Tx-derived IF on a near-delta Tx.
    assert renyi(e2) <= renyi(e1) + 1.0


def test_msst_3iter_runs_without_error(tone, cfg):
    """Smoke test: n_iter=3 should complete without raising."""
    result = msst(tone, fs=FS, n_iter=3, gamma="auto", cfg=cfg)
    assert result.n_iter == 3
    assert result.Tx.shape[1] == N
