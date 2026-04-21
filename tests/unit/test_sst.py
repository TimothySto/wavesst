import numpy as np
import pytest
import torch
from wavesst.transforms.sst import sst, SSTResult

FS = 256.0
N = 512


# cfg fixture provided by conftest.py


@pytest.fixture
def signal():
    rng = np.random.default_rng(0)
    return rng.standard_normal(N).astype(np.float64)


def test_sst_returns_sst_result(signal, cfg):
    result = sst(signal, wavelet="morlet", scales="auto", fs=FS, cfg=cfg)
    assert isinstance(result, SSTResult)


def test_sst_tx_shape(signal, cfg):
    result = sst(signal, wavelet="morlet", scales="auto", fs=FS, cfg=cfg)
    n_freqs, n_samples = result.Tx.shape
    assert n_samples == N
    assert n_freqs > 0


def test_sst_tx_is_tensor(signal, cfg):
    result = sst(signal, wavelet="morlet", scales="auto", fs=FS, cfg=cfg)
    assert isinstance(result.Tx, torch.Tensor)


def test_sst_tx_is_complex(signal, cfg):
    result = sst(signal, wavelet="morlet", scales="auto", fs=FS, cfg=cfg)
    assert result.Tx.is_complex()


def test_sst_freqs_length(signal, cfg):
    result = sst(signal, wavelet="morlet", scales="auto", fs=FS, cfg=cfg)
    assert len(result.freqs) == result.Tx.shape[0]


def test_sst_freqs_uniformly_spaced(signal, cfg):
    """SST output frequency grid must be uniform (unlike CWT scale grid)."""
    result = sst(signal, wavelet="morlet", scales="auto", fs=FS, cfg=cfg)
    diffs = np.diff(result.freqs)
    np.testing.assert_allclose(diffs, diffs[0], rtol=1e-6)


def test_sst_times_length(signal, cfg):
    result = sst(signal, wavelet="morlet", scales="auto", fs=FS, cfg=cfg)
    assert len(result.times) == N


def test_sst_wx_is_cwt_result(signal, cfg):
    from wavesst.transforms.cwt import CWTResult
    result = sst(signal, wavelet="morlet", scales="auto", fs=FS, cfg=cfg)
    assert isinstance(result.Wx, CWTResult)


def test_sst_gamma_none_no_thresholding(signal, cfg):
    """With gamma=None, all CWT coefficients participate — Tx should be non-zero."""
    result = sst(signal, wavelet="morlet", scales="auto", fs=FS, gamma=None, cfg=cfg)
    assert result.Tx.abs().gt(0).any().item()


def test_sst_gamma_auto(signal, cfg):
    result = sst(signal, wavelet="morlet", scales="auto", fs=FS, gamma="auto", cfg=cfg)
    assert isinstance(result, SSTResult)


def test_sst_gamma_float(signal, cfg):
    result = sst(signal, wavelet="morlet", scales="auto", fs=FS, gamma=0.01, cfg=cfg)
    assert isinstance(result, SSTResult)


def test_sst_gamma_callable(signal, cfg):
    result = sst(signal, wavelet="morlet", scales="auto", fs=FS,
                 gamma=lambda Wx: 0.05, cfg=cfg)
    assert isinstance(result, SSTResult)


def test_sst_gamma_universal(signal, cfg):
    result = sst(signal, wavelet="morlet", scales="auto", fs=FS,
                 gamma="universal", cfg=cfg)
    assert isinstance(result, SSTResult)


def test_sst_high_gamma_produces_sparse_tx(signal, cfg):
    """Very high threshold → most bins zero."""
    result = sst(signal, wavelet="morlet", scales="auto", fs=FS, gamma=1e10, cfg=cfg)
    nonzero_frac = result.Tx.abs().gt(0).float().mean().item()
    assert nonzero_frac < 0.01
