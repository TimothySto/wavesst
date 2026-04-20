import numpy as np
import pytest
from wavesst.transforms.sst import sst, SSTResult
from wavesst.backends import get_backend

FS = 256.0
N = 512


@pytest.fixture
def backend():
    return get_backend("numpy")


@pytest.fixture
def signal():
    rng = np.random.default_rng(0)
    return rng.standard_normal(N).astype(np.float64)


def test_sst_returns_sst_result(signal, backend):
    result = sst(signal, wavelet="morlet", scales="auto", fs=FS, backend=backend)
    assert isinstance(result, SSTResult)


def test_sst_tx_shape(signal, backend):
    result = sst(signal, wavelet="morlet", scales="auto", fs=FS, backend=backend)
    n_freqs, n_samples = result.Tx.shape
    assert n_samples == N
    assert n_freqs > 0


def test_sst_tx_is_complex(signal, backend):
    result = sst(signal, wavelet="morlet", scales="auto", fs=FS, backend=backend)
    assert np.iscomplexobj(result.Tx)


def test_sst_freqs_length(signal, backend):
    result = sst(signal, wavelet="morlet", scales="auto", fs=FS, backend=backend)
    assert len(result.freqs) == result.Tx.shape[0]


def test_sst_freqs_uniformly_spaced(signal, backend):
    """SST output frequency grid must be uniform (unlike CWT scale grid)."""
    result = sst(signal, wavelet="morlet", scales="auto", fs=FS, backend=backend)
    diffs = np.diff(result.freqs)
    np.testing.assert_allclose(diffs, diffs[0], rtol=1e-6)


def test_sst_times_length(signal, backend):
    result = sst(signal, wavelet="morlet", scales="auto", fs=FS, backend=backend)
    assert len(result.times) == N


def test_sst_wx_is_cwt_result(signal, backend):
    from wavesst.transforms.cwt import CWTResult
    result = sst(signal, wavelet="morlet", scales="auto", fs=FS, backend=backend)
    assert isinstance(result.Wx, CWTResult)


def test_sst_gamma_none_no_thresholding(signal, backend):
    """With gamma=None, all CWT coefficients participate — Tx should be non-zero."""
    result = sst(signal, wavelet="morlet", scales="auto", fs=FS, gamma=None, backend=backend)
    assert np.any(np.abs(result.Tx) > 0)


def test_sst_gamma_auto(signal, backend):
    result = sst(signal, wavelet="morlet", scales="auto", fs=FS, gamma="auto", backend=backend)
    assert isinstance(result, SSTResult)


def test_sst_gamma_float(signal, backend):
    result = sst(signal, wavelet="morlet", scales="auto", fs=FS, gamma=0.01, backend=backend)
    assert isinstance(result, SSTResult)


def test_sst_gamma_callable(signal, backend):
    result = sst(signal, wavelet="morlet", scales="auto", fs=FS,
                 gamma=lambda Wx: 0.05, backend=backend)
    assert isinstance(result, SSTResult)


def test_sst_gamma_universal(signal, backend):
    result = sst(signal, wavelet="morlet", scales="auto", fs=FS,
                 gamma="universal", backend=backend)
    assert isinstance(result, SSTResult)


def test_sst_high_gamma_produces_sparse_tx(signal, backend):
    """Very high threshold → most bins zero."""
    result = sst(signal, wavelet="morlet", scales="auto", fs=FS, gamma=1e10, backend=backend)
    nonzero_frac = np.count_nonzero(np.abs(result.Tx) > 0) / result.Tx.size
    assert nonzero_frac < 0.01
