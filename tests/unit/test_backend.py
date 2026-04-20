import numpy as np
import pytest
from wavesst.backends import get_backend
from wavesst.backends._protocol import ArrayBackend
from wavesst.backends.numpy_backend import NumpyBackend


def test_get_backend_numpy_hint():
    b = get_backend("numpy")
    assert isinstance(b, NumpyBackend)


def test_get_backend_returns_array_backend():
    b = get_backend("numpy")
    assert isinstance(b, ArrayBackend)


def test_numpy_backend_device():
    b = get_backend("numpy")
    assert b.device() == "cpu"


def test_numpy_backend_dtypes():
    b = get_backend("numpy")
    assert b.fdtype == np.float64
    assert b.cdtype == np.complex128


def test_numpy_backend_rfft_irfft_roundtrip():
    b = get_backend("numpy")
    rng = np.random.default_rng(0)
    x = rng.standard_normal(256)
    x_rt = b.irfft(b.rfft(x), n=len(x))
    np.testing.assert_allclose(x_rt, x, atol=1e-12)


def test_numpy_backend_abs():
    b = get_backend("numpy")
    x = np.array([-3.0, 0.0, 4.0])
    np.testing.assert_array_equal(b.abs(x), np.array([3.0, 0.0, 4.0]))


def test_numpy_backend_real():
    b = get_backend("numpy")
    x = np.array([1 + 2j, 3 + 4j])
    np.testing.assert_array_equal(b.real(x), np.array([1.0, 3.0]))


def test_numpy_backend_conj():
    b = get_backend("numpy")
    x = np.array([1 + 2j, 3 - 4j])
    np.testing.assert_array_equal(b.conj(x), np.array([1 - 2j, 3 + 4j]))


def test_numpy_backend_zeros():
    b = get_backend("numpy")
    z = b.zeros((3, 4), dtype=np.float64)
    assert z.shape == (3, 4)
    assert z.dtype == np.float64
    np.testing.assert_array_equal(z, np.zeros((3, 4)))


def test_numpy_backend_exp():
    b = get_backend("numpy")
    x = np.array([0.0, 1.0])
    np.testing.assert_allclose(b.exp(x), np.exp(x))


def test_numpy_backend_from_numpy_is_noop():
    b = get_backend("numpy")
    arr = np.array([1.0, 2.0])
    result = b.from_numpy(arr)
    assert result is arr


def test_numpy_backend_to_numpy():
    b = get_backend("numpy")
    arr = np.array([1.0, 2.0])
    result = b.to_numpy(arr)
    np.testing.assert_array_equal(result, arr)
