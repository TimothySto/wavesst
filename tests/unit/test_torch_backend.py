import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch not installed")
from wavesst.backends.torch_backend import TorchBackend
from wavesst.backends._protocol import ArrayBackend


@pytest.fixture
def b():
    return TorchBackend(device="cpu")


def test_torch_backend_is_array_backend(b):
    assert isinstance(b, ArrayBackend)


def test_torch_backend_device_cpu(b):
    assert b.device() == "cpu"


def test_torch_backend_fdtype(b):
    import torch
    assert b.fdtype == torch.float64


def test_torch_backend_cdtype(b):
    import torch
    assert b.cdtype == torch.complex128


def test_torch_backend_rfft_irfft_roundtrip(b):
    rng = np.random.default_rng(0)
    x_np = rng.standard_normal(256)
    x = b.from_numpy(x_np)
    x_rt = b.to_numpy(b.irfft(b.rfft(x), n=256))
    np.testing.assert_allclose(x_rt, x_np, atol=1e-12)


def test_torch_backend_abs(b):
    x = b.from_numpy(np.array([-3.0, 0.0, 4.0]))
    result = b.to_numpy(b.abs(x))
    np.testing.assert_array_equal(result, [3.0, 0.0, 4.0])


def test_torch_backend_real(b):
    import torch
    x = torch.tensor([1+2j, 3+4j], dtype=torch.complex128)
    np.testing.assert_array_equal(b.to_numpy(b.real(x)), [1.0, 3.0])


def test_torch_backend_conj(b):
    import torch
    x = torch.tensor([1+2j, 3-4j], dtype=torch.complex128)
    np.testing.assert_array_equal(b.to_numpy(b.conj(x)), [1-2j, 3+4j])


def test_torch_backend_zeros(b):
    import torch
    z = b.zeros((3, 4), dtype=torch.float64)
    assert z.shape == (3, 4)
    np.testing.assert_array_equal(b.to_numpy(z), np.zeros((3, 4)))


def test_torch_backend_exp(b):
    x = b.from_numpy(np.array([0.0, 1.0]))
    np.testing.assert_allclose(b.to_numpy(b.exp(x)), np.exp([0.0, 1.0]))


def test_torch_backend_from_numpy_to_numpy_roundtrip(b):
    arr = np.array([1.0, 2.0, 3.0])
    result = b.to_numpy(b.from_numpy(arr))
    np.testing.assert_array_equal(result, arr)
