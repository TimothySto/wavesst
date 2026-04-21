import numpy as np
import pytest
import torch
import wavesst


@pytest.fixture
def cfg():
    """CPU Config with float64 precision — used by all unit tests."""
    return wavesst.Config(device='cpu', dtype=torch.complex128)


@pytest.fixture
def rng():
    return np.random.default_rng(42)
