import pytest
import numpy as np
from wavesst.backends import get_backend


@pytest.fixture
def numpy_backend():
    return get_backend("numpy")


@pytest.fixture
def rng():
    return np.random.default_rng(42)
