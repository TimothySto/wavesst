import subprocess
import sys
import pytest
import numpy as np
from wavesst.backends import get_backend


def _torch_importable() -> bool:
    """Check if torch can be imported without crashing the process.
    Runs in a subprocess to isolate the fatal DLL crash (WinError 127 / 0xc0000139)
    that occurs on some Windows configurations.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import torch; print(torch.__version__)"],
            timeout=15,
            capture_output=True,
        )
        return result.returncode == 0
    except Exception:
        return False


# Exclude torch tests from collection if torch is not importable/stable
if not _torch_importable():
    collect_ignore_glob = ["**/test_torch_backend.py"]


@pytest.fixture
def numpy_backend():
    return get_backend("numpy")


@pytest.fixture
def rng():
    return np.random.default_rng(42)
