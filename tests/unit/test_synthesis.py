import numpy as np
import pytest
import wavesst
from wavesst.synthesis.chirp import make_chirp

FS = 256.0


def test_make_chirp_linear_shape():
    x = make_chirp(duration=1.0, fs=FS, f_start=20.0, f_end=80.0, method='linear')
    assert x.shape == (256,)


def test_make_chirp_linear_dtype():
    x = make_chirp(duration=1.0, fs=FS, f_start=20.0, f_end=80.0, method='linear')
    assert x.dtype == np.float32


def test_make_chirp_linear_t_end_zeros_suffix():
    """Samples after t_end should be exactly zero."""
    x = make_chirp(duration=1.0, fs=FS, f_start=40.0, f_end=40.0,
                   method='linear', t_end=0.5)
    assert np.all(x[int(0.5 * FS):] == 0.0)
    assert not np.all(x[:int(0.5 * FS)] == 0.0)


def test_make_chirp_quadratic_shape():
    x = make_chirp(duration=1.0, fs=FS, f_start=20.0, f_end=80.0, method='quadratic')
    assert x.shape == (256,)


def test_make_chirp_quadratic_dtype():
    x = make_chirp(duration=1.0, fs=FS, f_start=20.0, f_end=80.0, method='quadratic')
    assert x.dtype == np.float32


def test_make_chirp_missing_f_start_raises():
    with pytest.raises(ValueError, match="f_start"):
        make_chirp(duration=1.0, fs=FS, f_end=80.0, method='linear')


def test_make_chirp_unknown_method_raises():
    with pytest.raises(ValueError, match="Unknown method"):
        make_chirp(duration=1.0, fs=FS, f_start=20.0, f_end=80.0, method='bogus')
