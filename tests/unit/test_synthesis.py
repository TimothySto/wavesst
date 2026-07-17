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


def test_make_chirp_linear_amplitude():
    x = make_chirp(duration=2.0, fs=FS, f_start=20.0, f_end=80.0, method='linear')
    assert np.all(np.abs(x) <= 1.0 + 1e-5)


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
