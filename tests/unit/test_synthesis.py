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


def test_make_chirp_arbitrary_callable():
    """Callable f_inst should produce correct shape and dtype."""
    x = make_chirp(
        duration=1.0, fs=FS, method='arbitrary',
        f_inst=lambda t: 30.0 + 20.0 * t,
    )
    assert x.shape == (256,)
    assert x.dtype == np.float32


def test_make_chirp_arbitrary_array():
    """Array f_inst of matching length should work."""
    N = int(1.0 * FS)
    t = np.arange(N) / FS
    f_arr = 30.0 + 20.0 * t
    x = make_chirp(duration=1.0, fs=FS, method='arbitrary', f_inst=f_arr)
    assert x.shape == (N,)


def test_make_chirp_arbitrary_wrong_length_raises():
    with pytest.raises(ValueError, match="shape"):
        make_chirp(duration=1.0, fs=FS, method='arbitrary', f_inst=np.ones(10))


def test_make_chirp_t_start_zeros_prefix():
    """Samples before t_start should be zero."""
    x = make_chirp(duration=1.0, fs=FS, f_start=40.0, f_end=40.0,
                   method='linear', t_start=0.5)
    assert np.all(x[:int(0.5 * FS)] == 0.0)
    assert not np.all(x[int(0.5 * FS):] == 0.0)


def test_make_chirp_segments_shape():
    x = make_chirp(
        duration=1.0, fs=FS,
        segments=[(30.0, 0.5), (80.0, 0.5)],
    )
    assert x.shape == (256,)


def test_make_chirp_segments_piecewise_frequency():
    """First half should be a pure tone at 40 Hz, second at 80 Hz."""
    x = make_chirp(
        duration=2.0, fs=FS,
        segments=[(40.0, 1.0), (80.0, 1.0)],
    )
    half = int(1.0 * FS)
    fft1 = np.abs(np.fft.rfft(x[:half]))
    freqs = np.fft.rfftfreq(half, d=1.0 / FS)
    assert abs(freqs[np.argmax(fft1)] - 40.0) < 5.0


def test_make_chirp_missing_f_inst_raises():
    with pytest.raises(ValueError, match="f_inst"):
        make_chirp(duration=1.0, fs=FS, method='arbitrary')
