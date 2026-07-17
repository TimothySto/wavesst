import numpy as np
import pytest
import wavesst
from wavesst.synthesis.chirp import make_chirp, make_amfm

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


def test_make_amfm_shape():
    x = make_amfm(duration=1.0, fs=FS, f_carrier=50.0)
    assert x.shape == (256,)


def test_make_amfm_dtype():
    x = make_amfm(duration=1.0, fs=FS, f_carrier=50.0)
    assert x.dtype == np.float32


def test_make_amfm_no_modulation_is_pure_tone():
    """No am/fm → pure cosine at f_carrier; FFT peak should be at f_carrier."""
    x = make_amfm(duration=2.0, fs=FS, f_carrier=40.0)
    fft = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), d=1.0 / FS)
    assert abs(freqs[np.argmax(fft)] - 40.0) < 2.0


def test_make_amfm_am_callable_scales_amplitude():
    """am_func returning 0.5 everywhere → max amplitude ≤ 0.6."""
    x = make_amfm(duration=1.0, fs=FS, f_carrier=40.0, am_func=lambda t: 0.5)
    assert np.max(np.abs(x)) < 0.6


def test_make_amfm_am_array():
    N = int(1.0 * FS)
    am = np.full(N, 2.0)
    x = make_amfm(duration=1.0, fs=FS, f_carrier=40.0, am_func=am)
    assert x.shape == (N,)
    assert np.max(np.abs(x)) < 2.1


def test_make_amfm_fm_shifts_frequency():
    """Positive constant fm_func → carrier shifts upward."""
    x_base = make_amfm(duration=2.0, fs=FS, f_carrier=40.0)
    x_fm   = make_amfm(duration=2.0, fs=FS, f_carrier=40.0,
                        fm_func=lambda t: 20.0)
    fft_base = np.abs(np.fft.rfft(x_base))
    fft_fm   = np.abs(np.fft.rfft(x_fm))
    freqs = np.fft.rfftfreq(len(x_base), d=1.0 / FS)
    assert freqs[np.argmax(fft_fm)] > freqs[np.argmax(fft_base)]


def test_make_amfm_t_start_zeros_prefix():
    x = make_amfm(duration=1.0, fs=FS, f_carrier=40.0, t_start=0.5)
    assert np.all(x[:int(0.5 * FS)] == 0.0)


from wavesst.synthesis.noise import make_noise


def test_make_noise_white_shape():
    x = make_noise(duration=1.0, fs=FS, color='white')
    assert x.shape == (256,)


def test_make_noise_white_dtype():
    x = make_noise(duration=1.0, fs=FS, color='white')
    assert x.dtype == np.float32


def test_make_noise_seeded_reproducible():
    x1 = make_noise(duration=1.0, fs=FS, color='white', seed=0)
    x2 = make_noise(duration=1.0, fs=FS, color='white', seed=0)
    np.testing.assert_array_equal(x1, x2)


def test_make_noise_different_seeds_differ():
    x1 = make_noise(duration=1.0, fs=FS, color='white', seed=0)
    x2 = make_noise(duration=1.0, fs=FS, color='white', seed=1)
    assert not np.array_equal(x1, x2)


def test_make_noise_pink_shape():
    x = make_noise(duration=1.0, fs=FS, color='pink')
    assert x.shape == (256,)


def test_make_noise_brown_shape():
    x = make_noise(duration=1.0, fs=FS, color='brown')
    assert x.shape == (256,)


def test_make_noise_impulsive_shape():
    x = make_noise(duration=1.0, fs=FS, color='impulsive')
    assert x.shape == (256,)


def test_make_noise_impulsive_is_sparse():
    """Impulsive noise should have mostly zero samples."""
    x = make_noise(duration=10.0, fs=FS, color='impulsive', seed=0)
    frac_nonzero = float(np.count_nonzero(x)) / len(x)
    assert frac_nonzero < 0.5


def test_make_noise_amplitude_scales_output():
    x1 = make_noise(duration=1.0, fs=FS, color='white', amplitude=1.0, seed=0)
    x2 = make_noise(duration=1.0, fs=FS, color='white', amplitude=2.0, seed=0)
    np.testing.assert_allclose(x2, 2.0 * x1, rtol=1e-5)


def test_make_noise_unknown_color_raises():
    with pytest.raises(ValueError, match="color"):
        make_noise(duration=1.0, fs=FS, color='ultraviolet')
