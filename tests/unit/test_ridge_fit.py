import numpy as np
import pytest
import wavesst
from wavesst.analysis.ridge_fit import fit_ridge, fit_ridge_segments, RidgeFit, SegmentFit
from wavesst.analysis.ridge import extract_ridges
from wavesst.transforms.sst import sst

FS = 256.0
N = 512


def _make_linear_chirp_ridge(f_start, f_end, cfg):
    from wavesst.synthesis.chirp import make_chirp
    x = make_chirp(N / FS, FS, f_start=f_start, f_end=f_end, method='linear')
    result = sst(x.astype(np.float32), fs=FS, nv=32, gamma='auto', cfg=cfg)
    return extract_ridges(result, n=1, penalty=0.01)[0]


def test_fit_ridge_returns_ridge_fit(cfg):
    ridge = _make_linear_chirp_ridge(30.0, 70.0, cfg)
    fit = fit_ridge(ridge, order=1)
    assert isinstance(fit, RidgeFit)


def test_fit_ridge_has_fields(cfg):
    ridge = _make_linear_chirp_ridge(30.0, 70.0, cfg)
    fit = fit_ridge(ridge, order=1)
    assert hasattr(fit, 'order')
    assert hasattr(fit, 'coeffs')
    assert hasattr(fit, 'fitted_freq')
    assert hasattr(fit, 'residuals')
    assert hasattr(fit, 'rmse')


def test_fit_ridge_linear_order(cfg):
    ridge = _make_linear_chirp_ridge(30.0, 70.0, cfg)
    fit = fit_ridge(ridge, order=1)
    assert fit.order == 1
    assert fit.coeffs.shape == (2,)  # order+1 coefficients


def test_fit_ridge_quadratic_order(cfg):
    ridge = _make_linear_chirp_ridge(30.0, 70.0, cfg)
    fit = fit_ridge(ridge, order=2)
    assert fit.order == 2
    assert fit.coeffs.shape == (3,)


def test_fit_ridge_fitted_freq_shape(cfg):
    ridge = _make_linear_chirp_ridge(30.0, 70.0, cfg)
    fit = fit_ridge(ridge, order=1)
    assert fit.fitted_freq.shape == (N,)
    assert fit.residuals.shape == (N,)


def test_fit_ridge_rmse_is_float(cfg):
    ridge = _make_linear_chirp_ridge(30.0, 70.0, cfg)
    fit = fit_ridge(ridge, order=1)
    assert isinstance(fit.rmse, float)
    assert fit.rmse >= 0.0


def test_fit_ridge_linear_chirp_low_rmse(cfg):
    """Linear chirp should be well-fit by order=1 polynomial."""
    ridge = _make_linear_chirp_ridge(30.0, 70.0, cfg)
    fit = fit_ridge(ridge, order=1)
    assert fit.rmse < 5.0  # within 5 Hz for a clean linear chirp


def test_fit_ridge_t_start_t_end_restricts_fit(cfg):
    """When t_start/t_end are given, fit is computed over that window only."""
    ridge = _make_linear_chirp_ridge(30.0, 70.0, cfg)
    half = (N / FS) / 2
    fit_full = fit_ridge(ridge, order=1)
    fit_half = fit_ridge(ridge, order=1, t_start=0.0, t_end=half)
    # Coefficients should differ (fitting different portions of a changing signal)
    assert not np.allclose(fit_full.coeffs, fit_half.coeffs, rtol=0.01)
    # Shape must still be (n_time,) for the full ridge
    assert fit_half.fitted_freq.shape == (N,)


def test_fit_ridge_segments_returns_list(cfg):
    from wavesst.synthesis.chirp import make_chirp
    x = make_chirp(N / FS, FS, f_start=40.0, f_end=40.0, method='linear')
    result = sst(x.astype(np.float32), fs=FS, nv=32, gamma='auto', cfg=cfg)
    ridge = extract_ridges(result, n=1, penalty=1.0)[0]
    fits = fit_ridge_segments(ridge, threshold=0.1, order=1)
    assert isinstance(fits, list)


def test_fit_ridge_segments_returns_segment_fits(cfg):
    from wavesst.synthesis.chirp import make_chirp
    x = make_chirp(N / FS, FS, f_start=40.0, f_end=40.0, method='linear')
    result = sst(x.astype(np.float32), fs=FS, nv=32, gamma='auto', cfg=cfg)
    ridge = extract_ridges(result, n=1, penalty=1.0)[0]
    fits = fit_ridge_segments(ridge, threshold=0.1, order=1)
    assert len(fits) >= 1
    assert isinstance(fits[0], SegmentFit)
    assert hasattr(fits[0], 'onset')
    assert hasattr(fits[0], 'fit')


def test_fit_ridge_segments_empty_on_zero_energy(cfg):
    from wavesst.synthesis.chirp import make_chirp
    import dataclasses
    x = make_chirp(N / FS, FS, f_start=40.0, f_end=40.0, method='linear')
    result = sst(x.astype(np.float32), fs=FS, nv=32, gamma='auto', cfg=cfg)
    ridge = extract_ridges(result, n=1, penalty=1.0)[0]
    zero_ridge = dataclasses.replace(ridge, energy_path=np.zeros(N, dtype=np.float64))
    fits = fit_ridge_segments(zero_ridge, threshold=0.1, order=1)
    assert fits == []


def test_fit_ridge_exported_from_wavesst():
    assert hasattr(wavesst, 'fit_ridge')
    assert hasattr(wavesst, 'fit_ridge_segments')
    assert hasattr(wavesst, 'RidgeFit')
    assert hasattr(wavesst, 'SegmentFit')
