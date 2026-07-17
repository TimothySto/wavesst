import numpy as np
import pytest
import wavesst
from wavesst.analysis.ridge import Ridge, extract_ridges
from wavesst.analysis.onset import detect_onsets, OnsetResult
from wavesst.transforms.sst import sst
from wavesst.synthesis.chirp import make_chirp

FS = 256.0
N = 512


def _make_ridge_with_onset(f0: float, t_start_s: float, t_end_s: float, cfg):
    x = make_chirp(duration=N / FS, fs=FS, f_start=f0, f_end=f0,
                   method='linear', t_start=t_start_s, t_end=t_end_s)
    result = sst(x.astype(np.float32), fs=FS, nv=32, gamma='auto', cfg=cfg)
    ridges = extract_ridges(result, n=1, penalty=1.0)
    return ridges[0]


def test_detect_onsets_returns_onset_result(cfg):
    ridge = _make_ridge_with_onset(40.0, 0.0, N / FS, cfg)
    result = detect_onsets(ridge)
    assert isinstance(result, OnsetResult)


def test_detect_onsets_has_t_start_t_end(cfg):
    ridge = _make_ridge_with_onset(40.0, 0.0, N / FS, cfg)
    result = detect_onsets(ridge)
    assert hasattr(result, 't_start')
    assert hasattr(result, 't_end')


def test_detect_onsets_full_signal_spans_duration(cfg):
    """Ridge active over full duration: onset near 0, offset near duration."""
    ridge = _make_ridge_with_onset(40.0, 0.0, N / FS, cfg)
    result = detect_onsets(ridge, threshold=0.1)
    assert result.t_start < 0.2
    assert result.t_end > (N / FS) - 0.2


def test_detect_onsets_gated_signal_detects_onset(cfg):
    """Signal active only in [0.5, 1.5] s -> onset should be after 0.3 s."""
    ridge = _make_ridge_with_onset(40.0, 0.5, 1.5, cfg)
    result = detect_onsets(ridge, threshold=0.05)
    assert result.t_start >= 0.3


def test_detect_onsets_t_start_before_t_end(cfg):
    ridge = _make_ridge_with_onset(40.0, 0.0, N / FS, cfg)
    result = detect_onsets(ridge)
    assert result.t_start <= result.t_end


def test_ridge_has_energy_path(cfg):
    t = np.arange(N) / FS
    x = np.cos(2 * np.pi * 32.0 * t).astype(np.float32)
    result = sst(x, fs=FS, nv=32, gamma='auto', cfg=cfg)
    ridges = extract_ridges(result, n=1, penalty=1.0)
    r = ridges[0]
    assert hasattr(r, 'energy_path')
    assert r.energy_path.shape == (N,)
    assert r.energy_path.dtype == np.float64


def test_ridge_energy_path_sum_matches_total(cfg):
    t = np.arange(N) / FS
    x = np.cos(2 * np.pi * 32.0 * t).astype(np.float32)
    result = sst(x, fs=FS, nv=32, gamma='auto', cfg=cfg)
    ridges = extract_ridges(result, n=1, penalty=1.0)
    r = ridges[0]
    np.testing.assert_allclose(r.energy_path.sum(), r.energy, rtol=1e-10)


def test_detect_onsets_zero_energy_returns_zero(cfg):
    """Ridge with zero energy everywhere returns sentinel OnsetResult(0.0, 0.0)."""
    t = np.arange(N) / FS
    x = np.zeros(N, dtype=np.float32)
    result = sst(x, fs=FS, nv=32, gamma='auto', cfg=cfg)
    ridges = extract_ridges(result, n=1, penalty=1.0)
    r = ridges[0]
    # Manually zero out energy_path to test the guard branch
    import dataclasses
    r = dataclasses.replace(r, energy_path=np.zeros(N, dtype=np.float64))
    onset = detect_onsets(r)
    assert onset == OnsetResult(0.0, 0.0)
