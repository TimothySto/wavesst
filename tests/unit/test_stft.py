import numpy as np
import pytest
import torch
import wavesst
from wavesst.transforms.stft import stft, STFTResult

FS = 256.0
N = 1024


@pytest.fixture
def stft_cfg():
    """STFT uses float32 (complex64) by default; use complex64 for speed."""
    return wavesst.Config(device='cpu', dtype=torch.complex64)


@pytest.fixture
def signal():
    t = np.arange(N) / FS
    return np.cos(2 * np.pi * 32.0 * t).astype(np.float32)


def test_stft_returns_stft_result(signal, stft_cfg):
    result = stft(signal, fs=FS, nperseg=256, cfg=stft_cfg)
    assert isinstance(result, STFTResult)


def test_stft_v_shape(signal, stft_cfg):
    result = stft(signal, fs=FS, nperseg=256, cfg=stft_cfg)
    n_freqs, n_frames = result.V.shape
    assert n_freqs == 256 // 2 + 1


def test_stft_v_is_complex(signal, stft_cfg):
    result = stft(signal, fs=FS, nperseg=256, cfg=stft_cfg)
    assert result.V.is_complex()


def test_stft_freqs_length(signal, stft_cfg):
    result = stft(signal, fs=FS, nperseg=256, cfg=stft_cfg)
    assert len(result.freqs) == result.V.shape[0]


def test_stft_freqs_range(signal, stft_cfg):
    result = stft(signal, fs=FS, nperseg=256, cfg=stft_cfg)
    assert float(result.freqs[0]) == pytest.approx(0.0, abs=1e-6)
    assert float(result.freqs[-1]) == pytest.approx(FS / 2, rel=0.01)


def test_stft_times_length(signal, stft_cfg):
    result = stft(signal, fs=FS, nperseg=256, cfg=stft_cfg)
    assert len(result.times) == result.V.shape[1]


def test_stft_energy_peaks_at_tone(signal, stft_cfg):
    result = stft(signal, fs=FS, nperseg=256, cfg=stft_cfg)
    energy = result.V.abs().pow(2).sum(dim=1)
    peak_idx = energy.argmax().item()
    peak_freq = result.freqs[peak_idx].item()
    assert abs(peak_freq - 32.0) / 32.0 < 0.10
