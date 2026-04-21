import numpy as np
import pytest
import torch
import wavesst
from wavesst.transforms.stft_sst import stft_sst, STFTSSTResult

FS = 256.0
N = 1024


@pytest.fixture
def stft_sst_cfg():
    return wavesst.Config(device='cpu', dtype=torch.complex64)


@pytest.fixture
def tone():
    t = np.arange(N) / FS
    return np.cos(2 * np.pi * 32.0 * t).astype(np.float32)


def test_stft_sst_returns_result(tone, stft_sst_cfg):
    result = stft_sst(tone, fs=FS, nperseg=256, cfg=stft_sst_cfg)
    assert isinstance(result, STFTSSTResult)


def test_stft_sst_tx_shape(tone, stft_sst_cfg):
    result = stft_sst(tone, fs=FS, nperseg=256, cfg=stft_sst_cfg)
    n_freqs, n_frames = result.Tx.shape
    assert n_freqs == 256 // 2 + 1


def test_stft_sst_tx_is_complex(tone, stft_sst_cfg):
    result = stft_sst(tone, fs=FS, nperseg=256, cfg=stft_sst_cfg)
    assert result.Tx.is_complex()


def test_stft_sst_sharpens_stft(stft_sst_cfg):
    """STFT-SST Tx energy should be no less concentrated than raw STFT V.

    Uses a two-tone signal where the two components are slightly off bin-centre
    so there is inter-bin leakage for SST to correct.
    """
    # Two tones slightly off bin centres so STFT has leakage SST can sharpen
    t = np.arange(N) / FS
    x = (np.cos(2 * np.pi * 32.5 * t) + np.cos(2 * np.pi * 80.5 * t)).astype(np.float32)
    result = stft_sst(x, fs=FS, nperseg=256, gamma=None, cfg=stft_sst_cfg)
    stft_energy = result.Vx.V.abs().pow(2).sum(dim=1).numpy()
    sst_energy = result.Tx.abs().pow(2).sum(dim=1).numpy()

    def entropy(e):
        p = e / e.sum()
        p = p[p > 0]
        return -np.sum(p * np.log(p))

    # SST should be at least as concentrated (lower or equal entropy)
    assert entropy(sst_energy) <= entropy(stft_energy) + 1e-5


def test_stft_sst_peak_near_tone_freq(tone, stft_sst_cfg):
    result = stft_sst(tone, fs=FS, nperseg=256, gamma="auto", cfg=stft_sst_cfg)
    energy = result.Tx.abs().pow(2).sum(dim=1)
    peak_freq = result.freqs[energy.argmax()].item()
    assert abs(peak_freq - 32.0) / 32.0 < 0.10
