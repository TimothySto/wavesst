import torch
import pytest
import wavesst


def test_config_default_dtype():
    cfg = wavesst.Config()
    assert cfg.dtype == torch.complex64


def test_config_real_dtype_complex64():
    cfg = wavesst.Config(dtype=torch.complex64)
    assert cfg.real_dtype == torch.float32


def test_config_real_dtype_complex128():
    cfg = wavesst.Config(dtype=torch.complex128)
    assert cfg.real_dtype == torch.float64


def test_config_resolve_device_cpu():
    cfg = wavesst.Config(device='cpu')
    assert cfg.resolve_device() == torch.device('cpu')


def test_config_resolve_device_auto_returns_device():
    cfg = wavesst.Config(device='auto')
    d = cfg.resolve_device()
    assert isinstance(d, torch.device)
    assert d.type in ('cpu', 'cuda')


def test_config_resolve_chunk_scales_cpu_unlimited():
    cfg = wavesst.Config(device='cpu')
    assert cfg.resolve_chunk_scales(512) >= 2**20


def test_config_chunk_scales_explicit():
    cfg = wavesst.Config(chunk_scales=64)
    assert cfg.resolve_chunk_scales(512) == 64


def test_config_temporary_restores():
    cfg = wavesst.Config(dtype=torch.complex64)
    with cfg.temporary(dtype=torch.complex128):
        assert cfg.dtype == torch.complex128
    assert cfg.dtype == torch.complex64


def test_module_singleton_exists():
    import wavesst
    assert isinstance(wavesst.config, wavesst.Config)
