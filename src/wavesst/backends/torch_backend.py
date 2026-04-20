from __future__ import annotations
import numpy as np
import torch
from wavesst.backends._protocol import ArrayBackend


class TorchBackend(ArrayBackend):
    def __init__(self, device: str = "cpu"):
        self._device = torch.device(device)

    def rfft(self, x, n=None, axis=-1):
        return torch.fft.rfft(x, n=n, dim=axis)

    def irfft(self, x, n=None, axis=-1):
        return torch.fft.irfft(x, n=n, dim=axis)

    def abs(self, x):
        return torch.abs(x)

    def real(self, x):
        return torch.real(x)

    def conj(self, x):
        return torch.conj(x)

    def zeros(self, shape, dtype):
        return torch.zeros(shape, dtype=dtype, device=self._device)

    def exp(self, x):
        return torch.exp(x)

    def from_numpy(self, arr: np.ndarray):
        return torch.from_numpy(np.asarray(arr)).to(self._device)

    def to_numpy(self, x) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def device(self) -> str:
        return str(self._device)

    @property
    def cdtype(self):
        return torch.complex128

    @property
    def fdtype(self):
        return torch.float64
