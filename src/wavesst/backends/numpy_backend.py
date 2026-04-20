import numpy as np
from wavesst.backends._protocol import ArrayBackend


class NumpyBackend(ArrayBackend):
    def rfft(self, x, n=None, axis=-1):
        return np.fft.rfft(x, n=n, axis=axis)

    def irfft(self, x, n=None, axis=-1):
        return np.fft.irfft(x, n=n, axis=axis)

    def abs(self, x):
        return np.abs(x)

    def real(self, x):
        return np.real(x)

    def conj(self, x):
        return np.conj(x)

    def zeros(self, shape, dtype):
        return np.zeros(shape, dtype=dtype)

    def exp(self, x):
        return np.exp(x)

    def from_numpy(self, arr):
        return arr

    def to_numpy(self, x):
        return np.asarray(x)

    def device(self) -> str:
        return "cpu"

    @property
    def cdtype(self):
        return np.complex128

    @property
    def fdtype(self):
        return np.float64
