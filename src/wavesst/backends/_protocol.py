from abc import ABC, abstractmethod
from typing import Any


class ArrayBackend(ABC):
    @abstractmethod
    def rfft(self, x: Any, n: int | None = None, axis: int = -1) -> Any: ...

    @abstractmethod
    def irfft(self, x: Any, n: int | None = None, axis: int = -1) -> Any: ...

    @abstractmethod
    def abs(self, x: Any) -> Any: ...

    @abstractmethod
    def real(self, x: Any) -> Any: ...

    @abstractmethod
    def conj(self, x: Any) -> Any: ...

    @abstractmethod
    def zeros(self, shape: tuple[int, ...], dtype: Any) -> Any: ...

    @abstractmethod
    def exp(self, x: Any) -> Any: ...

    @abstractmethod
    def from_numpy(self, arr: Any) -> Any: ...

    @abstractmethod
    def to_numpy(self, x: Any) -> Any: ...

    @abstractmethod
    def device(self) -> str: ...

    @property
    @abstractmethod
    def cdtype(self) -> Any: ...

    @property
    @abstractmethod
    def fdtype(self) -> Any: ...
