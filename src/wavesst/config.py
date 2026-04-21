from __future__ import annotations
import contextlib
from dataclasses import dataclass, field
from typing import Iterator
import torch


@dataclass
class Config:
    dtype: torch.dtype = torch.complex64
    device: str = 'auto'
    vram_budget_gb: float | None = None
    chunk_scales: int | None = None
    safety_factor: float = 0.75

    @property
    def real_dtype(self) -> torch.dtype:
        return torch.float32 if self.dtype == torch.complex64 else torch.float64

    def resolve_device(self) -> torch.device:
        if self.device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(self.device)

    def resolve_chunk_scales(self, N: int) -> int:
        """Return max scales per GPU chunk. Returns 2**30 (unlimited) for CPU."""
        if self.chunk_scales is not None:
            return self.chunk_scales
        device = self.resolve_device()
        if device.type != 'cuda':
            return 2 ** 30
        bytes_per = 8 if self.dtype == torch.complex64 else 16
        if self.vram_budget_gb is not None:
            budget = int(self.vram_budget_gb * 1024 ** 3)
        else:
            free, _ = torch.cuda.mem_get_info(device)
            budget = free
        return max(1, int(budget * self.safety_factor / (N * bytes_per)))

    @contextlib.contextmanager
    def temporary(self, **overrides) -> Iterator['Config']:
        """Context manager for temporary Config overrides."""
        old = {k: getattr(self, k) for k in overrides}
        for k, v in overrides.items():
            setattr(self, k, v)
        try:
            yield self
        finally:
            for k, v in old.items():
                setattr(self, k, v)


# Module-level singleton — import and modify in user code, or pass cfg= per call
config = Config()
