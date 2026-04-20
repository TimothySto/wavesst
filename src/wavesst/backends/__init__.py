from wavesst.backends._protocol import ArrayBackend

__all__ = ["ArrayBackend", "get_backend"]


def get_backend(hint: str | None = None) -> "ArrayBackend":
    if hint == "numpy":
        from wavesst.backends.numpy_backend import NumpyBackend
        return NumpyBackend()
    try:
        import torch
        if torch.cuda.is_available():
            from wavesst.backends.torch_backend import TorchBackend
            return TorchBackend()
    except ImportError:
        pass
    try:
        import cupy  # noqa: F401
        from wavesst.backends.cupy_backend import CupyBackend
        return CupyBackend()
    except ImportError:
        pass
    from wavesst.backends.numpy_backend import NumpyBackend
    return NumpyBackend()
