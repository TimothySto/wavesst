import numpy as np
import pytest
import torch
import wavesst

matplotlib = pytest.importorskip("matplotlib", reason="matplotlib not installed")
matplotlib.use("Agg")  # non-interactive backend — safe in CI


@pytest.fixture(scope="module")
def sst_result_and_ridges():
    cfg = wavesst.Config(device='cpu', dtype=torch.complex128)
    t = np.arange(512) / 256.0
    x = np.cos(2 * np.pi * 32.0 * t)
    sst_r = wavesst.sst(x, fs=256.0, cfg=cfg)
    ridges = wavesst.extract_ridges(sst_r, n=1)
    components = wavesst.reconstruct(sst_r, ridges)
    return sst_r, ridges, components


def test_plot_cwt_returns_axes(sst_result_and_ridges):
    from wavesst.viz.tf_plot import plot_cwt
    sst_r, _, _ = sst_result_and_ridges
    ax = plot_cwt(sst_r.Wx)
    assert ax is not None


def test_plot_sst_returns_axes(sst_result_and_ridges):
    from wavesst.viz.tf_plot import plot_sst
    sst_r, _, _ = sst_result_and_ridges
    ax = plot_sst(sst_r)
    assert ax is not None


def test_plot_ridges_returns_axes(sst_result_and_ridges):
    from wavesst.viz.tf_plot import plot_ridges
    sst_r, ridges, _ = sst_result_and_ridges
    ax = plot_ridges(sst_r, ridges)
    assert ax is not None


def test_plot_components_returns_axes(sst_result_and_ridges):
    from wavesst.viz.tf_plot import plot_components
    sst_r, _, components = sst_result_and_ridges
    ax = plot_components(components, sst_r.times)
    assert ax is not None
