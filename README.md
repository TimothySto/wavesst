# wavesst

**GPU-native Synchrosqueezing Transform library for Python.**

> ⚠️ Work in progress — API is not yet stable.

wavesst provides fast, PyTorch-based implementations of the Continuous Wavelet
Transform (CWT) and its synchrosqueezed variants, designed for time-frequency
analysis of non-stationary signals.  All heavy computation runs on CUDA when
available; a `Config` object lets you control device, dtype, and VRAM budget
per call.

---

![Two-chirp demo — CWT, SST, MSST, STFT-SST, and component reconstruction](demo_chirps.png)

*Two crossing chirps: linear 20→100 Hz + quadratic 120→30 Hz.
From top-left: raw CWT, SST with extracted ridges, MSST (3 iterations),
STFT magnitude, STFT-SST, and CWT-SST component reconstruction.*

---

## Features

| Transform | Class | Description |
|-----------|-------|-------------|
| CWT | `CWTResult` | Morlet wavelet, VRAM-aware chunked scale processing |
| SST | `SSTResult` | Synchrosqueezed CWT via `scatter_add_` reassignment |
| MSST | `MSSTResult` | Multi-synchrosqueezing (iterative, n_iter=1 ≡ SST) |
| STFT | `STFTResult` | GPU-native windowed FFT |
| STFT-SST | `STFTSSTResult` | STFT synchrosqueezing; exact IF via window-derivative |
| Ridge extraction | `Ridge` | Dynamic-programming ridge tracker (Cython) |
| Reconstruction | `Component` | Inverse CWT-SST (admissibility) + STFT-SST (OLA) |

---

## Quick start

```python
import numpy as np
import wavesst

# Sample signal: two pure tones
fs = 256.0
t  = np.arange(1024) / fs
x  = np.cos(2 * np.pi * 32 * t) + np.cos(2 * np.pi * 80 * t)

# CWT → SST → ridge extraction → reconstruction
cfg        = wavesst.Config(device='auto', dtype='complex64')
sst_result = wavesst.sst(x, fs=fs, nv=32, gamma='auto', cfg=cfg)
ridges     = wavesst.extract_ridges(sst_result, n=2, penalty=1.0)
components = wavesst.reconstruct(sst_result, ridges)

print(f"Ridge 1 median: {np.median(ridges[0].freq_path):.1f} Hz")
print(f"Ridge 2 median: {np.median(ridges[1].freq_path):.1f} Hz")
```

### STFT-SST

```python
stft_result = wavesst.stft_sst(x, fs=fs, nperseg=256, noverlap=240,
                                gamma='auto', cfg=cfg)
ridges      = wavesst.extract_ridges(stft_result, n=2)
components  = wavesst.reconstruct(stft_result, ridges, fs=fs)
```

### Config options

```python
# Force CPU, double precision
cfg = wavesst.Config(device='cpu', dtype='complex128')

# Limit VRAM usage (useful on small GPUs)
cfg = wavesst.Config(device='auto', vram_budget_gb=4.0, safety_factor=0.8)

# Temporary override for one call
with cfg.temporary(dtype='complex128'):
    result = wavesst.cwt(x, fs=fs, cfg=cfg)
```

---

## Installation

wavesst uses Cython extensions; you need a C compiler.

```bash
# Clone and install in editable mode
git clone https://github.com/TimothySto/wavelets-3.git
cd wavelets-3
pip install -e . --no-build-isolation

# Optional extras
pip install -e ".[dev]"        # pytest, hypothesis, benchmark
pip install -e ".[reference]"  # ssqueezepy + PyWavelets for validation tests
pip install -e ".[cuda]"       # cupy for CUDA (torch CUDA included via torch)
```

### Requirements

- Python ≥ 3.10
- numpy ≥ 1.24, scipy ≥ 1.11
- torch ≥ 2.1 (CPU or CUDA)
- matplotlib ≥ 3.7
- A C compiler for Cython extensions (MSVC on Windows, GCC on Linux/macOS)

---

## Running tests

```bash
# Full suite (108 tests)
pytest tests/ -q

# Unit tests only
pytest tests/unit/ -q

# Cross-validation against pywt / ssqueezepy (requires extras)
pytest tests/validation/ -v
```

---

## Project layout

```
src/wavesst/
├── config.py                  # Config dataclass + module singleton
├── transforms/
│   ├── cwt.py                 # CWT (chunked, GPU-native)
│   ├── sst.py                 # SST (scatter_add_ reassignment)
│   ├── stft.py                # STFT (GPU-native rfft framing)
│   ├── stft_sst.py            # STFT-SST (window-derivative IF estimator)
│   └── msst.py                # Multi-synchrosqueezing (iterative)
├── analysis/
│   ├── ridge.py               # Dynamic-programming ridge extraction
│   └── reconstruction.py      # Inverse transform → Component
├── viz/
│   └── tf_plot.py             # plot_cwt, plot_sst, plot_ridges, plot_components
└── _core/
    ├── filters.pyx            # Cython: Morlet filter bank
    └── ridge_dp.pyx           # Cython: ridge DP solver
```

---

## Mathematical background

**Synchrosqueezing** (Daubechies & Maes 1996; Thakur & Wu 2011) sharpens a
time-frequency representation by *reassigning* each coefficient to the
instantaneous frequency it estimates, rather than the analysis frequency at
which it was computed.

For the CWT-based SST the instantaneous frequency estimator is:

```
ω̂(a, t) = Re(-i · ∂_t W_x(a, t) / W_x(a, t))
```

and the reassigned representation is:

```
T_x(f, t) = (log 2 / nv) · Σ_{a : |ω̂(a,t) - f| < Δf/2}  W_x(a, t)
```

Reconstruction uses the admissibility constant `C_ψ`:

```
x_k(t) = (2 / C_ψ) · Re[ Σ_{f ∈ band_k(t)}  T_x(f, t) ]
```

For STFT-SST the IF is estimated from the window derivative:

```
∂_τ V_x(η, τ) = −V_{g'}(η, τ)     (exact, no finite-difference error)
ω̂(η, τ)       = η + Im(∂_τ V_x / V_x) / (2π)
```

Reconstruction uses overlap-add synthesis on `V` at the ridge bin (not `T_x`)
to avoid Poisson-sum cancellation from the Hann window's leakage structure.

---

## References

- Daubechies, I. & Maes, S. (1996). *A nonlinear squeezing of the continuous
  wavelet transform based on auditory nerve models.*
- Thakur, G. & Wu, H.-T. (2011). *Synchrosqueezing-based recovery of
  instantaneous frequency from nonuniform samples.*
- Daubechies, I., Lu, J. & Wu, H.-T. (2011). *Synchrosqueezed wavelet
  transforms: an empirical mode decomposition-like tool.*
- Oberlin, T., Meignen, S. & Perrier, V. (2014). *The Fourier-based
  synchrosqueezing transform.*
- Pham, D.-H. & Meignen, S. (2017). *High-order synchrosqueezing transform
  for multicomponent signals analysis.*

---

## Status

This library was built session-by-session as an educational project exploring
GPU-accelerated time-frequency analysis.  It is functional and tested but the
API is not frozen — breaking changes may occur before v1.0.

**Implemented:** CWT, SST, STFT, STFT-SST, MSST, ridge extraction,
reconstruction, basic visualization.

**Planned:** Additional wavelets (Bump, Paul, DOG), true Pham-Meignen MSST,
inverse CWT, adaptive bandwidth reconstruction, save/load utilities, Sphinx docs.
