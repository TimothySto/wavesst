"""
demo_chirps.py — Two-chirp demo for wavesst Session 4.

Signal:
  x(t) = cos(phi_linear(t)) + 0.8 * cos(phi_quad(t))

  Linear chirp:    IF(t) = 20 + 80*(t/T)      →  20 Hz → 100 Hz over T seconds
  Quadratic chirp: IF(t) = 120 - 90*(t/T)^2   →  120 Hz → 30 Hz (downward parabola)

Plots (7 panels):
  1. Input signal
  2. CWT magnitude
  3. SST + extracted ridges
  4. MSST (3 iterations)
  5. STFT magnitude
  6. STFT-SST
  7. SST component reconstruction vs ground truth

True instantaneous frequencies are overlaid on every TF plot as dashed lines.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import wavesst
from wavesst.viz.tf_plot import plot_cwt, plot_ridges

# ─────────────────────────────────────────────────────────────────────────────
# Signal design
# ─────────────────────────────────────────────────────────────────────────────
FS = 512.0      # Hz
T  = 4.0        # seconds
N  = int(FS * T)  # 2048 samples
t  = np.arange(N) / FS

# --- Linear chirp: 20 → 100 Hz ---
f0_lin, f1_lin = 20.0, 100.0
# IF(t) = f0 + (f1-f0)/T * t   →  phase = 2π ∫ IF dt = 2π(f0·t + (f1-f0)/(2T)·t²)
phase_lin = 2 * np.pi * (f0_lin * t + (f1_lin - f0_lin) / (2 * T) * t**2)
IF_lin    = f0_lin + (f1_lin - f0_lin) / T * t

# --- Quadratic chirp: 120 → 30 Hz ---
f0_quad = 120.0
# IF(t) = f0_quad - (f0_quad - 30) * (t/T)^2 = 120 - 90*(t/T)^2
#   → phase = 2π ∫ IF dt = 2π(f0_quad·t - 90/(3*T²)·t³)
k_quad      = (f0_quad - 30.0) / T**2      # 90/16 = 5.625  Hz/s²
phase_quad  = 2 * np.pi * (f0_quad * t - k_quad / 3.0 * t**3)
IF_quad     = f0_quad - k_quad * t**2

# --- Combined signal ---
x = (np.cos(phase_lin) + 0.8 * np.cos(phase_quad)).astype(np.float64)

print(f"Signal: N={N}, fs={FS} Hz, T={T} s")
print(f"  Linear chirp:    {f0_lin:.0f} → {f1_lin:.0f} Hz")
print(f"  Quadratic chirp: {f0_quad:.0f} → {IF_quad[-1]:.1f} Hz")

# ─────────────────────────────────────────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────────────────────────────────────────
cfg = wavesst.Config(device='auto', dtype=torch.complex128)
device_name = cfg.resolve_device().type.upper()
print(f"\nDevice: {device_name}")

print("Running CWT …")
cwt_r = wavesst.cwt(x, fs=FS, nv=32, cfg=cfg)
print(f"  W shape: {cwt_r.W.shape}  freq range: {cwt_r.freqs.min():.1f}–{cwt_r.freqs.max():.1f} Hz")

print("Running SST …")
sst_r = wavesst.sst(x, fs=FS, nv=32, gamma="auto", cfg=cfg)
print(f"  Tx shape: {sst_r.Tx.shape}  nonzero: {sst_r.Tx.abs().gt(0).sum().item()}")

print("Running MSST (3 iterations) …")
msst_r = wavesst.msst(x, fs=FS, nv=32, n_iter=3, gamma="auto", cfg=cfg)
print(f"  Tx shape: {msst_r.Tx.shape}")

print("Running STFT (nperseg=256, noverlap=240) …")
stft_r = wavesst.stft(x, fs=FS, nperseg=256, noverlap=240, cfg=cfg)
print(f"  V shape: {stft_r.V.shape}  df={float(stft_r.freqs[1]-stft_r.freqs[0]):.2f} Hz/bin")

print("Running STFT-SST …")
stft_sst_r = wavesst.stft_sst(x, fs=FS, nperseg=256, noverlap=240, gamma="auto", cfg=cfg)
print(f"  Tx shape: {stft_sst_r.Tx.shape}")

print("Extracting 2 ridges from SST …")
ridges = wavesst.extract_ridges(sst_r, n=2, penalty=2.0)
# Sort by starting frequency (ascending) so ridges[0]=linear chirp, ridges[1]=quadratic chirp
ridges.sort(key=lambda r: r.freq_path[0])
for i, r in enumerate(ridges):
    print(f"  Ridge {i+1}: start={r.freq_path[0]:.1f} Hz  median={np.median(r.freq_path):.1f} Hz  energy={r.energy:.3e}")

print("Reconstructing components …")
components = wavesst.reconstruct(sst_r, ridges)

# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────
_IF_COLORS = ['#00e5ff', '#ffe500']   # cyan=linear, yellow=quadratic

def overlay_ifs(ax, alpha=0.75, legend=True):
    """Draw true IF curves as dashed lines over a TF panel."""
    ax.plot(t, IF_lin,  '--', color=_IF_COLORS[0], lw=1.4, alpha=alpha,
            label='True IF (linear)')
    ax.plot(t, IF_quad, '--', color=_IF_COLORS[1], lw=1.4, alpha=alpha,
            label='True IF (quadratic)')
    ax.set_ylim(0, FS / 2)
    ax.set_xlim(t[0], t[-1])
    if legend:
        ax.legend(fontsize=7, loc='upper right', framealpha=0.4)

def tf_panel(ax, times_arr, freqs_arr, magnitude, title):
    """Standard TF heatmap with log-normalised colour scale."""
    # log scale softens dominant peaks so weaker components are visible
    mag_log = np.log1p(magnitude / (magnitude.max() + 1e-30) * 1000)
    ax.pcolormesh(times_arr, freqs_arr, mag_log, shading='auto', cmap='inferno')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)

# ─────────────────────────────────────────────────────────────────────────────
# Figure layout: 4 rows × 2 cols
# ─────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 20))
fig.patch.set_facecolor('#111111')

title_text = (
    f"wavesst — Two-Chirp Demo  |  {device_name}\n"
    f"Linear chirp 20→100 Hz  +  Quadratic chirp 120→30 Hz  "
    f"(fs={FS:.0f} Hz, N={N})"
)
fig.suptitle(title_text, fontsize=12, color='white', y=0.99)

gs = fig.add_gridspec(4, 2, hspace=0.50, wspace=0.30,
                      left=0.07, right=0.97, top=0.96, bottom=0.04)

# ── Row 0: signal (full width) ───────────────────────────────────────────────
ax_sig = fig.add_subplot(gs[0, :])
ax_sig.set_facecolor('#1a1a1a')
ax_sig.plot(t, x, color='#88ccff', lw=0.7, alpha=0.85)
ax_sig.plot(t, np.cos(phase_lin),       color=_IF_COLORS[0], lw=0.9, alpha=0.55,
            label='Linear component')
ax_sig.plot(t, 0.8 * np.cos(phase_quad), color=_IF_COLORS[1], lw=0.9, alpha=0.55,
            label='Quadratic component (×0.8)')
ax_sig.set_xlim(t[0], t[-1])
ax_sig.set_xlabel("Time (s)", color='white')
ax_sig.set_ylabel("Amplitude",  color='white')
ax_sig.set_title("Input Signal", color='white')
ax_sig.tick_params(colors='white')
for sp in ax_sig.spines.values():
    sp.set_color('#444444')
ax_sig.legend(fontsize=8, loc='upper right', framealpha=0.3,
              labelcolor='white', facecolor='#222222')

# ── Row 1: CWT | SST + ridges ────────────────────────────────────────────────
ax_cwt = fig.add_subplot(gs[1, 0])
ax_cwt.set_facecolor('#000000')
tf_panel(ax_cwt, cwt_r.times, cwt_r.freqs,
         cwt_r.W.abs().numpy(), "CWT Magnitude")
overlay_ifs(ax_cwt)

ax_sst = fig.add_subplot(gs[1, 1])
ax_sst.set_facecolor('#000000')
tf_panel(ax_sst, sst_r.times, sst_r.freqs,
         sst_r.Tx.abs().numpy(), "SST + Ridges")
overlay_ifs(ax_sst, legend=False)
ridge_colors = ['#ff4444', '#44ff88']
for ridge, col in zip(ridges, ridge_colors):
    ax_sst.plot(sst_r.times, ridge.freq_path, color=col, lw=1.8, alpha=0.9)

# ── Row 2: MSST | STFT ───────────────────────────────────────────────────────
ax_msst = fig.add_subplot(gs[2, 0])
ax_msst.set_facecolor('#000000')
tf_panel(ax_msst, msst_r.times, msst_r.freqs,
         msst_r.Tx.abs().numpy(), "MSST (3 iterations)")
overlay_ifs(ax_msst)

ax_stft = fig.add_subplot(gs[2, 1])
ax_stft.set_facecolor('#000000')
tf_panel(ax_stft,
         stft_r.times.cpu().numpy(), stft_r.freqs.cpu().numpy(),
         stft_r.V.abs().cpu().numpy(), "STFT Magnitude")
overlay_ifs(ax_stft)

# ── Row 3: STFT-SST | Reconstruction ────────────────────────────────────────
ax_sstsft = fig.add_subplot(gs[3, 0])
ax_sstsft.set_facecolor('#000000')
tf_panel(ax_sstsft,
         stft_sst_r.times.cpu().numpy(), stft_sst_r.freqs.cpu().numpy(),
         stft_sst_r.Tx.abs().cpu().numpy(), "STFT-SST")
overlay_ifs(ax_sstsft)

ax_rec = fig.add_subplot(gs[3, 1])
ax_rec.set_facecolor('#1a1a1a')
ax_rec.plot(t, np.cos(phase_lin),          color=_IF_COLORS[0], lw=0.9, alpha=0.45,
            label='True linear')
ax_rec.plot(t, 0.8 * np.cos(phase_quad),   color=_IF_COLORS[1], lw=0.9, alpha=0.45,
            label='True quad (×0.8)')
for i, (comp, col) in enumerate(zip(components, ridge_colors)):
    ax_rec.plot(t, comp.signal, color=col, lw=1.2, alpha=0.9,
                label=f'Reconstructed {i+1}')
ax_rec.set_xlim(t[0], t[-1])
ax_rec.set_xlabel("Time (s)", color='white')
ax_rec.set_ylabel("Amplitude",  color='white')
ax_rec.set_title("SST Component Reconstruction", color='white')
ax_rec.tick_params(colors='white')
for sp in ax_rec.spines.values():
    sp.set_color('#444444')
ax_rec.legend(fontsize=8, loc='upper right', framealpha=0.3,
              labelcolor='white', facecolor='#222222')

# ── Style all TF axes ────────────────────────────────────────────────────────
for ax in [ax_cwt, ax_sst, ax_msst, ax_stft, ax_sstsft]:
    ax.set_xlabel("Time (s)", color='white')
    ax.set_ylabel("Frequency (Hz)", color='white')
    ax.title.set_color('white')
    ax.tick_params(colors='white')
    for sp in ax.spines.values():
        sp.set_color('#444444')

# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────
out = "D:/Claude General/wavelets-3/demo_chirps.png"
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"\nSaved → {out}")
plt.close()
