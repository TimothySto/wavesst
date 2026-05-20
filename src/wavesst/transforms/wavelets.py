from __future__ import annotations

import math

import torch


def _bump_filter_bank(omega, scales, w0, real_dtype):
    """
    Bump wavelet filter bank.

    ψ̂(a·ω) = exp( 1 / (u² − 1) )   for |u| < 1 and a·ω > 0
    where u = (a·ω) / w0 − 1

    Returns psi_hat of shape (n_chunk, N), real dtype.
    """
    scaled = scales[:, None] * omega[None, :]
    u = scaled / w0 - 1.0
    support = (u.abs() < 1.0) & (scaled > 0.0)
    denom = torch.where(support, u ** 2 - 1.0, torch.full_like(u, -1.0))
    psi_hat = torch.where(
        support,
        torch.exp(1.0 / denom),
        torch.zeros_like(denom),
    ).to(real_dtype)
    return psi_hat


def _paul_filter_bank(omega, scales, m, real_dtype):
    """
    Paul wavelet of order m.

    ψ̂(a·ω) = norm · (a·ω)^m · exp(−a·ω)   for a·ω > 0
    norm = 2^m / sqrt(m · (2m)!)

    Center frequency: scaled_omega = m
    Returns psi_hat of shape (n_chunk, N), real dtype.
    """
    norm = (2 ** m) / math.sqrt(m * math.factorial(2 * m))
    scaled = scales[:, None] * omega[None, :]
    support = scaled > 0.0
    scaled_pos = scaled.clamp(min=0.0)
    psi_hat = torch.where(
        support,
        (norm * scaled_pos ** m * torch.exp(-scaled_pos)).to(real_dtype),
        torch.zeros_like(scaled, dtype=real_dtype),
    )
    return psi_hat


def _dog_filter_bank(omega, scales, m, real_dtype):
    """
    Derivative of Gaussian (DOG) wavelet of order m.

    ψ̂(a·ω) = norm · (a·ω)^m · exp(−(a·ω)²/2)   for a·ω > 0
    norm = 1 / sqrt(Γ(m + 1/2))

    Center frequency: scaled_omega = sqrt(m)
    Returns psi_hat of shape (n_chunk, N), real dtype.
    """
    norm = 1.0 / math.sqrt(math.gamma(m + 0.5))
    scaled = scales[:, None] * omega[None, :]
    support = scaled > 0.0
    scaled_pos = scaled.clamp(min=0.0)
    psi_hat = torch.where(
        support,
        (norm * scaled_pos ** m * torch.exp(-0.5 * scaled_pos ** 2)).to(real_dtype),
        torch.zeros_like(scaled, dtype=real_dtype),
    )
    return psi_hat
