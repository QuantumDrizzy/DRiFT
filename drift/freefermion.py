"""
drift.freefermion — Phase 18: an exact oracle for the transverse-field Ising chain.

`drift.quantum` (exact, ~16 spins) and `drift.mps` (TEBD, 64+ spins) both solve the open
TFIM chain H = −Σ ZᵢZᵢ₊₁ − g Σ Xᵢ. That chain is free fermions, so it has an exact solution
at any length — the oracle the other two are checked against, the same role
`exact_ground_state` plays for the classical faces.

Jordan–Wigner with X as the occupation, Majoranas a₂ᵢ = (Π_{k<i} X_k) Zᵢ,
a₂ᵢ₊₁ = (Π_{k<i} X_k) Yᵢ, gives Xᵢ = i a₂ᵢ a₂ᵢ₊₁ and ZᵢZᵢ₊₁ = i a₂ᵢ₊₁ a₂ᵢ₊₂, so
H = (i/4) Σ A_kl a_k a_l with A real antisymmetric. The ground state has
⟨a_k a_l⟩ = δ_kl + iΓ_kl, Γ = −i·sign(iA), E₀ = −¼ Σ|eig(iA)|, and the entanglement of the
first ℓ sites follows from the eigenvalues ±ν of the 2ℓ × 2ℓ block of Γ.

What it measures (Phase 18): S(ℓ) at the critical point grows like (c/6)·log, with the
universal c = ½ — the scaling MERA is built to reproduce, and the quantitative content of
the "space from entanglement" picture. ``docs/results/PHASE18-results.md``.
"""

from __future__ import annotations

import numpy as np


def majorana_matrix(L: int, g: float, j: float = 1.0) -> np.ndarray:
    A = np.zeros((2 * L, 2 * L))
    for i in range(L):
        A[2 * i, 2 * i + 1] = -2.0 * g
    for i in range(L - 1):
        A[2 * i + 1, 2 * i + 2] = -2.0 * j
    return A - A.T


def ground_state(L: int, g: float, j: float = 1.0) -> tuple[np.ndarray, float]:
    """(Γ, E₀) of the open chain."""
    lam, U = np.linalg.eigh(1j * majorana_matrix(L, g, j))
    Gamma = np.real(-1j * ((U * np.sign(lam)) @ U.conj().T))
    return Gamma, float(-0.25 * np.abs(lam).sum())


def block_entropies(L: int, g: float, j: float = 1.0) -> np.ndarray:
    """S(ℓ) in nats for the first ℓ sites, ℓ = 1 … L−1."""
    Gamma, _ = ground_state(L, g, j)
    S = []
    for l in range(1, L):
        nu = np.linalg.eigvalsh(1j * Gamma[:2 * l, :2 * l])
        nu = np.clip(nu[nu > 0], 0.0, 1 - 1e-15)
        p = (1 + nu) / 2
        q = 1 - p
        S.append(float(-(p * np.log(p) + q * np.log(q)).sum()))
    return np.array(S)


def fit_central_charge(S: np.ndarray, L: int, window: tuple[float, float] = (1 / 8, 7 / 8)):
    """Fit S(ℓ) = (c/6)·log[(2L/π) sin(πℓ/L)] + b (open boundaries) over ℓ ∈ window·L.
    Returns (c, b)."""
    l = np.arange(1, L)
    keep = (l >= window[0] * L) & (l <= window[1] * L)
    x = np.log(2 * L / np.pi * np.sin(np.pi * l / L)) / 6
    c, b = np.polyfit(x[keep], S[keep], 1)
    return float(c), float(b)
