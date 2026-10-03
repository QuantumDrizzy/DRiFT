"""
drift.scrambling — Phase 17: what chaos looks like in a spectrum, and what only looks like it.

The mixed-field Ising chain (open boundaries)

    H = Σ ZᵢZᵢ₊₁ + g Σ Xᵢ + h Σ Zᵢ

is chaotic at (g, h) = (1.05, 0.5) (Bañuls, Cirac, Hastings 2011) and free fermions at h = 0.
Two probes, both exact:

  * **spacing ratio** ⟨r⟩, rₙ = min(sₙ, sₙ₊₁)/max(sₙ, sₙ₊₁), inside one symmetry sector:
    GOE ≈ 0.5307 (chaos) against Poisson 2 ln 2 − 1 ≈ 0.3863 (integrable). Mixing sectors
    hides the repulsion, so the reflection-even sector is projected exactly.
  * **OTOC** at infinite temperature, F(t) = Tr(W(t) V W(t) V)/2ᴸ. It measures how far an
    operator has spread — a light cone — and it is **not by itself a chaos test**: on the
    free-fermion chain the Z–Z OTOC decays (Z carries a Jordan–Wigner string) while the X–X
    OTOC does not (X is a local fermion bilinear).

Measured, with pre-registered predictions, in rse-hpc-lab exercise 13 (ADR-005):
``docs/results/PHASE17-results.md``.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg  # noqa: F401  (sp.linalg.norm)

GOE_R = 0.5307
POISSON_R = 2 * np.log(2) - 1


def mixed_field_ising(L: int, g: float, h: float) -> sp.csr_matrix:
    """Sparse H in the Z basis; site k is bit k of the basis index."""
    D = 1 << L
    s = np.arange(D)
    z = 1 - 2 * ((s[:, None] >> np.arange(L)) & 1)
    diag = (z[:, :-1] * z[:, 1:]).sum(1) + h * z.sum(1)
    rows = [s] + [s] * L
    cols = [s] + [s ^ (1 << k) for k in range(L)]
    vals = [diag.astype(float)] + [np.full(D, float(g))] * L
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                         shape=(D, D))


def reflection_even_basis(L: int) -> sp.csr_matrix:
    """Isometry B (2ᴸ × d) onto the reflection-even sector."""
    D = 1 << L
    s = np.arange(D)
    rev = np.zeros(D, dtype=np.int64)
    for k in range(L):
        rev |= ((s >> k) & 1) << (L - 1 - k)
    reps = s[s <= rev]
    rows, cols, vals = [], [], []
    w = 1 / np.sqrt(2)
    for c, a in enumerate(reps):
        b = rev[a]
        if a == b:
            rows.append(a); cols.append(c); vals.append(1.0)
        else:
            rows += [a, b]; cols += [c, c]; vals += [w, w]
    return sp.csr_matrix((vals, (rows, cols)), shape=(D, len(reps)))


def spacing_ratio(L: int, g: float, h: float, degenerate: float = 1e-10) -> dict:
    """⟨r⟩ over the central half of the reflection-even spectrum."""
    H = mixed_field_ising(L, g, h)
    B = reflection_even_basis(L)
    He = (B.T @ H @ B).toarray()
    leak = float(sp.linalg.norm(H @ B - B @ (B.T @ H @ B)))
    E = np.linalg.eigvalsh(He)
    d = len(E)
    s = np.diff(E[d // 4: 3 * d // 4])
    a, b = s[:-1], s[1:]
    keep = np.maximum(a, b) > degenerate
    r = np.minimum(a, b)[keep] / np.maximum(a, b)[keep]
    return {"r_mean": float(r.mean()), "n_ratios": int(r.size), "sector_dim": d,
            "dropped": int((~keep).sum()), "leak": leak}


def site_op(L: int, k: int, which: str) -> np.ndarray:
    """Dense X_k or Z_k."""
    s = np.arange(1 << L)
    if which == "Z":
        return np.diag((1 - 2 * ((s >> k) & 1)).astype(float))
    M = np.zeros((1 << L, 1 << L))
    M[s ^ (1 << k), s] = 1.0
    return M


def otoc(L: int, g: float, h: float, which: str, distances, times) -> np.ndarray:
    """F(t, r) for W = O₀, V = O_r, O ∈ {X, Z}, infinite temperature, exact."""
    E, U = np.linalg.eigh(mixed_field_ising(L, g, h).toarray())
    W = U.T @ site_op(L, 0, which) @ U
    dE = E[:, None] - E[None, :]
    F = np.zeros((len(distances), len(times)))
    for ri, r in enumerate(distances):
        V = U.T @ site_op(L, r, which) @ U
        for ti, t in enumerate(times):
            M = (np.exp(1j * dE * t) * W) @ V            # W(t) = e^{iHt} W e^{−iHt}
            F[ri, ti] = float(np.real(np.sum(M * M.T))) / (1 << L)
    return F

