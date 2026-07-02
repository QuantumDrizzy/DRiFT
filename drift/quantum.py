"""
drift.quantum — the quantum ground state and the χ thermometer.

Phases 1-2 found CLASSICAL ground states: a single spin configuration, a product state
with bond dimension χ = 1 — no entanglement, nothing for a tensor network to measure.
The interesting regime is the QUANTUM transverse-field Ising model

    H = -Σ_ij J_ij Z_i Z_j  -  Γ Σ_i X_i

whose ground state is a *superposition* that can be deeply entangled. We build H as a
sparse 2ⁿ × 2ⁿ matrix, find its ground state exactly with Lanczos (small n), and measure:

  * the entanglement entropy S across a bipartition, and
  * the effective bond dimension χ a matrix-product state would need to hold the state.

χ is the thermometer for *how much* the state computes: it stays small in the ordered
and disordered phases and **peaks at the quantum phase transition** (Γ ≈ J), exactly
where the state is hardest to compress. That is the Blaze lesson — χ-truncation is
information budgeting — turned into a probe of physical computation.

Exact only: tractable to ~14 spins (2¹⁴ = 16 384). This is the honest *baseline* — the full
2ⁿ state, from which χ is read off exactly. The scalable counterpart lives in `drift.mps`
(Phase 13): a matrix-product-state solver that finds the ground state by imaginary-time
evolution and reaches n = 64+, with χ as its own truncation budget rather than a measurement
taken after the fact. Use this module as the small-n reference the MPS solver is validated
against; use `drift.mps` when you need to go past the wall. (A GPU/Rust port is the next
scaling story; the Python MPS reference is what finally delivers the "read with tensor
networks" thesis.)
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

# single-qubit operators
_I2 = sp.identity(2, format="csr")
_Z = sp.csr_matrix(np.array([[1.0, 0.0], [0.0, -1.0]]))
_X = sp.csr_matrix(np.array([[0.0, 1.0], [1.0, 0.0]]))


def _op_at(op: sp.spmatrix, i: int, n: int) -> sp.spmatrix:
    """Embed a single-qubit operator on site i into the full n-qubit space (Kron chain)."""
    out = op if i == 0 else _I2
    for k in range(1, n):
        out = sp.kron(out, op if k == i else _I2, format="csr")
    return out


def ising_chain_1d(n: int, j: float = 1.0, periodic: bool = False) -> np.ndarray:
    """Coupling matrix of a 1-D Ising chain (nearest-neighbor, J = j)."""
    J = np.zeros((n, n))
    for i in range(n - 1):
        J[i, i + 1] = J[i + 1, i] = j
    if periodic and n > 2:
        J[0, n - 1] = J[n - 1, 0] = j
    return J


def tfim_terms(J: np.ndarray):
    """Pre-split the transverse-field Ising Hamiltonian into (H_zz, H_x).

    H(Γ) = H_zz + Γ·H_x, so a Γ-sweep reuses the (expensive) ZZ part. Returns sparse CSR.
    """
    n = J.shape[0]
    Z = [_op_at(_Z, i, n) for i in range(n)]
    dim = 1 << n
    H_zz = sp.csr_matrix((dim, dim))
    for i in range(n):
        for jj in range(i + 1, n):
            if J[i, jj] != 0.0:
                H_zz = H_zz - J[i, jj] * (Z[i] @ Z[jj])
    H_x = sp.csr_matrix((dim, dim))
    for i in range(n):
        H_x = H_x - _op_at(_X, i, n)
    return H_zz.tocsr(), H_x.tocsr()


def ground_state(H: sp.spmatrix):
    """Lowest eigenpair (E0, ψ0) via Lanczos."""
    vals, vecs = eigsh(H, k=1, which="SA")
    return float(vals[0]), np.asarray(vecs[:, 0])


def entanglement_entropy(psi: np.ndarray, n: int, cut: int | None = None):
    """Von Neumann entropy S (in bits) across the bipartition [0:cut | cut:n], plus the
    Schmidt spectrum. S = -Σ p log2 p with p the squared Schmidt values."""
    if cut is None:
        cut = n // 2
    M = np.asarray(psi).reshape(1 << cut, 1 << (n - cut))
    sv = np.linalg.svd(M, compute_uv=False)
    sv = sv[sv > 1e-12]
    p = sv**2
    p = p / p.sum()
    S = float(-np.sum(p * np.log2(p)))
    return S, sv


def effective_chi(schmidt: np.ndarray, tol: float = 1e-3) -> int:
    """Bond dimension a matrix-product state would keep: the number of Schmidt values
    needed to capture (1 - tol²) of the state's weight. This is the Blaze-style
    energy-budget truncation — a physical, stable measure of how much the state
    entangles, rather than counting numerically tiny tails."""
    schmidt = np.asarray(schmidt)
    if schmidt.size == 0:
        return 0
    w = np.sort(schmidt**2)[::-1]
    w = w / w.sum()
    cum = np.cumsum(w)
    return int(np.searchsorted(cum, 1.0 - tol**2) + 1)
