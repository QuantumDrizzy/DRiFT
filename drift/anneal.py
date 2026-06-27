"""
drift.anneal — Phase 9: the optimization face, run *quantum* (adiabatic annealing)
==================================================================================
Phase 2 found the Ising ground state by **thermal** relaxation (simulated annealing).
Here the *same* ground state is reached by **quantum adiabatic evolution**:

    H(s) = (1 - s) · H_driver  +  s · H_problem,        s : 0 → 1

- ``H_driver = -Σ_i X_i`` — a uniform transverse field. Its ground state is the equal
  superposition over all 2ⁿ configurations |+…+⟩ (the quantum analogue of "start hot,
  everywhere at once").
- ``H_problem`` — **diagonal in the computational basis**, with ``diag[b]`` equal to the
  *classical* Ising energy of configuration ``b``. So the quantum optimization problem is,
  by construction, exactly the classical one (the Phase-2 ground state is its ground state).

Start in |+…+⟩ at s=0 and turn on H_problem slowly. By the adiabatic theorem the state
tracks the instantaneous ground state **if** the anneal time obeys ``T ≳ 1/Δ_min²``, where
``Δ_min`` is the smallest spectral gap along the path. That is the whole honest story, in
DRIFT's microscope spirit: **the cost of quantum annealing is set by the gap — and when the
gap closes, quantum annealing fails too. No magic, just measured.**

Exact for small n (dense state vector, sparse operators); this is the regime where we can
watch the mechanism directly.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, expm_multiply

from drift.ising import IsingModel

_I2 = sp.identity(2, format="csr")
_X = sp.csr_matrix(np.array([[0.0, 1.0], [1.0, 0.0]]))


def transverse_driver(n: int) -> sp.csr_matrix:
    """``H_driver = -Σ_i X_i`` on n qubits (qubit 0 is the most significant tensor factor).
    Its unique ground state is the uniform superposition |+…+⟩ with energy -n."""
    dim = 1 << n
    H = sp.csr_matrix((dim, dim))
    for i in range(n):
        op = sp.csr_matrix(np.array([[1.0]]))
        for q in range(n):
            op = sp.kron(op, _X if q == i else _I2, format="csr")
        H = H - op
    return H.tocsr()


def problem_hamiltonian(model: IsingModel) -> tuple[sp.csr_matrix, np.ndarray]:
    """Diagonal problem Hamiltonian whose ``diag[b]`` is the classical Ising energy of basis
    state ``b`` (qubit 0 = MSB, |0⟩ ↦ spin +1). Returns ``(H_problem, energies)``.

    Building the diagonal straight from ``model.energy`` guarantees the quantum problem is
    *identical* to the classical one — no coupling/sign convention can drift between them."""
    n = model.n
    dim = 1 << n
    bit = (np.arange(dim)[:, None] >> (n - 1 - np.arange(n))[None, :]) & 1
    spins = (1 - 2 * bit).astype(np.float64)  # 0 → +1, 1 → -1
    energies = model.energy_batch(spins)
    return sp.diags(energies, format="csr"), energies


def driver_ground_state(n: int) -> np.ndarray:
    """The s=0 starting state: |+…+⟩, i.e. every amplitude equal to 1/√(2ⁿ)."""
    dim = 1 << n
    return np.full(dim, 1.0 / np.sqrt(dim), dtype=complex)


def anneal_hamiltonian(H_driver: sp.spmatrix, H_problem: sp.spmatrix, s: float) -> sp.spmatrix:
    """``H(s) = (1 - s)·H_driver + s·H_problem``."""
    return (1.0 - s) * H_driver + s * H_problem


def spectral_gap_path(
    H_driver: sp.spmatrix, H_problem: sp.spmatrix, npts: int = 41
) -> dict:
    """Two lowest eigenvalues along s ∈ [0, 1] and the minimum gap.

    Returns a dict with ``s``, ``e0``, ``e1``, ``gap`` (arrays) and the scalars
    ``gap_min`` and ``s_at_min``. Dense for small dim (exact), sparse Lanczos otherwise."""
    dim = H_driver.shape[0]
    ss = np.linspace(0.0, 1.0, npts)
    e0 = np.empty(npts)
    e1 = np.empty(npts)
    for k, s in enumerate(ss):
        H = anneal_hamiltonian(H_driver, H_problem, s)
        if dim <= 512:
            vals = np.linalg.eigvalsh(H.toarray())
        else:
            vals = np.sort(eigsh(H.tocsc(), k=2, which="SA", return_eigenvectors=False))
        e0[k], e1[k] = vals[0], vals[1]
    gap = e1 - e0
    kmin = int(np.argmin(gap))
    return {
        "s": ss,
        "e0": e0,
        "e1": e1,
        "gap": gap,
        "gap_min": float(gap[kmin]),
        "s_at_min": float(ss[kmin]),
    }


def quantum_anneal(
    H_driver: sp.spmatrix, H_problem: sp.spmatrix, total_time: float, steps: int = 200
) -> np.ndarray:
    """Evolve |+…+⟩ under the time-dependent H(t/T) for t ∈ [0, T] and return the final
    state. Each of ``steps`` segments uses a piecewise-constant midpoint Hamiltonian and a
    Krylov matrix exponential (``expm_multiply``) — exact up to the time discretisation."""
    n = int(round(np.log2(H_driver.shape[0])))
    psi = driver_ground_state(n)
    dt = total_time / steps
    for k in range(steps):
        s = (k + 0.5) / steps  # midpoint of segment k
        H = anneal_hamiltonian(H_driver, H_problem, s)
        psi = expm_multiply(-1j * dt * H, psi)
    nrm = np.linalg.norm(psi)
    return psi / nrm if nrm > 0 else psi


def success_probability(psi: np.ndarray, energies: np.ndarray, tol: float = 1e-9) -> float:
    """Probability of measuring a ground-state configuration. Because H_problem is diagonal,
    the ground subspace is spanned by the basis states at the minimum energy."""
    e_min = energies.min()
    mask = energies <= e_min + tol
    return float(np.sum(np.abs(psi[mask]) ** 2))


def expected_energy(psi: np.ndarray, energies: np.ndarray) -> float:
    """⟨ψ|H_problem|ψ⟩ — the mean problem energy of the final state."""
    return float(np.sum(np.abs(psi) ** 2 * energies))
