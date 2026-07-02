"""
drift.mps — the tensor-network ground state: χ as the engine, not just the thermometer.

Phases 1–12 read ground states one of two ways: brute force over 2ⁿ configurations, or
exact Lanczos on the full 2ⁿ × 2ⁿ Hamiltonian. Both die at n ≈ 14–22 — the wall every
recent phase confesses. And yet DRIFT's thesis is that ground states are *read with tensor
networks*: in Phase 3 χ (the bond dimension) was only ever *measured*, after an exact
diagonalization. The lens was never actually the microscope.

This module makes it the microscope. It finds the ground state of the 1-D transverse-field
Ising chain

    H = -j Σ_i Z_i Z_{i+1}  -  Γ Σ_i X_i

by **imaginary-time evolution of a matrix-product state** (TEBD). e^{-τH} projects any state
with nonzero ground-state overlap onto the ground state; we Trotterize it into nearest-
neighbor two-site gates and, after each gate, re-compress the MPS by an SVD truncated to bond
dimension χ. That truncation *is* the physics: χ is exactly how much entanglement — how much
computation — the state carries. The engine is exact while χ stays small and degrades
gracefully, measurably, when it doesn't. So the honesty is built into the algorithm rather
than bolted on: the same χ that was our thermometer is now our compute budget.

Canonical-form bookkeeping is done by moving the orthogonality center with SVDs (no Λ⁻¹
inverses), which is numerically robust. Everything is real (the TFIM is a real symmetric
Hamiltonian), so no complex arithmetic is needed.

Scales to n = 64+ on a low-entanglement chain — far past the exact wall — with the cost, χ,
reported and never hidden. The large-n / higher-dimensional / GPU story is the Rust/CUDA
port; this is the honest Python reference that finally delivers what the README promised.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

from .quantum import effective_chi

# single-site operators (real)
_Z = np.array([[1.0, 0.0], [0.0, -1.0]])
_X = np.array([[0.0, 1.0], [1.0, 0.0]])
_I = np.eye(2)

Tensor = np.ndarray  # an MPS site tensor, shape (Dl, d, Dr)


# ── the Hamiltonian, split into nearest-neighbor bonds ─────────────────────────
def bond_hamiltonians(n: int, j: float = 1.0, gamma: float = 1.0) -> list[np.ndarray]:
    """The open TFIM as a list of n-1 two-site bond terms h_{i,i+1} (each 4×4).

    H = -j Σ Z_iZ_{i+1} - Γ Σ X_i. The single-site X on an interior site is shared ½/½
    between its two bonds; the two end sites live in a single bond and take their full X.
    Summed over bonds this reproduces H exactly.
    """
    zz = np.kron(_Z, _Z)
    xi = np.kron(_X, _I)
    ix = np.kron(_I, _X)
    hs = []
    for i in range(n - 1):
        wl = 1.0 if i == 0 else 0.5
        wr = 1.0 if i + 1 == n - 1 else 0.5
        hs.append(-j * zz - gamma * (wl * xi + wr * ix))
    return hs


# ── MPS construction ───────────────────────────────────────────────────────────
def product_plus_x(n: int) -> list[Tensor]:
    """The all-|+x⟩ product state — the exact ground state at Γ → ∞, a good, symmetric seed
    for imaginary-time evolution at any Γ. Bond dimension 1, zero entanglement."""
    a = np.zeros((1, 2, 1))
    a[0, :, 0] = 1.0 / np.sqrt(2.0)
    return [a.copy() for _ in range(n)]


def _apply_gate(theta: np.ndarray, gate: np.ndarray) -> np.ndarray:
    """Apply a two-site gate (d,d,d,d)=[o1,o2,i1,i2] to θ (Dl,d,d,Dr).

    out[a,i,j,b] = Σ_kl gate[i,j,k,l] θ[a,k,l,b]. Done as a plain matmul (reshape the gate to
    (d²,d²) and θ's physical legs to a matrix) — far cheaper than einsum path-finding on the
    tiny tensors this inner loop hammers millions of times.
    """
    dl, d1, d2, dr = theta.shape
    g = gate.reshape(d1 * d2, d1 * d2)                       # [(ij),(kl)]
    m = np.transpose(theta, (0, 3, 1, 2)).reshape(dl * dr, d1 * d2)  # [(ab),(kl)]
    out = m @ g.T                                            # [(ab),(ij)]
    return np.transpose(out.reshape(dl, dr, d1, d2), (0, 2, 3, 1))


def _split(theta: np.ndarray, chi_max: int, tol: float):
    """SVD-split θ (Dl,d,d,Dr) across its middle, truncating to χ. Returns (U, S, Vh, k)
    with S renormalized to unit norm (the whole state stays normalized)."""
    dl, d1, d2, dr = theta.shape
    u, s, vh = np.linalg.svd(theta.reshape(dl * d1, d2 * dr), full_matrices=False)
    cutoff = tol * s[0] if s[0] > 0 else 0.0
    k = int(np.sum(s > cutoff))
    k = max(1, min(chi_max, k))
    u, s, vh = u[:, :k], s[:k], vh[:k, :]
    nrm = np.linalg.norm(s)
    if nrm > 0:
        s = s / nrm
    return u, s, vh, k, dl, d1, d2, dr


def _sweep(a: list[Tensor], gates: list[np.ndarray], chi_max: int, tol: float) -> int:
    """One back-and-forth TEBD sweep (left→right then right→left), applying every bond gate
    on each pass. Keeps the MPS in canonical form via center-moving SVDs. Returns max χ used.

    On entry the orthogonality center sits at site 0; it returns there.
    """
    n = len(a)
    chi_used = 1
    # right-moving: leave left-canonical U behind, carry the center rightward
    for i in range(n - 1):
        theta = _apply_gate(np.tensordot(a[i], a[i + 1], axes=(2, 0)), gates[i])
        u, s, vh, k, dl, d1, d2, dr = _split(theta, chi_max, tol)
        a[i] = u.reshape(dl, d1, k)
        a[i + 1] = (np.diag(s) @ vh).reshape(k, d2, dr)
        chi_used = max(chi_used, k)
    # left-moving: leave right-canonical Vh behind, carry the center leftward
    for i in range(n - 2, -1, -1):
        theta = _apply_gate(np.tensordot(a[i], a[i + 1], axes=(2, 0)), gates[i])
        u, s, vh, k, dl, d1, d2, dr = _split(theta, chi_max, tol)
        a[i] = (u @ np.diag(s)).reshape(dl, d1, k)
        a[i + 1] = vh.reshape(k, d2, dr)
        chi_used = max(chi_used, k)
    return chi_used


# ── observables (require canonical form: center at site 0) ─────────────────────
def energy(a: list[Tensor], hs: list[np.ndarray]) -> float:
    """⟨H⟩ = Σ_i ⟨h_{i,i+1}⟩. Sweeps the orthogonality center rightward; at each bond the
    left environment is left-canonical and the right is right-canonical, so ⟨θ|θ⟩ = 1 and
    the local contraction is exactly the bond energy. Non-destructive (works on a copy)."""
    a = [t.copy() for t in a]
    n = len(a)
    e = 0.0
    for i in range(n - 1):
        theta = np.tensordot(a[i], a[i + 1], axes=(2, 0))  # (Dl,d,d,Dr)
        hth = _apply_gate(theta, hs[i].reshape(2, 2, 2, 2))
        e += float(np.real(np.vdot(theta, hth) / np.vdot(theta, theta)))
        dl, d1, d2, dr = theta.shape
        u, s, vh = np.linalg.svd(theta.reshape(dl * d1, d2 * dr), full_matrices=False)
        s = s / np.linalg.norm(s)
        a[i] = u.reshape(dl, d1, -1)
        a[i + 1] = (np.diag(s) @ vh).reshape(-1, d2, dr)
    return e


def _schmidt_scan(a: list[Tensor]) -> list[np.ndarray]:
    """The (renormalized) Schmidt spectrum across every bond of a canonical (center-0) MPS.
    One rightward center-moving pass; everything else is derived from these spectra."""
    a = [t.copy() for t in a]
    n = len(a)
    spectra = []
    for i in range(n - 1):
        theta = np.tensordot(a[i], a[i + 1], axes=(2, 0))
        dl, d1, d2, dr = theta.shape
        u, s, vh = np.linalg.svd(theta.reshape(dl * d1, d2 * dr), full_matrices=False)
        s = s / np.linalg.norm(s)
        spectra.append(s)
        a[i] = u.reshape(dl, d1, -1)
        a[i + 1] = (np.diag(s) @ vh).reshape(-1, d2, dr)
    return spectra


def _entropy(schmidt: np.ndarray) -> float:
    """Von Neumann entanglement entropy (bits) of one Schmidt spectrum."""
    p = schmidt**2
    p = p[p > 1e-15]
    return float(-np.sum(p * np.log2(p)))


def bond_entropies(a: list[Tensor]) -> list[float]:
    """Entanglement entropy (bits) across every bond of a canonical MPS. The peak of this
    profile is the entanglement the ground state actually carries."""
    return [_entropy(s) for s in _schmidt_scan(a)]


# ── the solver ─────────────────────────────────────────────────────────────────
@dataclass
class MpsResult:
    """The outcome of a tensor-network ground-state search."""

    energy: float          # ⟨H⟩ of the converged MPS (variational upper bound on E0)
    chi: int               # effective bond dimension at the center bond — the SAME Phase-3
                           # ruler (effective_chi, tol=1e-3), now on the solver's own state
    chi_kept: int          # raw bond dimension the truncation physically kept (memory cost)
    max_entropy: float     # peak entanglement entropy across the chain (bits)
    n: int
    gamma: float
    mps: list[Tensor]

    @property
    def energy_per_spin(self) -> float:
        return self.energy / self.n


def ground_state(
    n: int,
    j: float = 1.0,
    gamma: float = 1.0,
    chi_max: int = 32,
    tol: float = 1e-10,
    max_sweeps: int = 25,
    econv: float = 1e-6,
    dt_schedule: list[float] | None = None,
) -> MpsResult:
    """Ground state of the open TFIM chain by imaginary-time TEBD on an MPS.

    Runs a schedule of decreasing imaginary-time steps; at each step it sweeps until the
    energy stops moving (< econv) or `max_sweeps` is hit. Smaller final dt ⇒ smaller Trotter
    error. The returned energy is a **variational upper bound** on the true ground energy,
    exact up to the χ truncation and the final Trotter step.

    χ (`.chi`) is the honest cost: the state is captured exactly while χ ≤ chi_max, and any
    shortfall shows up as χ pinned at chi_max with rising `.max_entropy`.
    """
    hs = bond_hamiltonians(n, j, gamma)
    a = product_plus_x(n)
    if dt_schedule is None:
        dt_schedule = [0.2, 0.05, 0.01, 0.002]

    e = energy(a, hs)
    chi_kept = 1
    for dt in dt_schedule:
        gates = [expm(-dt * h).reshape(2, 2, 2, 2) for h in hs]
        for _ in range(max_sweeps):
            chi_kept = _sweep(a, gates, chi_max, tol)
            e_new = energy(a, hs)
            # relative convergence: at criticality the gap is small (critical slowing down),
            # so an absolute floor would spin forever — plateau at each dt, then refine dt.
            if abs(e_new - e) < econv * max(1.0, abs(e_new)):
                e = e_new
                break
            e = e_new

    spectra = _schmidt_scan(a)
    center = spectra[len(spectra) // 2]
    return MpsResult(
        energy=e,
        chi=effective_chi(center, tol=1e-3),
        chi_kept=chi_kept,
        max_entropy=max(_entropy(s) for s in spectra) if spectra else 0.0,
        n=n,
        gamma=gamma,
        mps=a,
    )
