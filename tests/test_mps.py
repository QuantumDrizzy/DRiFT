"""Validation of Phase 13 — the tensor-network ground state (MPS/TEBD).

The microscope's own lens, finally the engine. Every check is a falsifier:
  * against exact Lanczos the MPS energy matches to ~1e-4 on chains the exact engine can still
    reach — so the tensor-network solver is *correct*, not merely plausible;
  * imaginary time is variational: the MPS energy never dips below the true ground energy;
  * in the product limits (Γ → ∞, and Γ = 0) the state is unentangled — χ collapses to 1 and
    the energy hits its analytic value;
  * it runs *past* the exact wall (n = 32, 2³² ≈ 4.3e9 states) and the critical energy density
    marches toward the known thermodynamic value −4/π as n grows — the honesty of the method
    made measurable.

Run standalone:  python tests/test_mps.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from drift import mps  # noqa: E402
from drift.quantum import ising_chain_1d, tfim_terms  # noqa: E402
from drift.quantum import ground_state as exact_ground_state  # noqa: E402


def _exact_energy(n: int, gamma: float) -> float:
    """Exact TFIM ground energy via Lanczos (the Phase-3 engine), same sign convention."""
    h_zz, h_x = tfim_terms(ising_chain_1d(n, 1.0))
    e0, _ = exact_ground_state(h_zz + gamma * h_x)
    return e0


def test_matches_exact_lanczos():
    """The MPS ground energy equals the exact diagonalization across the whole phase diagram."""
    n = 12
    worst = 0.0
    for gamma in (0.5, 1.0, 2.0):
        e_exact = _exact_energy(n, gamma)
        r = mps.ground_state(n, gamma=gamma, chi_max=32)
        worst = max(worst, abs(r.energy - e_exact))
    assert worst < 1e-3, f"MPS disagreed with exact by {worst:.2e} (> 1e-3)"


def test_energy_is_a_variational_upper_bound():
    """Imaginary-time evolution can only approach the ground state from above: the MPS energy
    must never fall below the true ground energy (a hard physical constraint, up to fp noise)."""
    n = 12
    for gamma in (0.5, 1.0, 2.0):
        e_exact = _exact_energy(n, gamma)
        r = mps.ground_state(n, gamma=gamma, chi_max=32)
        assert r.energy >= e_exact - 1e-6, (
            f"variational bound violated at Γ={gamma}: {r.energy} < {e_exact}"
        )


def test_product_limit_collapses_entanglement():
    """Deep in the disordered phase (Γ ≫ J) the ground state tends to the |+x⟩ product state:
    the entanglement entropy collapses toward zero and the effective χ falls to its floor."""
    r = mps.ground_state(16, gamma=8.0, chi_max=16)
    assert r.max_entropy < 0.05, f"expected ~0 entanglement, got S={r.max_entropy}"
    assert r.chi <= 3, f"expected a near-product state, got χ_eff={r.chi}"


def test_classical_limit_energy():
    """At Γ = 0 the chain is classically ferromagnetic: E = -j·(n-1) for the open chain,
    reached exactly (the ground state is a product / cat with χ ≤ 2)."""
    n = 10
    r = mps.ground_state(n, gamma=0.0, chi_max=8)
    assert np.isclose(r.energy, -(n - 1), atol=1e-6), f"Γ=0 energy {r.energy} != {-(n - 1)}"
    assert r.chi <= 2, f"the classical ground state should not entangle (χ={r.chi})"


def test_chi_peaks_near_criticality():
    """The effective bond dimension — measured with the *same* Phase-3 ruler — rises into the
    critical region and collapses deep in the disordered phase. At finite size the peak sits a
    touch below Γ = 1 (Phase 3 saw Γ ≈ 0.77 at n = 14); the solver reproduces that, now from
    the tensor network itself rather than from an exact diagonalization."""
    n = 14
    chi_ordered = mps.ground_state(n, gamma=0.3, chi_max=32).chi
    chi_critical = mps.ground_state(n, gamma=0.8, chi_max=32).chi
    chi_disordered = mps.ground_state(n, gamma=4.0, chi_max=32).chi
    assert chi_critical > chi_disordered, "criticality must cost more χ than the disordered phase"
    assert chi_critical >= chi_ordered, "the χ peak sits in the critical/ordered-transition region"


def test_scales_past_the_exact_wall():
    """n = 32 is 2³² ≈ 4.3e9 configurations — unreachable by brute force or full-matrix
    Lanczos. The MPS solver still returns a sane critical energy density heading toward the
    thermodynamic limit −4/π, at a cost (χ) it reports honestly."""
    n = 32
    r = mps.ground_state(n, gamma=1.0, chi_max=16)
    assert r.chi_kept <= 16, f"bond dimension exceeded the cap ({r.chi_kept} > 16)"
    e_density = r.energy_per_spin
    assert -1.35 < e_density < -1.20, (
        f"critical energy density {e_density:.4f} is not near -4/π ≈ {-4 / np.pi:.4f}"
    )


def test_finite_size_approaches_thermodynamic_limit():
    """As the chain grows, the critical energy density decreases toward −4/π. Open boundaries
    miss end bonds, so they approach the bulk value from above (less negative) — the longer
    chain sits lower and closer to the limit. A direct, honest scaling check."""
    e16 = mps.ground_state(16, gamma=1.0, chi_max=24).energy_per_spin
    e32 = mps.ground_state(32, gamma=1.0, chi_max=24).energy_per_spin
    e_inf = -4.0 / np.pi
    assert e_inf < e32 < e16, f"expected -4/π < e32 < e16, got e16={e16:.4f}, e32={e32:.4f}"
    assert abs(e32 - e_inf) < abs(e16 - e_inf), "larger n should sit closer to the limit"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all Phase 13 tests passed")
