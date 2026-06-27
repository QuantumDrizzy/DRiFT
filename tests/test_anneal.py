"""Validation of Phase 9 — the optimization face run quantum (adiabatic annealing).

Checks the mechanism and its honest limit:
  * the driver's ground state is the uniform superposition (the correct s=0 start);
  * the diagonal problem Hamiltonian *is* the classical Ising problem;
  * a slow anneal reaches the true ground state, a sudden quench does not (adiabaticity);
  * a smaller spectral gap is genuinely harder — quantum annealing pays the gap, no magic.

Run standalone:  python tests/test_anneal.py
"""

from __future__ import annotations

import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from drift.anneal import (  # noqa: E402
    driver_ground_state,
    problem_hamiltonian,
    quantum_anneal,
    spectral_gap_path,
    success_probability,
    transverse_driver,
)
from drift.ising import IsingModel  # noqa: E402


def _ferro_chain(n: int, field: float) -> IsingModel:
    J = np.zeros((n, n))
    for i in range(n - 1):
        J[i, i + 1] = J[i + 1, i] = 1.0
    return IsingModel(J=J, h=np.full(n, field))


def test_driver_ground_state_is_uniform():
    """H_driver = -Σ X_i has ground energy -n and the uniform superposition as ground state."""
    n = 6
    H = transverse_driver(n)
    psi = driver_ground_state(n)
    energy = float((psi.conj() @ (H @ psi)).real)
    assert np.isclose(energy, -n), f"driver ground energy {energy} != {-n}"
    # every amplitude equal and real-positive
    assert np.allclose(np.abs(psi) ** 2, 1.0 / (1 << n))
    vals = np.linalg.eigvalsh(H.toarray())
    assert np.isclose(vals[0], -n), "uniform state is not the true ground state of the driver"


def test_problem_diagonal_is_the_classical_ising():
    """The diagonal of H_problem equals the classical Ising energy of every configuration,
    so the quantum optimum is the classical optimum — exactly."""
    n = 5
    rng = np.random.default_rng(0)
    J = rng.normal(size=(n, n))
    model = IsingModel(J=J, h=rng.normal(size=n))
    _, energies = problem_hamiltonian(model)
    # brute force in the same basis convention (qubit 0 = MSB, |0> -> +1)
    for b in range(1 << n):
        spins = np.array([1 - 2 * ((b >> (n - 1 - i)) & 1) for i in range(n)], dtype=float)
        assert np.isclose(energies[b], model.energy(spins))
    bf_min = min(
        model.energy(np.array(s, dtype=float))
        for s in itertools.product((1, -1), repeat=n)
    )
    assert np.isclose(energies.min(), bf_min)


def test_slow_anneal_reaches_ground_state():
    """A slow anneal on the open-gap problem finds the true ground state with high probability."""
    n = 8
    model = _ferro_chain(n, 0.40)  # ferro + positive field -> unique all-up ground state
    H_problem, energies = problem_hamiltonian(model)
    psi = quantum_anneal(transverse_driver(n), H_problem, total_time=50.0, steps=200)
    assert success_probability(psi, energies) >= 0.9
    # the most probable configuration is the classical ground state (all spins +1 -> index 0)
    assert int(np.argmax(np.abs(psi) ** 2)) == int(np.argmin(energies)) == 0


def test_sudden_quench_does_not_solve():
    """A sudden quench (tiny T) is diabatic — it stays near the uniform start, not the ground."""
    n = 8
    model = _ferro_chain(n, 0.40)
    H_problem, energies = problem_hamiltonian(model)
    psi = quantum_anneal(transverse_driver(n), H_problem, total_time=0.4, steps=60)
    assert success_probability(psi, energies) < 0.1


def test_smaller_gap_is_harder():
    """The honest limit: a smaller minimum gap is genuinely harder. The closing-gap problem
    has both a smaller Δ_min and a lower success at the same anneal time."""
    n = 8
    driver = transverse_driver(n)
    easy = _ferro_chain(n, 0.40)
    hard = _ferro_chain(n, 0.04)

    He, Ee = problem_hamiltonian(easy)
    Hh, Eh = problem_hamiltonian(hard)
    gap_easy = spectral_gap_path(driver, He, npts=31)["gap_min"]
    gap_hard = spectral_gap_path(driver, Hh, npts=31)["gap_min"]
    assert gap_hard < gap_easy, f"expected a smaller gap for the harder problem ({gap_hard} !< {gap_easy})"

    succ_easy = success_probability(quantum_anneal(driver, He, 20.0, steps=200), Ee)
    succ_hard = success_probability(quantum_anneal(driver, Hh, 20.0, steps=200), Eh)
    assert succ_hard < succ_easy, f"smaller gap should be harder ({succ_hard} !< {succ_easy})"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all Phase 9 tests passed")
