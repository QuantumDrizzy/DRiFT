"""Validation of drift.solve — the unified best-method dispatcher.

Checks the contract that makes it trustworthy: small problems are solved *exactly* and flagged
`certified`; larger ones fall back to a heuristic (never labelled certified) and still find the
right answer where we can check it; and the reported energy always matches the returned config.

The GPU path isn't exercised here (no GPU in CI) — it's covered by experiments/phase14_gpu.py.

Run standalone:  python tests/test_solve.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from drift.builders.qubo import maxcut_ising, random_graph  # noqa: E402
from drift.ising import IsingModel  # noqa: E402
from drift.solve import solve  # noqa: E402
from drift.solvers.exact import exact_ground_state  # noqa: E402


def test_small_is_exact_and_certified():
    """A small problem is solved by brute force — certified the true ground state."""
    model = maxcut_ising(random_graph(14, p=0.5, seed=1))
    _, e_exact, _ = exact_ground_state(model)
    sol = solve(model, exact_max=18)
    assert sol.method == "exact"
    assert sol.certified is True
    assert np.isclose(sol.energy, e_exact)
    assert np.isclose(model.energy(sol.s), sol.energy)   # energy matches the returned config


def test_large_falls_back_to_cpu_and_is_not_certified():
    """Past the exact bound with no GPU, it uses CPU parallel tempering — a strong, *uncertified*
    minimum — and still finds the exact ground energy on an instance we can check."""
    model = maxcut_ising(random_graph(20, p=0.5, seed=2))
    _, e_exact, _ = exact_ground_state(model, max_n=22)
    sol = solve(model, exact_max=16, use_gpu=False, n_replicas=24, n_rounds=200, sweeps_per_round=4)
    assert sol.method == "cpu-pt"
    assert sol.certified is False
    assert np.isclose(model.energy(sol.s), sol.energy)
    assert np.isclose(sol.energy, e_exact), f"cpu-pt {sol.energy} != exact {e_exact}"


def test_exact_bound_is_the_boundary():
    """The switch happens exactly at exact_max: n == exact_max is still certified-exact."""
    model = maxcut_ising(random_graph(12, p=0.5, seed=3))
    sol = solve(model, exact_max=12, use_gpu=False)
    assert sol.method == "exact" and sol.certified
    sol2 = solve(model, exact_max=11, use_gpu=False, n_rounds=150)
    assert sol2.method == "cpu-pt" and not sol2.certified


def test_ferromagnet_ground_state():
    """Sanity anchor through the dispatcher: a ferromagnet's ground energy is -(n-1)."""
    n = 24
    J = np.zeros((n, n))
    for i in range(n - 1):
        J[i, i + 1] = J[i + 1, i] = 1.0
    model = IsingModel(J=J, h=np.zeros(n))
    sol = solve(model, exact_max=16, use_gpu=False, n_replicas=16, n_rounds=200)
    assert np.isclose(sol.energy, -(n - 1)), f"{sol.energy} != {-(n - 1)}"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all solve tests passed")
