"""Validation of Phase 14's CPU reference — replica-exchange (parallel tempering).

The reference is what "correct" means for the GPU port, so every check pins it to ground truth:
  * on instances the exact engine can still solve, PT finds the exact ground energy — MaxCut,
    a frustrated spin glass, and a ferromagnet;
  * replica exchange genuinely helps: on a rugged landscape PT reaches the true minimum where a
    single cold walker (no swaps) gets stuck;
  * the temperature ladder is healthy (ascending, with a non-degenerate swap rate).

These are the same falsifiers the CUDA engine must pass on its first GPU run.

Run standalone:  python tests/test_parallel_tempering.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from drift.builders.qubo import maxcut_ising, random_graph  # noqa: E402
from drift.ising import IsingModel  # noqa: E402
from drift.solvers.exact import exact_ground_state  # noqa: E402
from drift.solvers.parallel_tempering import parallel_tempering, geometric_ladder  # noqa: E402


def _ferro_chain(n: int) -> IsingModel:
    J = np.zeros((n, n))
    for i in range(n - 1):
        J[i, i + 1] = J[i + 1, i] = 1.0
    return IsingModel(J=J, h=np.zeros(n))


def test_finds_exact_maxcut():
    """On a random MaxCut the coldest replica reaches the exact maximum cut (= ground state)."""
    W = random_graph(16, p=0.5, seed=3)
    model = maxcut_ising(W)
    _, e_exact, _ = exact_ground_state(model)
    res = parallel_tempering(model, n_replicas=16, T_min=0.05, T_max=4.0,
                             n_rounds=200, sweeps_per_round=4, seed=1)
    assert np.isclose(res.best_E, e_exact), f"PT {res.best_E} != exact {e_exact}"


def test_finds_exact_spin_glass():
    """A frustrated ±J spin glass (the hard case) — PT still lands on the exact ground energy."""
    rng = np.random.default_rng(7)
    n = 14
    J = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            J[i, j] = J[j, i] = rng.choice([-1.0, 1.0])
    model = IsingModel(J=J, h=rng.normal(scale=0.3, size=n))
    _, e_exact, _ = exact_ground_state(model)
    res = parallel_tempering(model, n_replicas=20, T_min=0.05, T_max=5.0,
                             n_rounds=300, sweeps_per_round=5, seed=2)
    assert np.isclose(res.best_E, e_exact), f"PT {res.best_E} != exact {e_exact}"


def test_finds_ferromagnet_ground_state():
    """The sanity anchor: a ferromagnet's ground energy is -(n-1), reached exactly."""
    n = 18
    model = _ferro_chain(n)
    res = parallel_tempering(model, n_replicas=12, T_min=0.05, T_max=3.0,
                             n_rounds=150, sweeps_per_round=4, seed=0)
    assert np.isclose(res.best_E, -(n - 1)), f"PT {res.best_E} != {-(n - 1)}"


def test_exchange_beats_a_single_cold_walker():
    """Replica exchange is the point: on a frustrated glass, PT reaches the exact minimum while a
    lone cold walker (a degenerate one-replica ladder, no swaps possible) freezes above it."""
    rng = np.random.default_rng(11)
    n = 16
    J = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            J[i, j] = J[j, i] = rng.choice([-1.0, 1.0])
    model = IsingModel(J=J, h=np.zeros(n))
    _, e_exact, _ = exact_ground_state(model)

    full = parallel_tempering(model, n_replicas=20, T_min=0.05, T_max=5.0,
                              n_rounds=300, sweeps_per_round=5, seed=4)
    cold = parallel_tempering(model, n_replicas=1, T_min=0.05, T_max=0.05,
                              n_rounds=300, sweeps_per_round=5, seed=4)

    assert np.isclose(full.best_E, e_exact), f"full PT {full.best_E} != exact {e_exact}"
    assert cold.best_E > e_exact + 1e-9, "a single cold walker should get stuck above the ground"


def test_ladder_is_healthy():
    """The ladder ascends and mixes: temperatures sorted, swap rate strictly between 0 and 1."""
    temps = geometric_ladder(16, 0.1, 5.0)
    assert np.all(np.diff(temps) > 0), "temperature ladder must be strictly ascending"
    model = maxcut_ising(random_graph(20, p=0.4, seed=5))
    res = parallel_tempering(model, n_replicas=16, n_rounds=120, seed=6)
    assert 0.0 < res.swap_rate < 1.0, f"unhealthy swap rate {res.swap_rate}"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all Phase 14 CPU-reference tests passed")
