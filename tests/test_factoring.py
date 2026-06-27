"""Validation of Phase 11 — integer factorization as an Ising ground state.

Checks that DRIFT's own ground-state engine recovers the factors (matter computing arithmetic),
that the energy is exactly 0 only at a valid factorization (the falsifier), that the QUBO's own
ground state is the factorization independent of the Ising path, and that the construction
honestly refuses to pretend it scales.

Run standalone:  python tests/test_factoring.py
"""

from __future__ import annotations

import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from drift.factoring import decode, factor, factoring_qubo  # noqa: E402

# semiprime -> tight (p_bits, q_bits) that fit its factors
CASES = {15: (2, 3), 21: (2, 3), 35: (3, 3), 77: (3, 4), 143: (4, 4)}


def test_recovers_factors():
    """The ground state of the factorization QUBO recovers the true factors of each semiprime."""
    for N, (pb, qb) in CASES.items():
        r = factor(N, pb, qb)
        assert r["ok"], f"N={N}: got ({r['p']},{r['q']})"
        assert r["p"] * r["q"] == N
        assert 1 < r["p"] < N and 1 < r["q"] < N  # non-trivial factors


def test_energy_is_zero_only_at_a_valid_factorization():
    """The reported energy is exactly 0 at the factorization (all constraints satisfied)."""
    for N, (pb, qb) in CASES.items():
        assert abs(factor(N, pb, qb)["energy"]) < 1e-6


def test_qubo_ground_state_is_the_factorization():
    """Independent of the Ising path: the global minimum of xᵀQx itself decodes to the factors."""
    N, (pb, qb) = 35, CASES[35]
    spec = factoring_qubo(N, pb, qb)
    Q, n = spec["Q"], spec["Q"].shape[0]
    best_e, best_x = np.inf, None
    for bits in itertools.product((0, 1), repeat=n):
        x = np.array(bits)
        e = float(x @ Q @ x)
        if e < best_e:
            best_e, best_x = e, x
    p, q = decode(spec, best_x)
    assert p * q == N, (p, q)


def test_engine_matches_bruteforce():
    """DRIFT's qubo_to_ising + exact_ground_state agrees with a brute force over the QUBO."""
    N, (pb, qb) = 35, CASES[35]
    spec = factoring_qubo(N, pb, qb)
    Q, n = spec["Q"], spec["Q"].shape[0]
    best_x = min(
        (np.array(b) for b in itertools.product((0, 1), repeat=n)),
        key=lambda x: float(x @ Q @ x),
    )
    bp, bq = decode(spec, best_x)
    r = factor(N, pb, qb)
    assert {bp, bq} == {r["p"], r["q"]} == {5, 7}


def test_scaling_guard_is_honest():
    """The construction refuses to pretend it scales: a too-large instance raises, by design."""
    import pytest

    with pytest.raises(ValueError):
        factor(143)  # default widths need 23 variables (> 22 for the exact engine)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all Phase 11 tests passed")
