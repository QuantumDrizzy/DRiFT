"""Validation of Phase 11 — integer factorization as an Ising ground state.

Checks that DRIFT's own ground-state engine recovers the factors (matter computing arithmetic),
that the energy is exactly 0 only at a valid factorization (the falsifier), that the QUBO's own
ground state is the factorization independent of the Ising path, and that past exact reach the
face goes through ``drift.solve`` with an honest ``certified=False`` rather than pretending.

Run standalone:  python tests/test_factoring.py
"""

from __future__ import annotations

import itertools
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from drift.factoring import decode, factor, factoring_qubo, n_qubo_vars  # noqa: E402
from drift.gpu import gpu_available  # noqa: E402
from drift.solvers.parallel_tempering import PtResult  # noqa: E402

# semiprime -> tight (p_bits, q_bits) that fit its factors
CASES = {15: (2, 3), 21: (2, 3), 35: (3, 3), 77: (3, 4), 143: (4, 4)}


def test_recovers_factors():
    """The ground state of the factorization QUBO recovers the true factors of each semiprime."""
    for N, (pb, qb) in CASES.items():
        r = factor(N, pb, qb)
        assert r["ok"], f"N={N}: got ({r['p']},{r['q']})"
        assert r["p"] * r["q"] == N
        assert 1 < r["p"] < N and 1 < r["q"] < N  # non-trivial factors
        assert r["certified"] is True
        assert r["method"] == "exact"


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


def test_small_factor_is_certified_exact():
    """Tight-width 15 is well inside exact reach — certified, method exact, true factors."""
    r = factor(15, 2, 3)
    assert r["method"] == "exact"
    assert r["certified"] is True
    assert r["ok"] and r["p"] * r["q"] == 15
    assert r["n"] == n_qubo_vars(2, 3)


def test_method_switches_past_exact_max():
    """The same small instance, with exact_max below n, is CPU-PT and not certified."""
    r = factor(15, 2, 3, exact_max=4, use_gpu=False, n_replicas=8, n_rounds=40, seed=0)
    assert r["method"] == "cpu-pt"
    assert r["certified"] is False
    assert r["n"] == n_qubo_vars(2, 3)


def test_past_exact_reach_does_not_raise_or_pretend():
    """Default-width 143 needs 23 vars — past exact, so heuristic provenance, not a raise."""
    assert n_qubo_vars(4, 6) == 23  # factor(143) default widths
    r = factor(143, use_gpu=False, n_replicas=2, n_rounds=2, sweeps_per_round=1)
    assert r["certified"] is False
    assert r["method"] == "cpu-pt"
    assert r["n"] == 23
    assert "p" in r and "q" in r
    # do not assert r["ok"]: a 2-round walk is not a pretend factorization


def test_require_certified_restores_the_exact_wall():
    """Callers who need a proven ground state still get a loud refusal past exact reach."""
    with pytest.raises(ValueError, match="certified"):
        factor(143, require_certified=True)


def test_gpu_path_is_not_certified(monkeypatch):
    """When the GPU probe is True, the face reports gpu-pt and certified=False (mocked)."""

    def fake_pt(model, **kwargs):
        s = np.ones(model.n)
        return PtResult(
            best_s=s, best_E=float(model.energy(s)),
            temperatures=np.array([0.1]), swap_rate=0.0,
        )

    monkeypatch.setattr("drift.gpu.gpu_available", lambda: True)
    monkeypatch.setattr("drift.gpu.parallel_tempering_gpu", fake_pt)
    r = factor(15, 2, 3, exact_max=0, use_gpu=True)
    assert r["method"] == "gpu-pt"
    assert r["certified"] is False


@pytest.mark.gpu
@pytest.mark.skipif(not gpu_available(), reason="CUDA engine binary not built")
def test_factor_gpu_binary_path_optional():
    """On-device: past exact_max, factor() uses gpu-pt and does not claim certified."""
    r = factor(15, 2, 3, exact_max=0, use_gpu=True, n_replicas=16, n_rounds=40, seed=1)
    assert r["method"] == "gpu-pt"
    assert r["certified"] is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
