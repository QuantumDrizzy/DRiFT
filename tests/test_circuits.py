"""Validation of Phase 12 — universal computation as an Ising ground state.

Checks that synthesised gates compose into a working arithmetic circuit whose ground state,
found through ``drift.solve``, computes the function for every input — and that an
inconsistent evaluation genuinely costs energy (the gates are real constraints, not decoration).
Small netlists stay certified-exact; past exact reach the face reports ``certified=False``.

Run standalone:  python tests/test_circuits.py
"""

from __future__ import annotations

import itertools
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from drift.circuits import Circuit, adder_truth_table, full_adder  # noqa: E402
from drift.gpu import gpu_available  # noqa: E402
from drift.inverse_logic import qubo_ground_states, synthesize, truth_from_fn  # noqa: E402
from drift.solvers.parallel_tempering import PtResult  # noqa: E402


def test_primitive_gates_are_2local():
    """AND/OR/NOT synthesise to a QUBO whose ground states are exactly the gate's truth table."""
    for fn, n_in in [(lambda x, y: x & y, 2), (lambda x, y: x | y, 2), (lambda a: 1 - a, 1)]:
        truth = truth_from_fn(fn, n_in)
        Q, off = synthesize(truth, n_in + 1)
        assert qubo_ground_states(Q, off, n_in + 1) == truth


def test_full_adder_computes_every_input():
    """A 1-bit full adder, composed from gates, computes (sum, cout) for all 8 inputs — and the
    circuit penalty is 0 at each (every gate satisfied). Small: certified exact."""
    c = Circuit()
    full_adder(c, "a", "b", "cin", "sum", "cout")
    rows = adder_truth_table(c, "a", "b", "cin", "sum", "cout")
    assert len(rows) == 8
    for r in rows:
        assert r["ok"], r
        assert abs(r["energy"]) < 1e-6
        assert r["method"] == "exact"
        assert r["certified"] is True


def test_xor_by_composition():
    """XOR (not 2-local on its own) computed by composing AND/OR/NOT over shared wires."""
    c = Circuit()
    c.add_xor("x", "y", "z")
    for vx, vy in itertools.product((0, 1), repeat=2):
        out = c.evaluate({"x": vx, "y": vy}, ["z"])["out"]
        assert out["z"] == (vx ^ vy), (vx, vy, out)


def test_inconsistent_output_costs_energy():
    """Forcing a wrong output makes every assignment violate a gate — penalty > 0. The gates are
    real constraints: the circuit cannot 'compute' a false result for free."""
    c = Circuit()
    full_adder(c, "a", "b", "cin", "sum", "cout")
    # 0 + 0 + 0 = 0, so sum must be 0; clamp it to 1 and the penalty can no longer reach 0
    r = c.evaluate({"a": 0, "b": 0, "cin": 0, "sum": 1}, ["sum", "cout"])
    assert r["energy"] > 0.5


def test_evaluate_method_switches_past_exact_max():
    """The same 1-bit adder, with exact_max below n, is CPU-PT and not certified."""
    c = Circuit()
    full_adder(c, "a", "b", "cin", "sum", "cout")
    assert c.n > 10
    r = c.evaluate(
        {"a": 0, "b": 0, "cin": 0}, ["sum", "cout"],
        exact_max=10, use_gpu=False, n_replicas=4, n_rounds=4, sweeps_per_round=1,
    )
    assert r["method"] == "cpu-pt"
    assert r["certified"] is False


def test_scaling_is_honest():
    """A 1-bit adder is certified-exact; a 2-bit ripple adder is past exact reach.

    Heuristic evaluation is allowed (and not labelled certified). Callers who need a
    proven ground state still hit the wall via ``require_certified=True``.
    """
    one = Circuit()
    full_adder(one, "a", "b", "cin", "sum", "cout")
    r = one.evaluate({"a": 0, "b": 1, "cin": 0}, ["sum", "cout"])
    assert r["certified"] is True and r["method"] == "exact"

    two = Circuit()
    full_adder(two, "a0", "b0", "c0", "s0", "c1")
    full_adder(two, "a1", "b1", "c1", "s1", "c2")
    assert two.n > 18
    with pytest.raises(ValueError, match="certified"):
        two.evaluate(
            {"a0": 0, "b0": 0, "c0": 0, "a1": 0, "b1": 0}, ["s0", "s1", "c2"],
            require_certified=True,
        )
    r2 = two.evaluate(
        {"a0": 0, "b0": 0, "c0": 0, "a1": 0, "b1": 0}, ["s0", "s1", "c2"],
        use_gpu=False, n_replicas=2, n_rounds=2, sweeps_per_round=1,
    )
    assert r2["certified"] is False
    assert r2["method"] == "cpu-pt"


def test_evaluate_gpu_path_is_not_certified(monkeypatch):
    """Mocked GPU dispatch through Circuit.evaluate is never labelled certified."""

    def fake_pt(model, **kwargs):
        s = np.ones(model.n)
        return PtResult(
            best_s=s, best_E=float(model.energy(s)),
            temperatures=np.array([0.1]), swap_rate=0.0,
        )

    monkeypatch.setattr("drift.gpu.gpu_available", lambda: True)
    monkeypatch.setattr("drift.gpu.parallel_tempering_gpu", fake_pt)
    c = Circuit()
    c.add("NOT", "x", "z")
    r = c.evaluate({"x": 0}, ["z"], exact_max=0, use_gpu=True)
    assert r["method"] == "gpu-pt"
    assert r["certified"] is False


@pytest.mark.gpu
@pytest.mark.skipif(not gpu_available(), reason="CUDA engine binary not built")
def test_evaluate_gpu_binary_path_optional():
    c = Circuit()
    c.add("NOT", "x", "z")
    r = c.evaluate({"x": 0}, ["z"], exact_max=0, use_gpu=True, n_replicas=8, n_rounds=20)
    assert r["method"] == "gpu-pt"
    assert r["certified"] is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
