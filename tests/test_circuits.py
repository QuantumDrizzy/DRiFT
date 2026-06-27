"""Validation of Phase 12 — universal computation as an Ising ground state.

Checks that synthesised gates compose into a working arithmetic circuit whose ground state,
found by DRIFT's own Ising engine, computes the function for every input — and that an
inconsistent evaluation genuinely costs energy (the gates are real constraints, not decoration).

Run standalone:  python tests/test_circuits.py
"""

from __future__ import annotations

import itertools
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from drift.circuits import Circuit, adder_truth_table, full_adder  # noqa: E402
from drift.inverse_logic import qubo_ground_states, synthesize, truth_from_fn  # noqa: E402


def test_primitive_gates_are_2local():
    """AND/OR/NOT synthesise to a QUBO whose ground states are exactly the gate's truth table."""
    for fn, n_in in [(lambda x, y: x & y, 2), (lambda x, y: x | y, 2), (lambda a: 1 - a, 1)]:
        truth = truth_from_fn(fn, n_in)
        Q, off = synthesize(truth, n_in + 1)
        assert qubo_ground_states(Q, off, n_in + 1) == truth


def test_full_adder_computes_every_input():
    """A 1-bit full adder, composed from gates, computes (sum, cout) for all 8 inputs — and the
    circuit penalty is 0 at each (every gate satisfied)."""
    c = Circuit()
    full_adder(c, "a", "b", "cin", "sum", "cout")
    rows = adder_truth_table(c, "a", "b", "cin", "sum", "cout")
    assert len(rows) == 8
    for r in rows:
        assert r["ok"], r
        assert abs(r["energy"]) < 1e-6


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


def test_scaling_is_honest():
    """A 1-bit adder fits the exact engine; a 2-bit ripple adder does not — the same wall as
    factoring, stated honestly rather than pretended away."""
    one = Circuit()
    full_adder(one, "a", "b", "cin", "sum", "cout")
    assert one.n <= 22  # solvable exactly

    two = Circuit()
    full_adder(two, "a0", "b0", "c0", "s0", "c1")
    full_adder(two, "a1", "b1", "c1", "s1", "c2")
    assert two.n > 22  # past the exact engine — wider circuits are the principle, not solved here


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all Phase 12 tests passed")
