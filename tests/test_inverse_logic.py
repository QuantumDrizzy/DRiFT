"""Validate inverse computronium: target truth table -> synthesised QUBO (round-trip)."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from drift.logic import TRUTH, NAND_TRUTH  # noqa: E402
from drift.solvers import simulated_annealing  # noqa: E402
from drift.inverse_logic import (  # noqa: E402
    NotQuadratic,
    minimise_qubo,
    qubo_energy,
    qubo_ground_states,
    qubo_to_ising,
    spins_to_bits,
    synthesize,
    truth_from_fn,
    xor_via_composition,
)


def _xor(x, y):
    return x ^ y


def test_round_trip_primitive_gates():
    """Synthesising from a gate's truth table recovers that exact truth table."""
    for name, n in (("AND", 3), ("OR", 3), ("NOT", 2)):
        Q, off = synthesize(TRUTH[name], n)
        assert qubo_ground_states(Q, off, n) == TRUTH[name]


def test_synthesised_penalty_is_zero_on_valid_positive_on_invalid():
    n = 3
    Q, off = synthesize(TRUTH["AND"], n)
    for x in [(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)]:
        e = qubo_energy(Q, off, x)
        if x in TRUTH["AND"]:
            assert abs(e) < 1e-6
        else:
            assert e > 0.5            # a real gap above the valid rows


def test_arbitrary_targets_round_trip():
    """Functions known to be quadratically representable synthesise and round-trip."""
    majority = truth_from_fn(lambda x, y, w: (x + y + w) >= 2, 3)
    implies = truth_from_fn(lambda x, y: (1 - x) | y, 2)        # x -> y
    for tt, n in ((majority, 4), (implies, 3), (NAND_TRUTH, 3)):
        Q, off = synthesize(tt, n)
        assert qubo_ground_states(Q, off, n) == tt


def test_xor_is_not_quadratic():
    """The textbook non-2-local case is detected, not faked (honest [KNOWN_LIMIT])."""
    with pytest.raises(NotQuadratic):
        synthesize(truth_from_fn(_xor, 2), 3)


def test_xor_recovered_by_composition():
    """XOR built by adding synthesised AND/OR/NOT penalties over wires == XOR truth table."""
    assert xor_via_composition() == truth_from_fn(_xor, 2)


def test_inverse_is_multimodal_validate_by_ground_states_not_coeffs():
    """Two valid syntheses (different gap) of one gate share ground states, not coefficients."""
    Q1, o1 = synthesize(TRUTH["OR"], 3, gap=1.0)
    Q2, o2 = synthesize(TRUTH["OR"], 3, gap=3.0)
    assert qubo_ground_states(Q1, o1, 3) == qubo_ground_states(Q2, o2, 3) == TRUTH["OR"]
    assert not np.allclose(Q1, Q2)            # the coupling matrix is not unique


def test_consume_anneal_spine_recovers_valid_row():
    """Synthesised QUBO -> Ising -> DRIFT's annealer relaxes to a valid truth-table row."""
    for name, n in (("AND", 3), ("OR", 3)):
        Q, off = synthesize(TRUTH[name], n)
        model = qubo_to_ising(Q, off)
        sol = minimise_qubo(Q, off)
        assert sol.certified is True and sol.method == "exact"
        assert spins_to_bits(sol.s) in TRUTH[name]
        s_sa, _, _, _ = simulated_annealing(model, n_sweeps=400, seed=0)
        assert spins_to_bits(s_sa) in TRUTH[name]


def test_minimise_qubo_switches_past_exact_max():
    """A heuristic minimum of a synthesised gate is not labelled certified."""
    Q, off = synthesize(TRUTH["AND"], 3)
    sol = minimise_qubo(Q, off, exact_max=2, use_gpu=False, n_replicas=8, n_rounds=40, seed=0)
    assert sol.method == "cpu-pt"
    assert sol.certified is False


def test_minimise_qubo_require_certified_raises():
    Q, off = synthesize(TRUTH["AND"], 3)
    with pytest.raises(ValueError, match="certified"):
        minimise_qubo(Q, off, exact_max=2, require_certified=True)


def test_empty_and_malformed_rejected():
    with pytest.raises(ValueError):
        synthesize(frozenset(), 2)
    with pytest.raises(ValueError):
        synthesize(TRUTH["AND"], 2)           # rows are length 3, n_vars=2
