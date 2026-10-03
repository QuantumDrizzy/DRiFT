"""Validation of Phase 15 — the drawing is the function.

  * the exhaustive, degeneracy-aware scorer agrees with ``drift.solve`` on the intact adder;
  * deleting the a–b coupling is invisible with the inputs clamped but breaks the drawing
    run backwards — the "redundant line" is a property of the clamp;
  * rewiring preserves every graph statistic it claims to (degree sequence, weights);
  * at β → 0 the drawing does nothing (P = ¼, ΔS = 0); at large β it computes, and the
    heat it dumps covers its Landauer bill.

Run standalone:  python tests/test_drawing.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from drift import drawing as dr  # noqa: E402
from drift.circuits import adder_truth_table  # noqa: E402


def _setup():
    c, h, W, _ = dr.adder_drawing()
    return c, h, W, dr.AdderScorer(c._idx)


def test_intact_adder_matches_solve():
    c, h, W, sc = _setup()
    assert sc.score(h, W) == 8
    ref = adder_truth_table(c, *dr.ADDER_INPUTS, *dr.ADDER_OUTPUTS, require_certified=True)
    assert all(r["ok"] for r in ref)


def test_input_input_line_is_redundant_only_under_the_clamp():
    c, h, W, sc = _setup()
    ia, ib = sorted((c._idx["a"], c._idx["b"]))
    Wd = W.copy()
    Wd[ia, ib] = 0.0
    assert sc.score(h, Wd) == 8                         # forward: harmless
    assert sc.ground_set(h, Wd) != sc.ground_set(h, W)  # every direction: broken
    assert len(sc.ground_set(h, W)) == 8                # the 8 consistent evaluations


def test_most_single_deletions_break_it():
    c, h, W, sc = _setup()
    edges = dr.edges_of(W)
    broken = sum(sc.score(h, dr.matrix_of(c.n, [x for x in edges if x is not e])) < 8
                 for e in edges)
    assert broken == len(edges) - 1


def test_rewire_preserves_the_look():
    c, _, W, _ = _setup()
    edges = dr.edges_of(W)
    rng = np.random.default_rng(0)
    for k in (1, 5, 250):
        r = dr.rewire(edges, k, rng)
        assert dr.degree_sequence(c.n, r) == dr.degree_sequence(c.n, edges)
        assert sorted(w for *_, w in r) == sorted(w for *_, w in edges)


def test_coupling_against_temperature():
    _, h, W, sc = _setup()
    cold, hot = dr.boltzmann_adder(sc, h, W, 100.0), dr.boltzmann_adder(sc, h, W, 1e-3)
    assert abs(hot["p_correct"] - 0.25) < 1e-3 and hot["dS_bits"] < 1e-3
    assert cold["p_correct"] > 0.999 and abs(cold["dS_bits"] - 11.0) < 1e-6
    assert cold["clausius"] >= 1.0 and hot["clausius"] >= 1.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
