"""Validation of Phase 16 — graphity.

  * triangle and 4-cycle counts match brute force;
  * every Metropolis energy delta matches a from-scratch recount, for every Hamiltonian
    shape (quadratic/quartic valence, triangles, 4-cycles);
  * the valence-only anneal reaches a v₀-regular graph from the complete graph;
  * the quadratic-valence 4-cycle Hamiltonian prefers the complete graph to disjoint K4,4
    (the ADR-005 design error, kept as a regression), and the quartic one does not.

Run standalone:  python tests/test_graphity.py
"""

from __future__ import annotations

import itertools
import os
import random
import sys

import networkx as nx

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from drift import graphity as gr  # noqa: E402


def _c4_brute(g):
    c = 0
    for a, b, x, d in itertools.permutations(g.nodes(), 4):
        if g.has_edge(a, b) and g.has_edge(b, x) and g.has_edge(x, d) and g.has_edge(d, a):
            c += 1
    return c // 8


def test_cycle_counts_match_brute_force():
    for seed in range(4):
        g = nx.gnp_random_graph(10, 0.45, seed=seed)
        rows = gr.from_networkx(g)
        assert gr.triangles(rows) == sum(nx.triangles(g).values()) // 3
        assert gr.four_cycles(rows) == _c4_brute(g)


def test_energy_deltas_are_exact():
    shapes = [dict(), dict(g3=0.5), dict(g4=0.5), dict(g4=0.125, power=4)]
    rng = random.Random(1)
    for kw in shapes:
        H = gr.GraphHamiltonian(**kw)
        rows = gr.from_networkx(nx.gnp_random_graph(9, 0.5, seed=3))
        deg = gr.degrees(rows)
        for _ in range(300):
            i, j = rng.sample(range(9), 2)
            d = H.delta(rows, deg, i, j)
            e0 = H.energy(rows)
            rows[i] ^= 1 << j
            rows[j] ^= 1 << i
            deg = gr.degrees(rows)
            assert abs(H.energy(rows) - e0 - d) < 1e-9


def test_valence_only_anneal_is_regular():
    r = gr.anneal(gr.GraphHamiltonian(), n=24, sweeps=40, seed=0)
    assert r["tracked_err"] < 1e-9
    assert set(gr.degrees(r["rows"])) == {4}


def test_quadratic_valence_lets_four_cycles_win_density():
    n = 32
    full = [((1 << n) - 1) & ~(1 << i) for i in range(n)]
    k44 = gr.disjoint_blocks(n, "K44")
    quad = gr.GraphHamiltonian(g4=0.5)
    quart = gr.GraphHamiltonian(g4=0.125, power=4)
    assert quad.energy(full) < quad.energy(k44)
    assert quart.energy(k44) < quart.energy(full)


def test_connectivity_rule_holds():
    r = gr.anneal(gr.GraphHamiltonian(g4=0.125, power=4, keep_connected=True),
                  n=24, sweeps=20, seed=0)
    assert gr.components(r["rows"]) == [24]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
