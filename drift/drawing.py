"""
drift.drawing — Phase 15: the drawing is the function (and only while it couples).

Phase 12 compiles a Boolean circuit into a coupling matrix whose ground state *is* the
circuit's output. Read the other way round, the coupling graph is a **drawing that
computes**: wires are nodes, couplings are traces, coupling strengths are line weights.
That makes three questions measurable that are usually only argued about:

  * **fragility** — does deleting or moving one line break the function?
  * **look-alikes** — does a drawing with every graph statistic of the original (node and
    edge count, degree sequence, weight multiset) compute the same thing?
  * **coupling vs kT** — does the drawing do anything when its coupling is small against
    the temperature?

Scoring is exhaustive and degeneracy-aware: inputs are fixed by selecting configurations
(no clamp constant enters), and a row counts as correct only if *every* minimum-energy
configuration carries the right output — a tie between a right and a wrong answer is wrong.

The measured answers, with pre-registered predictions, are in rse-hpc-lab exercise 11
(ADR-004): see ``docs/results/PHASE15-results.md``.
"""

from __future__ import annotations

import itertools
import math

import numpy as np

from .circuits import Circuit, full_adder

ADDER_INPUTS = ("a", "b", "cin")
ADDER_OUTPUTS = ("s", "cout")
TOL = 1e-6


def adder_drawing():
    """DRiFT's 1-bit full adder as a drawing: (circuit, fields h, couplings W, offset)."""
    c = Circuit()
    full_adder(c, *ADDER_INPUTS, *ADDER_OUTPUTS)
    Q, offset = c._total_qubo()
    return c, np.diag(Q).copy(), np.triu(Q, 1), offset


def edges_of(W: np.ndarray) -> list[tuple[int, int, float]]:
    iu, ju = np.nonzero(W)
    return [(int(i), int(j), float(W[i, j])) for i, j in zip(iu, ju)]


def matrix_of(n: int, edges) -> np.ndarray:
    W = np.zeros((n, n))
    for i, j, w in edges:
        a, b = (i, j) if i < j else (j, i)
        W[a, b] += w
    return W


class AdderScorer:
    """Exhaustive truth-table scoring of any (h, W) over the adder's wires."""

    def __init__(self, idx: dict[str, int]):
        n = len(idx)
        self.n = n
        self.X = np.array(list(itertools.product((0, 1), repeat=n)), dtype=np.float64)
        ia = [idx[w] for w in ADDER_INPUTS]
        self.io = [idx[w] for w in ADDER_OUTPUTS]
        self.rows = []
        for vals in itertools.product((0, 1), repeat=3):
            sel = np.nonzero(np.all(self.X[:, ia] == np.array(vals), axis=1))[0]
            total = sum(vals)
            want = np.array([total & 1, total >> 1], dtype=np.float64)
            right = np.all(self.X[:, self.io] == want, axis=1)
            self.rows.append((vals, sel, right))

    def energies(self, h, W) -> np.ndarray:
        return self.X @ h + np.einsum("ki,ij,kj->k", self.X, W, self.X)

    def score(self, h, W, detail: bool = False):
        """Rows (of 8) computed correctly; with ``detail``, also rows whose ground state
        does not determine the output."""
        E = self.energies(h, W)
        correct = ambiguous = 0
        for _, sel, right in self.rows:
            Es = E[sel]
            ground = sel[Es <= Es.min() + TOL]
            correct += int(np.all(right[ground]))
            ambiguous += int(len({tuple(r) for r in self.X[ground][:, self.io]}) > 1)
        return (correct, ambiguous) if detail else correct

    def ground_set(self, h, W) -> frozenset:
        """Zero-energy set of the *unclamped* penalty: the drawing run in every direction."""
        E = self.energies(h, W)
        return frozenset(np.nonzero(E <= E.min() + TOL)[0].tolist())


def degree_sequence(n: int, edges) -> list[int]:
    d = [0] * n
    for i, j, _ in edges:
        d[i] += 1
        d[j] += 1
    return d


def rewire(edges, k: int, rng: np.random.Generator):
    """k successful Maslov–Sneppen double-edge swaps; weights travel with their edge, so
    degree sequence and weight multiset are preserved exactly."""
    edges = list(edges)
    present = {(min(i, j), max(i, j)) for i, j, _ in edges}
    done = attempts = 0
    while done < k:
        attempts += 1
        if attempts > 10000 * k:
            raise RuntimeError("rewiring stalled")
        p, q = rng.choice(len(edges), size=2, replace=False)
        a, b, w1 = edges[p]
        c, d, w2 = edges[q]
        if rng.random() < 0.5:
            c, d = d, c
        if len({a, b, c, d}) < 4:
            continue
        n1, n2 = (min(a, d), max(a, d)), (min(c, b), max(c, b))
        if n1 in present or n2 in present:
            continue
        present -= {(min(a, b), max(a, b)), (min(c, d), max(c, d))}
        present |= {n1, n2}
        edges[p] = (a, d, w1)
        edges[q] = (c, b, w2)
        done += 1
    return edges


def boltzmann_adder(scorer: AdderScorer, h, W, beta: float) -> dict:
    """At inverse temperature beta: P(correct output), bits the drawing removes from the free
    system, and the heat a quench-and-relax dumps against its Landauer bill."""
    E = scorer.energies(h, W)
    E = E - E.min()
    pc = []
    for _, sel, right in scorer.rows:
        Es = E[sel]
        w = np.exp(-beta * (Es - Es.min()))
        pc.append(float(w[right[sel]].sum() / w.sum()))
    logw = -beta * E
    logp = logw - (logw.max() + math.log(np.exp(logw - logw.max()).sum()))
    p = np.exp(logp)
    dS_nats = scorer.n * math.log(2) + float((p * logp).sum())
    q_out = float(E.mean()) - float((p * E).sum())
    return {
        "p_correct": float(np.mean(pc)),
        "dS_bits": dS_nats / math.log(2),
        "clausius": q_out / (dS_nats / beta) if dS_nats > 1e-12 else None,
    }
