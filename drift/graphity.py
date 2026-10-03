"""
drift.graphity — Phase 16: does a geometry emerge from a graph Hamiltonian, or is it put in?

The classical, diagonal part of Quantum Graphity (Konopka, Markopoulou, Smolin). The spins
are **edges**: each of the N(N−1)/2 possible links is a binary variable, the system starts
as the complete graph, and Metropolis edge flips anneal it. The Hamiltonian family is

    H = g_V Σᵢ (vᵢ − v₀)^p  −  g₃·#triangles  −  g₄·#4-cycles     [+ optional connectivity rule]

Graphs are lists of Python-int bitsets (`rows[i]` has bit j iff edge i–j), so the energy
change of a flip is exact and cheap: triangles through (i, j) are one AND + popcount, and
4-cycles through (i, j) are the simple 3-paths i–a–b–j.

What "has a geometry" means here is modest on purpose: a d-dimensional lattice of N nodes
has diameter ~N^(1/d), an expander ~log N. At one N the honest comparison is diameter against
a random-regular null model, not a fitted dimension.

Lesson recorded in rse-hpc-lab ADR-005 Amendment 1: with p = 2 the 4-cycle reward (~d³ per
vertex in a dense graph) beats the valence penalty (~d²), so the *complete graph* is the
minimum — which is why Konopka–Markopoulou–Smolin use an exponential valence penalty.
p = 4 with g₄ = 1/8 restores degree ≈ v₀ as the optimum.
"""

from __future__ import annotations

import math
import random
import statistics

import networkx as nx
import numpy as np


# ── bitset graphs ───────────────────────────────────────────────────────────────
def bits(x: int):
    """Indices of the set bits of x, ascending."""
    while x:
        low = x & -x
        yield low.bit_length() - 1
        x ^= low


def from_networkx(g) -> list[int]:
    rows = [0] * g.number_of_nodes()
    idx = {v: k for k, v in enumerate(g.nodes())}
    for u, v in g.edges():
        rows[idx[u]] |= 1 << idx[v]
        rows[idx[v]] |= 1 << idx[u]
    return rows


def degrees(rows) -> list[int]:
    return [r.bit_count() for r in rows]


def n_edges(rows) -> int:
    return sum(degrees(rows)) // 2


def adjacency(rows) -> np.ndarray:
    n = len(rows)
    A = np.zeros((n, n), dtype=np.int64)
    for i, r in enumerate(rows):
        for j in bits(r):
            A[i, j] = 1
    return A


def triangles(rows) -> int:
    return sum((rows[i] & rows[j]).bit_count()
               for i in range(len(rows)) for j in bits(rows[i]) if j > i) // 3


def four_cycles(rows) -> int:
    """C4 = (tr A⁴ − 2 Σ d² + 2m) / 8."""
    A = adjacency(rows)
    A2 = A @ A
    d = A.sum(1)
    return (int((A2 * A2).sum()) - 2 * int((d * d).sum()) + int(d.sum())) // 8


def bfs(rows, src: int) -> list[int]:
    """Distances from src; −1 where unreachable."""
    dist = [-1] * len(rows)
    dist[src] = 0
    seen = frontier = 1 << src
    d = 0
    while frontier:
        d += 1
        nxt = 0
        for v in bits(frontier):
            nxt |= rows[v]
        nxt &= ~seen
        for v in bits(nxt):
            dist[v] = d
        seen |= nxt
        frontier = nxt
    return dist


def connected(rows, a: int, b: int) -> bool:
    seen = frontier = 1 << a
    target = 1 << b
    while frontier:
        if seen & target:
            return True
        nxt = 0
        for v in bits(frontier):
            nxt |= rows[v]
        frontier = nxt & ~seen
        seen |= frontier
    return bool(seen & target)


def _components(rows) -> list[list[int]]:
    left = (1 << len(rows)) - 1
    comps = []
    while left:
        s = (left & -left).bit_length() - 1
        comp = [v for v, x in enumerate(bfs(rows, s)) if x >= 0]
        comps.append(comp)
        for v in comp:
            left &= ~(1 << v)
    return sorted(comps, key=len, reverse=True)


def components(rows) -> list[int]:
    """Component sizes, largest first."""
    return [len(c) for c in _components(rows)]


def largest_component(rows) -> list[int]:
    best = _components(rows)[0]
    new = {v: k for k, v in enumerate(best)}
    out = [0] * len(best)
    for v in best:
        for u in bits(rows[v]):
            out[new[v]] |= 1 << new[u]
    return out


def diameter(rows) -> int:
    sub = largest_component(rows)
    return max(max(bfs(sub, s)) for s in range(len(sub)))


def ball_growth(rows) -> list[float]:
    sub = largest_component(rows)
    dists = np.array([bfs(sub, s) for s in range(len(sub))])
    return [float((dists <= r).sum(1).mean()) for r in range(int(dists.max()) + 1)]


def random_regular_diameters(n: int, d: int, seeds) -> list[int]:
    """The expander null model: diameters of uniformly random d-regular graphs."""
    return [diameter(from_networkx(nx.random_regular_graph(d, n, seed=int(s)))) for s in seeds]


def disjoint_blocks(n: int, block: str) -> list[int]:
    """Disjoint K5 or K4,4 blocks over n nodes, leftovers as a path."""
    size = {"K5": 5, "K44": 8}[block]
    rows = [0] * n
    k = 0
    while k + size <= n:
        for a in range(k, k + size):
            for b in range(k, k + size):
                if a != b and (block == "K5" or (a - k < 4) != (b - k < 4)):
                    rows[a] |= 1 << b
        k += size
    for a in range(k, n - 1):
        rows[a] |= 1 << (a + 1)
        rows[a + 1] |= 1 << a
    return rows


# ── the Hamiltonian and its Metropolis anneal ───────────────────────────────────
class GraphHamiltonian:
    def __init__(self, v0: int = 4, g_v: float = 1.0, g3: float = 0.0, g4: float = 0.0,
                 power: int = 2, keep_connected: bool = False):
        self.v0, self.g_v, self.g3, self.g4 = v0, g_v, g3, g4
        self.p, self.keep_connected = power, keep_connected

    def energy(self, rows) -> float:
        e = self.g_v * sum((d - self.v0) ** self.p for d in degrees(rows))
        if self.g3:
            e -= self.g3 * triangles(rows)
        if self.g4:
            e -= self.g4 * four_cycles(rows)
        return e

    def delta(self, rows, deg, i: int, j: int) -> float:
        """Exact energy change of flipping edge (i, j)."""
        present = (rows[i] >> j) & 1
        s = -1 if present else 1
        di, dj, v0, p = deg[i], deg[j], self.v0, self.p
        dE = self.g_v * ((di + s - v0) ** p - (di - v0) ** p + (dj + s - v0) ** p - (dj - v0) ** p)
        if self.g3:
            dE -= self.g3 * s * (rows[i] & rows[j]).bit_count()
        if self.g4:
            ri = rows[i] & ~(1 << j)
            rj = rows[j] & ~(1 << i)
            dE -= self.g4 * s * sum((rows[a] & rj).bit_count() for a in bits(ri))
        return dE


def anneal(H: GraphHamiltonian, n: int, sweeps: int, t_hi: float = 5.0, t_lo: float = 0.02,
           seed: int = 0, start: list[int] | None = None) -> dict:
    """Metropolis edge flips, geometric cooling, from the complete graph unless `start` is
    given. One sweep = n(n−1)/2 proposals. Returns the graph and a tracked-energy check."""
    full = (1 << n) - 1
    rows = list(start) if start is not None else [full & ~(1 << i) for i in range(n)]
    deg = degrees(rows)
    rng = random.Random(seed)
    E = H.energy(rows)
    rejected = 0
    per_sweep = n * (n - 1) // 2
    for k in range(sweeps):
        T = t_hi * (t_lo / t_hi) ** (k / max(sweeps - 1, 1))
        for _ in range(per_sweep):
            i = rng.randrange(n)
            j = rng.randrange(n - 1)
            j += j >= i
            present = (rows[i] >> j) & 1
            dE = H.delta(rows, deg, i, j)
            if dE > 0 and rng.random() >= math.exp(-dE / T):
                continue
            rows[i] ^= 1 << j
            rows[j] ^= 1 << i
            step = -1 if present else 1
            deg[i] += step
            deg[j] += step
            if present and H.keep_connected and not connected(rows, i, j):
                rows[i] ^= 1 << j
                rows[j] ^= 1 << i
                deg[i] -= step
                deg[j] -= step
                rejected += 1
                continue
            E += dE
    return {"rows": rows, "energy": H.energy(rows), "tracked_err": abs(E - H.energy(rows)),
            "rejected_disconnect": rejected}


def observe(rows) -> dict:
    comps = components(rows)
    deg = degrees(rows)
    return {
        "largest_fraction": comps[0] / len(rows),
        "n_components": len(comps),
        "mean_degree": statistics.fmean(deg),
        "triangles": triangles(rows),
        "four_cycles": four_cycles(rows),
        "diameter": diameter(rows),
    }
