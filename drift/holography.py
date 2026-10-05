"""
drift.holography — random tensor networks as Ising models (phase 19).

Put a Gaussian random tensor on every bulk vertex of a graph and a maximally entangled pair of bond
dimension D on every edge; the dangling legs are the boundary. The second Rényi entropy of a set A of
boundary legs, averaged as a ratio of averages,

    S₂(A) = −log( E[Tr ρ_A²] / E[(Tr ρ)²] ),

is a free-energy difference of an Ising model on the bulk vertices (Hayden, Nezami, Qi, Thomas,
Walter, Yang, JHEP 2016): spin +1 (identity) or −1 (swap) per vertex, cost log D per bond whose ends
disagree, and each boundary leg pinned to −1 if it is in A and +1 otherwise. As D → ∞ the model
freezes into one domain wall and S₂(A) → |γ_A| log D, |γ_A| the minimal cut separating A from the
rest: the discrete Ryu–Takayanagi formula.

This module builds the graphs, the map onto DRIFT's `IsingModel`, the exact partition functions
(by enumeration, ≤ ~22 bulk spins), the minimal cut by max-flow (independent of the Ising map), and a
direct contraction of actual random tensors -- the only place the *average of the ratio* is measured.
"""

from __future__ import annotations

import string
from dataclasses import dataclass, field

import numpy as np

from drift.ising import IsingModel


# ── graphs with an ordered boundary ──────────────────────────────────────────────────────────────
@dataclass
class BoundaryGraph:
    """Bulk vertices 0..n-1, undirected bonds, and boundary legs in boundary order.

    ``legs[i]`` is the bulk vertex leg i hangs from; a vertex may carry several legs.
    """

    n: int
    bonds: list[tuple[int, int]]
    legs: list[int]
    name: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def n_legs(self) -> int:
        return len(self.legs)


def hyperbolic_rings(n_rings: int, base: int = 3) -> BoundaryGraph:
    """Concentric rings of sizes base·2^r; ring neighbours joined, node j → children 2j, 2j+1.

    Exponential growth with loops: a discrete hyperbolic disk (not a regular {p,q} tiling).
    The boundary legs hang from the outer ring, one per node, in ring order.
    """
    bonds: list[tuple[int, int]] = []
    starts, sizes = [], []
    nxt = 0
    for r in range(n_rings):
        size = base * 2**r
        starts.append(nxt)
        sizes.append(size)
        nxt += size
    for r in range(n_rings):
        s0, m = starts[r], sizes[r]
        for j in range(m):
            bonds.append((s0 + j, s0 + (j + 1) % m))
        if r + 1 < n_rings:
            c0 = starts[r + 1]
            for j in range(m):
                bonds.append((s0 + j, c0 + 2 * j))
                bonds.append((s0 + j, c0 + 2 * j + 1))
    outer = list(range(starts[-1], starts[-1] + sizes[-1]))
    return BoundaryGraph(nxt, bonds, outer, f"hyperbolic-{n_rings}rings", {"ring_sizes": sizes})


def square_grid(rows: int, cols: int) -> BoundaryGraph:
    """A rows × cols grid; one leg per perimeter vertex, clockwise from the top-left corner."""
    vid = lambda i, j: i * cols + j  # noqa: E731
    bonds = [(vid(i, j), vid(i, j + 1)) for i in range(rows) for j in range(cols - 1)]
    bonds += [(vid(i, j), vid(i + 1, j)) for i in range(rows - 1) for j in range(cols)]
    perim = [vid(0, j) for j in range(cols)]
    perim += [vid(i, cols - 1) for i in range(1, rows)]
    perim += [vid(rows - 1, j) for j in range(cols - 2, -1, -1)]
    perim += [vid(i, 0) for i in range(rows - 2, 0, -1)]
    return BoundaryGraph(rows * cols, bonds, perim, f"grid-{rows}x{cols}")


def random_regular(n: int, degree: int, n_legs: int, seed: int) -> BoundaryGraph:
    """A random ``degree``-regular graph with legs on random vertices in a random order (no geometry)."""
    import networkx as nx

    g = nx.random_regular_graph(degree, n, seed=seed)
    rng = np.random.default_rng(seed)
    legs = [int(v) for v in rng.choice(n, size=n_legs, replace=False)]
    return BoundaryGraph(n, [(int(a), int(b)) for a, b in g.edges()], legs, f"random-{degree}reg-{n}")


def intervals(n_legs: int, start: int = 0, max_len: int | None = None) -> list[tuple[int, ...]]:
    """Contiguous boundary intervals starting at ``start`` (cyclic), lengths 1 .. max_len (default n/2)."""
    max_len = n_legs // 2 if max_len is None else max_len
    return [tuple((start + k) % n_legs for k in range(m)) for m in range(1, max_len + 1)]


# ── the map onto DRIFT's Ising model ─────────────────────────────────────────────────────────────
def ising_model(g: BoundaryGraph, region: tuple[int, ...] | set[int], log_d: float) -> IsingModel:
    """The Ising model whose free energy gives Tr ρ_A² (region = A) or (Tr ρ)² (region empty).

    DRIFT's energy is E = −½ sᵀJs − hᵀs. A bond costs log D when its ends disagree:
    log D · [s_a ≠ s_b] = (log D/2)(1 − s_a s_b), so J_ab = log D / 2 (up to a constant). A leg at x
    pinned to b ∈ {±1} costs (log D/2)(1 − b s_x), so it adds (log D/2)·b to h_x. Constants cancel in
    the ratio Z_A / Z_∅.
    """
    region = set(region)
    J = np.zeros((g.n, g.n))
    h = np.zeros(g.n)
    for a, b in g.bonds:
        J[a, b] += log_d / 2.0
        J[b, a] += log_d / 2.0
    for i, x in enumerate(g.legs):
        h[x] += (log_d / 2.0) * (-1.0 if i in region else 1.0)
    return IsingModel(J, h)


def domain_wall_costs(g: BoundaryGraph, region: tuple[int, ...] | set[int]) -> np.ndarray:
    """Integer domain-wall length of every configuration (bit v = 1 ⇔ s_v = −1): cut bonds + bad legs.

    The Ising energy of `ising_model` is log D times this (plus a constant); counting it in integers
    keeps the enumeration exact and lets one table serve every D.
    """
    if g.n > 24:
        raise ValueError(f"enumeration only to 24 bulk spins (got {g.n})")
    region = set(region)
    idx = np.arange(2**g.n, dtype=np.uint32)
    cost = np.zeros(idx.shape, dtype=np.int16)
    for a, b in g.bonds:
        cost += (((idx >> a) ^ (idx >> b)) & 1).astype(np.int16)
    for i, x in enumerate(g.legs):
        bit = ((idx >> x) & 1).astype(np.int16)
        cost += (1 - bit) if i in region else bit  # in A the leg wants s = −1 (bit 1)
    return cost


def _log_z(cost: np.ndarray, log_d: float) -> float:
    e = -log_d * cost.astype(np.float64)
    m = e.max()
    return float(m + np.log(np.exp(e - m).sum()))


def renyi2(g: BoundaryGraph, region, d: float) -> float:
    """S₂(A) = −log(Z_A / Z_∅) at bond dimension d, exact by enumeration (ratio of averages)."""
    log_d = float(np.log(d))
    return _log_z(domain_wall_costs(g, ()), log_d) - _log_z(domain_wall_costs(g, region), log_d)


def ground_state_degeneracy(g: BoundaryGraph, region) -> tuple[int, int]:
    """(minimal domain-wall length, number of configurations that reach it), by enumeration."""
    c = domain_wall_costs(g, region)
    m = int(c.min())
    return m, int(np.count_nonzero(c == m))


# ── the minimal cut, independently of the Ising map ──────────────────────────────────────────────
def min_cut(g: BoundaryGraph, region) -> int:
    """Minimal number of bonds and legs separating A's legs from the others (max-flow, unit capacity)."""
    import networkx as nx

    region = set(region)
    net = nx.Graph()
    for a, b in g.bonds:
        cap = net[a][b]["capacity"] + 1 if net.has_edge(a, b) else 1
        net.add_edge(a, b, capacity=cap)
    for i, x in enumerate(g.legs):
        term = "A" if i in region else "B"
        cap = net[term][x]["capacity"] + 1 if net.has_edge(term, x) else 1
        net.add_edge(term, x, capacity=cap)
    if "A" not in net or "B" not in net:
        return 0
    return int(nx.maximum_flow_value(net, "A", "B"))


# ── a direct contraction of actual random tensors ────────────────────────────────────────────────
def random_boundary_state(g: BoundaryGraph, d: int, rng: np.random.Generator) -> np.ndarray:
    """Contract Gaussian complex tensors over the bonds; return the boundary state, legs in order.

    A maximally entangled pair on a bond contracted with its two vertex tensors is a sum over the
    shared index (its normalisation cancels in Tr ρ_A² / (Tr ρ)²).
    """
    letters = iter(string.ascii_letters)
    bond_idx = {}
    for k, (a, b) in enumerate(g.bonds):
        bond_idx[k] = next(letters)
    leg_idx = [next(letters) for _ in g.legs]
    operands, subs = [], []
    for v in range(g.n):
        inds = [bond_idx[k] for k, (a, b) in enumerate(g.bonds) if v in (a, b)]
        inds += [leg_idx[i] for i, x in enumerate(g.legs) if x == v]
        shape = [d] * len(inds)
        t = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        operands.append(t)
        subs.append("".join(inds))
    expr = ",".join(subs) + "->" + "".join(leg_idx)
    return np.einsum(expr, *operands, optimize="greedy")


def state_renyi2(state: np.ndarray, region: tuple[int, ...]) -> float:
    """−log(Tr ρ_A² / (Tr ρ)²) of one (unnormalised) boundary state; legs = the state's axes."""
    region = list(region)
    rest = [i for i in range(state.ndim) if i not in region]
    da = int(np.prod([state.shape[i] for i in region]))
    m = np.transpose(state, region + rest).reshape(da, -1)
    # ρ on the smaller side (the two reduced states share their spectrum): Tr ρ² = ‖ρ‖²_F.
    rho = m @ m.conj().T if m.shape[0] <= m.shape[1] else m.conj().T @ m
    tr = float(np.trace(rho).real)
    return float(-np.log(float(np.vdot(rho, rho).real) / tr**2))
