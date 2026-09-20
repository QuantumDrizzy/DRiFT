"""Seeded instance generators for the versioned DRiFT scale-path bank.

Each family is a *fixed* ensemble: the same (n, seed, params) always yields the same
(J, h). Stable IDs in the manifest point at these generators, so a sweep is reproducible
without checking in dense matrices for every n.

Families
--------
maxcut-er          Erdős–Rényi MaxCut (Phase 2), G(n, p), antiferromagnetic Ising.
pmj-glass          Complete ±J spin glass (the Phase 14 hard case), h = 0.
bipartite-maxcut   Random bipartite MaxCut; the planted 2-colouring cuts every edge,
                   so the optimum is known at any n (Phase 14 quality check).
ferro-chain        Open 1-D ferromagnet; analytic ground energy E = −(n−1).
crystal            Period-4 stripe crystal (Phase 6); analytic E = −2n when the
                   lattice fits the unit cell (cols multiple of 4).
tfim-chain         Open 1-D transverse-field Ising chain — the existing MPS/TEBD
                   path (Phase 13). Not a ``drift.solve`` load; χ is only defined here.
"""

from __future__ import annotations

from typing import Literal, NoReturn

import numpy as np

from ..builders.crystal import crystal_2d
from ..builders.qubo import maxcut_ising, random_graph
from ..ising import IsingModel

FamilyName = Literal[
    "maxcut-er",
    "pmj-glass",
    "bipartite-maxcut",
    "ferro-chain",
    "crystal",
    "tfim-chain",
]

FAMILIES: tuple[FamilyName, ...] = (
    "maxcut-er",
    "pmj-glass",
    "bipartite-maxcut",
    "ferro-chain",
    "crystal",
    "tfim-chain",
)

# χ is a tensor-network observable. The existing MPS solver is 1-D TFIM TEBD;
# it does not apply to dense MaxCut / spin-glass graphs or 2-D crystals.
MPS_FAMILIES: frozenset[str] = frozenset({"tfim-chain"})

# Documented MPS cutoff for the scale sweep (CPU). Phase 13 has shown n=48 with a
# larger budget; the sweep stays at this cap so CI/CPU results stay cheap and honest.
MPS_N_MAX_SWEEP = 16
MPS_N_MAX_LOCAL = 24


def unknown_family(family: str) -> NoReturn:
    """Exhaustive-switch sink: a new family must be handled explicitly."""
    raise ValueError(f"unknown instance family {family!r}; expected one of {FAMILIES}")


def ferro_chain(n: int, j: float = 1.0) -> IsingModel:
    """Open 1-D ferromagnet. Ground energy is −j(n−1) (all spins aligned)."""
    J = np.zeros((n, n))
    for i in range(n - 1):
        J[i, i + 1] = J[i + 1, i] = j
    return IsingModel(J=J, h=np.zeros(n))


def pmj_spin_glass(n: int, seed: int, *, p: float = 1.0, field: float = 0.0) -> IsingModel:
    """±J couplings (optionally Erdős–Rényi-thinned) with an optional Gaussian field."""
    rng = np.random.default_rng(seed)
    J = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                J[i, j] = J[j, i] = float(rng.choice([-1.0, 1.0]))
    h = rng.normal(scale=field, size=n) if field else np.zeros(n)
    return IsingModel(J=J, h=h)


def bipartite_graph(n: int, p: float, seed: int) -> np.ndarray:
    """Random bipartite graph on two equal halves; edges only cross the split.

    The maximum cut is every edge (the planted 2-colouring), so the Ising ground
    energy is −n_edges. ``n`` must be even.
    """
    if n % 2:
        raise ValueError(f"bipartite MaxCut family requires even n, got {n}")
    rng = np.random.default_rng(seed)
    w = np.zeros((n, n))
    half = n // 2
    for i in range(half):
        for j in range(half, n):
            if rng.random() < p:
                w[i, j] = w[j, i] = 1.0
    return w


def n_edges(W: np.ndarray) -> int:
    """Undirected edge count of a symmetric zero-diagonal weight matrix."""
    return int(np.count_nonzero(np.triu(np.asarray(W))))


def analytic_energy(family: str, n: int, extra: dict) -> tuple[float | None, str | None]:
    """Closed-form ground energy where the family has one; else (None, None)."""
    if family == "ferro-chain":
        return -float(n - 1), "analytic-open-ferro-chain"
    if family == "bipartite-maxcut":
        return -float(n_edges(extra["W"])), "analytic-bipartite-all-edges-cut"
    if family == "crystal":
        return -2.0 * n, "analytic-period4-crystal-E/n=-2"
    if family in FAMILIES:
        return None, None
    unknown_family(family)


def build_model(family: str, n: int, seed: int, params: dict) -> tuple[IsingModel, dict]:
    """Materialise (IsingModel, extra) for one catalog row. Deterministic in (family, n, seed, params)."""
    extra: dict = {}
    if family == "maxcut-er":
        p = float(params.get("p", 0.5))
        W = random_graph(n, p=p, seed=seed)
        extra = {"W": W, "p": p}
        return maxcut_ising(W), extra
    if family == "pmj-glass":
        p = float(params.get("p", 1.0))
        field = float(params.get("field", 0.0))
        model = pmj_spin_glass(n, seed, p=p, field=field)
        extra = {"p": p, "field": field}
        return model, extra
    if family == "bipartite-maxcut":
        p = float(params.get("p", 0.5))
        W = bipartite_graph(n, p=p, seed=seed)
        extra = {"W": W, "p": p, "n_edges": n_edges(W)}
        return maxcut_ising(W), extra
    if family == "ferro-chain":
        j = float(params.get("j", 1.0))
        extra = {"j": j}
        return ferro_chain(n, j=j), extra
    if family == "crystal":
        rows = int(params.get("rows", 4))
        cols = int(params.get("cols", 4))
        if rows * cols != n:
            raise ValueError(f"crystal n={n} != rows*cols={rows * cols}")
        extra = {"rows": rows, "cols": cols}
        return crystal_2d(rows, cols), extra
    if family == "tfim-chain":
        j = float(params.get("j", 1.0))
        gamma = float(params.get("gamma", 1.0))
        extra = {"j": j, "gamma": gamma}
        return ferro_chain(n, j=j), extra
    unknown_family(family)
