"""
drift.solve — one entry point that solves any Ising/QUBO with the best available method.

DRIFT grew several solvers, each with a different reach: `exact_ground_state` is *certified* but
dies at ~20 spins; the GPU parallel-tempering engine (Phase 14) reaches thousands of spins fast;
the CPU reference is the portable fallback. `solve` picks among them by size and hardware, and —
crucially — tells you whether the answer is **certified** (provably the ground state) or a **strong
heuristic** minimum. So the faces (MaxCut, factoring, circuits, …) get a single call that is exact
when it can be and scales when it must, without ever hiding which one you got.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import gpu
from .ising import IsingModel
from .solvers.exact import exact_ground_state
from .solvers.parallel_tempering import parallel_tempering


@dataclass
class Solution:
    """A ground-state search result, honest about its provenance."""

    s: np.ndarray       # the configuration found
    energy: float       # its energy
    method: str         # "exact" | "gpu-pt" | "cpu-pt"
    certified: bool     # True only when provably the ground state (the exact engine)

    @property
    def n(self) -> int:
        return int(self.s.shape[0])


def solve(
    model: IsingModel,
    *,
    exact_max: int = 18,
    use_gpu: bool = True,
    n_replicas: int | None = None,
    n_rounds: int = 400,
    sweeps_per_round: int = 4,
    seed: int = 0,
) -> Solution:
    """Ground state of `model`, by the best method available for its size.

    - **n ≤ exact_max** → `exact_ground_state`: brute force, **certified** the true ground state.
    - **larger, GPU built** → the Phase-14 GPU parallel-tempering engine (strong minimum).
    - **larger, no GPU** → the CPU reference parallel tempering (strong minimum).

    The result says which method ran and whether it is certified — a fast heuristic minimum is never
    passed off as the proven optimum.
    """
    n = model.n
    if n <= exact_max:
        s, e, _ = exact_ground_state(model, max_n=max(exact_max, n))
        return Solution(s=s, energy=e, method="exact", certified=True)

    if use_gpu and gpu.gpu_available():
        r = gpu.parallel_tempering_gpu(
            model, n_replicas=n_replicas or 256, n_rounds=n_rounds,
            sweeps_per_round=sweeps_per_round, seed=seed,
        )
        return Solution(s=r.best_s, energy=r.best_E, method="gpu-pt", certified=False)

    r = parallel_tempering(
        model, n_replicas=n_replicas or 32, n_rounds=n_rounds,
        sweeps_per_round=sweeps_per_round, seed=seed,
    )
    return Solution(s=r.best_s, energy=r.best_E, method="cpu-pt", certified=False)
