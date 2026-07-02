"""
drift.solvers.parallel_tempering — replica-exchange Metropolis (the CPU reference).

Simulated annealing runs one walker down one cooling schedule; on a rugged landscape it
freezes into whatever basin it fell into. Parallel tempering runs **R walkers at once**, held
at a ladder of fixed temperatures T_min … T_max, and periodically proposes to *swap* the whole
configuration between adjacent-temperature replicas. Hot replicas roam freely and tunnel over
barriers; cold replicas refine; the swaps let a good configuration discovered while hot flow
down to the cold replica that reports the answer. It is the standard, honest way to find deep
Ising minima that single-temperature annealing misses.

This is the **reference implementation**: correct, readable, and validated against the exact
ground state on small n. It is also the exact algorithm the CUDA engine mirrors — one GPU block
per replica, single-spin-flip sweeps with local-field maintenance, a swap kernel between rounds
— so the GPU port has a CPU oracle it must agree with (see `cuda/` and Phase 14).

Not built for raw speed (that is the GPU's job); built to define what "correct" means.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..ising import IsingModel


def geometric_ladder(n_replicas: int, T_min: float, T_max: float) -> np.ndarray:
    """A geometric temperature ladder T_min … T_max (n_replicas rungs, ascending).

    Geometric spacing keeps the swap-acceptance roughly uniform across the ladder, which is what
    makes replica exchange mix well."""
    if n_replicas == 1:
        return np.array([T_min], dtype=np.float64)
    return T_min * (T_max / T_min) ** (np.arange(n_replicas) / (n_replicas - 1))


@dataclass
class PtResult:
    """The outcome of a parallel-tempering run."""

    best_s: np.ndarray          # lowest-energy configuration found, shape (n,)
    best_E: float               # its energy
    temperatures: np.ndarray    # the temperature ladder, shape (R,)
    swap_rate: float            # fraction of accepted adjacent swaps (health of the ladder)
    best_history: np.ndarray = field(default=None)  # best-so-far energy per round
    throughput: float = None    # spin-flips/sec (filled by the GPU engine; None on CPU)
    seconds: float = None       # wall time of the solve (filled by the GPU engine)


def parallel_tempering(
    model: IsingModel,
    *,
    n_replicas: int = 16,
    T_min: float = 0.1,
    T_max: float = 5.0,
    n_rounds: int = 300,
    sweeps_per_round: int = 5,
    seed: int = 0,
) -> PtResult:
    """Find a low-energy configuration of `model` by replica exchange.

    Each *round* runs `sweeps_per_round` Metropolis sweeps on every replica (a sweep = n
    single-spin-flip attempts), then proposes swaps between every adjacent pair on the ladder.
    Returns the coldest, deepest configuration seen across all replicas and rounds.
    """
    rng = np.random.default_rng(seed)
    n = model.n
    R = n_replicas

    temps = geometric_ladder(R, T_min, T_max)
    beta = 1.0 / np.maximum(temps, 1e-12)

    # one spin configuration per replica, and its running energy
    s = (rng.integers(0, 2, size=(R, n)) * 2 - 1).astype(np.float64)
    E = np.array([model.energy(s[r]) for r in range(R)])

    best_idx = int(np.argmin(E))
    best_s = s[best_idx].copy()
    best_E = float(E[best_idx])
    best_history = np.empty(n_rounds, dtype=np.float64)

    swaps_tried = 0
    swaps_done = 0

    for rnd in range(n_rounds):
        # ── Metropolis sweeps on every replica at its own temperature ──────────
        for r in range(R):
            inv_T = beta[r]
            sr = s[r]
            Er = E[r]
            for _ in range(sweeps_per_round * n):
                i = int(rng.integers(n))
                dE = model.delta_energy_flip(sr, i)   # 2 s_i (J_i·s + h_i)
                if dE <= 0.0 or rng.random() < np.exp(-dE * inv_T):
                    sr[i] = -sr[i]
                    Er += dE
            E[r] = Er
            if Er < best_E:
                best_E = float(Er)
                best_s = sr.copy()

        # ── replica exchange: swap adjacent configs with the PT criterion ──────
        # accept swapping configs of replicas r, r+1 with prob min(1, exp((β_r − β_{r+1})(E_r − E_{r+1})))
        start = rnd % 2  # alternate even/odd pairings so every bond gets tried
        for r in range(start, R - 1, 2):
            swaps_tried += 1
            delta = (beta[r] - beta[r + 1]) * (E[r] - E[r + 1])
            if delta >= 0.0 or rng.random() < np.exp(delta):
                s[[r, r + 1]] = s[[r + 1, r]]
                E[r], E[r + 1] = E[r + 1], E[r]
                swaps_done += 1

        best_history[rnd] = best_E

    swap_rate = swaps_done / swaps_tried if swaps_tried else 0.0
    return PtResult(
        best_s=best_s,
        best_E=best_E,
        temperatures=temps,
        swap_rate=swap_rate,
        best_history=best_history,
    )
