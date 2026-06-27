"""
drift.tunneling — Phase 10: quantum annealing vs simulated annealing, honestly.
===============================================================================
Phase 9 reached the Ising ground state by quantum adiabatic evolution and was careful *not*
to claim it beats thermal annealing. This phase asks the deferred question directly — and
answers it the honest way: **quantum annealing's edge is specific, not general.**

Both annealers are run on the *same* energy landscape ``energies[b]`` (the diagonal of the
problem Hamiltonian), so the comparison is exact:

- **Quantum annealing** (`drift.anneal.quantum_anneal`): tunnels under a transverse field.
- **Simulated annealing** (`metropolis_sa` below): single-spin-flip Metropolis — local moves
  that change the Hamming weight by ±1, so it must *climb* any barrier in its path.

The landscape we use is the textbook **spike** (Farhi et al.): the cost is the Hamming weight
``w`` (a smooth funnel to the all-`+1` ground state at ``w = 0``), with a tall thin barrier
added at one intermediate weight. A walker that starts in the large-``w`` region must cross
that one weight to descend. Single-spin-flip SA has to climb the spike (exponentially slow in
its height); quantum annealing tunnels through it (the spike is *thin* in Hamming space). On a
plain funnel (no spike), the control, SA descends easily and there is **no** quantum edge.

So the honest punchline: QA wins where the bottleneck is a barrier thin enough to tunnel, ties
where the landscape is easy, and (Phase 9) loses where the spectral gap closes. We measure all
of it.
"""

from __future__ import annotations

import numpy as np


def hamming_cost_energies(
    n: int, *, spike_at: int | None = None, spike_height: float = 0.0
) -> np.ndarray:
    """Energy of every basis state ``b`` = its Hamming weight (number of `-1` spins, i.e.
    `popcount(b)` in the qubit-0-MSB / |0⟩↦+1 convention shared with `drift.anneal`), with an
    optional tall thin spike added at one weight. Global minimum: ``b = 0`` (all +1, w = 0).

    ``spike_at=None`` → a plain funnel (the control). ``spike_at=k, spike_height=B`` → a
    barrier of height ``B`` at weight ``k`` that local single-flip moves must climb."""
    idx = np.arange(1 << n)
    weight = np.array([int(x).bit_count() for x in idx], dtype=float)
    energies = weight.copy()
    if spike_at is not None:
        energies[weight == spike_at] += spike_height
    return energies


def metropolis_sa(
    energies: np.ndarray,
    n: int,
    *,
    num_reads: int = 50,
    sweeps: int = 200,
    t0: float = 3.0,
    t1: float = 0.05,
    seed: int = 0,
) -> dict:
    """Single-spin-flip simulated annealing on the diagonal landscape ``energies`` (indexed by
    basis integer, qubit 0 = MSB to match `drift.anneal`). Each sweep proposes ``n`` random
    bit flips under a geometric temperature schedule. Returns the fraction of independent
    reads that reach the global minimum, plus the best energy found.

    This is the *classical* competitor for Phase 10: it can only move between configurations
    that differ by one spin, so a barrier in Hamming weight is a wall it must climb."""
    rng = np.random.default_rng(seed)
    e_min = float(energies.min())
    cooling = (t1 / t0) ** (1.0 / max(sweeps, 1))
    successes = 0
    best_overall = np.inf

    for _ in range(num_reads):
        b = int(rng.integers(0, 1 << n))
        e = float(energies[b])
        best = e
        temp = t0
        for _ in range(sweeps):
            for _ in range(n):
                k = int(rng.integers(0, n))
                b2 = b ^ (1 << (n - 1 - k))  # flip spin k (MSB convention)
                de = float(energies[b2]) - e
                if de <= 0.0 or rng.random() < np.exp(-de / temp):
                    b, e = b2, float(energies[b2])
                    if e < best:
                        best = e
            temp *= cooling
        if np.isclose(best, e_min):
            successes += 1
        best_overall = min(best_overall, best)

    return {
        "success": successes / num_reads,
        "best_energy": float(best_overall),
        "num_reads": num_reads,
    }
