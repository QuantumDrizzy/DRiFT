"""Experiment: the self-replication face (Phase 6), grown at scale on the GPU engine (Phase 14).

The point of the GPU engine was never speed for its own sake — it is the microscope's new lens for
the *four faces*. Here the **crystallisation / self-replication** face is grown far past the exact
wall: `crystal_2d` gives translation-invariant frustrated couplings whose ground state is the
period-4 ↑↑↓↓ stripe crystal (energy density exactly −2), and `drift.solve` finds it — certified-exact
at 4×4, then on the GPU up to 64×64 = 4096 spins (Phase 6 reached only a 12×12 anneal).

The result: a **defect-free** crystal at every scale (period 4, E/n = −2), i.e. parallel tempering
finds the true global crystal, not a domain-riddled local minimum. Matter replicating a motif from
purely local rules, at a scale no exact method can touch.

    python -m experiments.crystal_gpu
"""

from __future__ import annotations

import numpy as np

from drift import solve
from drift.builders.crystal import column_period, crystal_2d, is_striped


def main(outdir: str = "figures") -> None:
    print("Self-replication face grown at scale — known ground energy E/n = −2, period-4 stripes:")
    print(f"    {'L×L':>7} {'n':>5} {'method':>7} {'E/n':>8} {'period':>7} {'defect-free':>12}")
    for L in (4, 8, 16, 32, 48, 64):
        s = solve(crystal_2d(L, L), n_replicas=256, n_rounds=600, sweeps_per_round=5, seed=0)
        grid = s.s.reshape(L, L)
        print(f"    {L:>3}×{L:<3} {L * L:>5} {s.method:>7} {s.energy / (L * L):>8.3f} "
              f"{column_period(grid):>7} {str(is_striped(grid)):>12}")

    # figure: a hot disordered melt vs the grown crystal (largest lattice)
    L = 64
    sol = solve(crystal_2d(L, L), n_replicas=256, n_rounds=600, sweeps_per_round=5, seed=0)
    rng = np.random.default_rng(0)
    melt = rng.integers(0, 2, L * L) * 2 - 1
    try:
        from drift.viz import plot_crystal
    except Exception as exc:
        print(f"  (skipping figure: {exc})")
        return
    path = plot_crystal(melt, sol.s, L, L, period=column_period(sol.s.reshape(L, L)),
                        title=f"Self-replication grown on the GPU — {L}×{L} = {L * L} spins",
                        out=f"{outdir}/crystal_gpu.png")
    print(f"  saved {path}")


if __name__ == "__main__":
    main()
