"""Phase 10 experiment — quantum vs simulated annealing on the same landscape.

Produces figures/phase10_tunneling.png:
  (a) quantum annealing: success vs anneal time T, on a plain funnel and on a spike. QA
      solves the funnel quickly and *tunnels* its way up the spike as the anneal slows.
  (b) simulated annealing: success vs computational effort (sweeps), same two landscapes.
      Single-spin-flip SA solves the funnel, but the spike is a wall it cannot climb — more
      effort barely helps.

The honest reading: quantum annealing has a real, measured edge exactly where the bottleneck
is a barrier thin enough to tunnel (the spike) — and none at all on the easy funnel. It is a
specific advantage, not a general one.

Run:  python experiments/phase10_tunneling.py
"""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from drift.anneal import quantum_anneal, success_probability, transverse_driver  # noqa: E402
from drift.tunneling import hamming_cost_energies, metropolis_sa  # noqa: E402

BG, FG, A1, A2 = "#0f1117", "#e2e8f0", "#5ac4b8", "#aa96f0"


def main() -> None:
    n = 10
    driver = transverse_driver(n)
    funnel = hamming_cost_energies(n)
    spike = hamming_cost_energies(n, spike_at=2, spike_height=10.0)

    def qa_success(energies, T):
        psi = quantum_anneal(driver, sp.diags(energies, format="csr"), T, steps=200)
        return success_probability(psi, energies)

    times = np.array([2.0, 5.0, 10.0, 20.0, 40.0, 70.0, 110.0])
    sweeps_grid = np.array([20, 50, 100, 200, 400, 700])

    qa_funnel = [qa_success(funnel, T) for T in times]
    qa_spike = [qa_success(spike, T) for T in times]
    sa_funnel = [metropolis_sa(funnel, n, num_reads=30, sweeps=int(s), seed=1)["success"]
                 for s in sweeps_grid]
    sa_spike = [metropolis_sa(spike, n, num_reads=30, sweeps=int(s), seed=1)["success"]
                for s in sweeps_grid]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=BG)
    ax1.plot(times, qa_funnel, "o-", color=A1, label="funnel (no barrier)")
    ax1.plot(times, qa_spike, "o-", color=A2, label="spike (thin barrier)")
    ax2.plot(sweeps_grid, sa_funnel, "s-", color=A1, label="funnel (no barrier)")
    ax2.plot(sweeps_grid, sa_spike, "s-", color=A2, label="spike (thin barrier)")

    ax1.set_xscale("log")
    ax1.set_xlabel("quantum anneal time T", color=FG)
    ax1.set_title("(a) quantum annealing — tunnels the spike", color=FG)
    ax2.set_xscale("log")
    ax2.set_xlabel("simulated-annealing effort (sweeps)", color=FG)
    ax2.set_title("(b) simulated annealing — walled out by the spike", color=FG)

    for ax in (ax1, ax2):
        ax.set_facecolor(BG)
        ax.set_ylabel("success probability", color=FG)
        ax.set_ylim(-0.03, 1.05)
        ax.axhline(1.0, color=FG, lw=0.6, ls=":")
        ax.tick_params(colors=FG)
        for sp_ in ax.spines.values():
            sp_.set_color(FG)
        leg = ax.legend(facecolor=BG, edgecolor=FG, labelcolor=FG, fontsize=9)
        leg.get_frame().set_alpha(0.6)

    fig.suptitle("DRIFT P10 — quantum vs simulated annealing: the edge is specific, not general",
                 color=FG, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(os.path.dirname(__file__), "..", "figures", "phase10_tunneling.png")
    fig.savefig(out, dpi=130, facecolor=BG, bbox_inches="tight")
    print(f"wrote {os.path.normpath(out)}")


if __name__ == "__main__":
    main()
