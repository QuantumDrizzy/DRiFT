"""Phase 9 experiment — the optimization face, run quantum (adiabatic annealing).

Produces figures/phase9_quantum_anneal.png:
  (a) success probability vs anneal time T, for an open-gap and a closing-gap problem —
      success → 1 as the anneal slows (the adiabatic theorem), but the small-gap problem
      lags at every T;
  (b) the spectral gap Δ(s) along the schedule for both — the smaller Δ_min is exactly why
      the second problem is harder.

The honest punchline (DRIFT's microscope): quantum annealing reaches the *same* Ising ground
state as Phase-2 thermal annealing, but its cost is set by the gap (≈ 1/Δ_min²) — and when
the gap closes, it fails too. No magic.

Run:  python experiments/phase9_quantum_anneal.py
"""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from drift.anneal import (  # noqa: E402
    problem_hamiltonian,
    quantum_anneal,
    spectral_gap_path,
    success_probability,
    transverse_driver,
)
from drift.ising import IsingModel  # noqa: E402

# Light theme: published figures are read on white. Accents are dark
# enough to stay legible there and in greyscale print.
BG, FG, A1, A2 = "#ffffff", "#1a1d21", "#0b6ea8", "#6a3d9a"


def ferro_chain(n: int, field: float) -> IsingModel:
    """Ferromagnetic chain (J=1) in a uniform longitudinal field. A strong field gives a
    unique ground state and an open gap; a tiny field leaves two nearly-degenerate minima
    (all-up / all-down) separated by a wide tunnelling barrier — a small gap."""
    J = np.zeros((n, n))
    for i in range(n - 1):
        J[i, i + 1] = J[i + 1, i] = 1.0
    return IsingModel(J=J, h=np.full(n, field))


def main() -> None:
    n = 8
    times = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 14.0, 20.0, 32.0, 50.0])
    driver = transverse_driver(n)

    problems = {
        "open gap (h=0.40)": (ferro_chain(n, 0.40), A1),
        "closing gap (h=0.04)": (ferro_chain(n, 0.04), A2),
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=BG)
    for label, (model, color) in problems.items():
        H_problem, energies = problem_hamiltonian(model)
        gp = spectral_gap_path(driver, H_problem, npts=41)
        succ = [success_probability(quantum_anneal(driver, H_problem, T, steps=200), energies)
                for T in times]
        ax1.plot(times, succ, "o-", color=color, label=f"{label}  (Δ_min={gp['gap_min']:.2f})")
        ax2.plot(gp["s"], gp["gap"], "-", color=color, label=label)
        ax2.scatter([gp["s_at_min"]], [gp["gap_min"]], color=color, zorder=5)

    for ax in (ax1, ax2):
        ax.set_facecolor(BG)
        ax.tick_params(colors=FG)
        for sp in ax.spines.values():
            sp.set_color(FG)
        leg = ax.legend(facecolor=BG, edgecolor=FG, labelcolor=FG, fontsize=9)
        leg.get_frame().set_alpha(0.6)

    ax1.set_xscale("log")
    ax1.set_xlabel("anneal time T", color=FG)
    ax1.set_ylabel("success probability", color=FG)
    ax1.set_title("(a) slower anneal → ground state — but the gap sets the price", color=FG)
    ax1.axhline(1.0, color=FG, lw=0.6, ls=":")

    ax2.set_xlabel("schedule s", color=FG)
    ax2.set_ylabel("spectral gap  Δ(s) = E₁ − E₀", color=FG)
    ax2.set_title("(b) the smaller Δ_min is why it is harder", color=FG)

    fig.suptitle("DRIFT P9 — the optimization face, run quantum (adiabatic annealing)",
                 color=FG, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(os.path.dirname(__file__), "..", "figures", "phase9_quantum_anneal.png")
    fig.savefig(out, dpi=130, facecolor=BG, bbox_inches="tight")
    print(f"wrote {os.path.normpath(out)}")


if __name__ == "__main__":
    main()
