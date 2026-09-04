"""Phase 8 experiment — the reservoir face: memory capacity & the edge of chaos.

Produces figures/phase8_reservoir.png:
  (a) the forgetting curve MC_k vs delay k (memory fades with delay),
  (b) total memory capacity vs spectral radius (it peaks near the edge of chaos).

Run:  python experiments/phase8_reservoir.py
"""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from drift.reservoir import IsingReservoir, memory_capacity  # noqa: E402

# Light theme: published figures are read on white. Accents are dark
# enough to stay legible there and in greyscale print.
BG, FG, A1, A2 = "#ffffff", "#1a1d21", "#0b6ea8", "#6a3d9a"


def main() -> None:
    n = 200
    max_delay = int(1.5 * n)

    # (a) forgetting curve at the edge of chaos
    res = IsingReservoir(n=n, spectral_radius=0.95, leak=0.3, seed=0)
    mc, curve = memory_capacity(res, max_delay=max_delay, n_train=1800)

    # (b) memory capacity vs spectral radius
    radii = np.linspace(0.3, 1.4, 12)
    mcs = []
    for r in radii:
        rr = IsingReservoir(n=n, spectral_radius=float(r), leak=0.3, seed=0)
        m, _ = memory_capacity(rr, max_delay=max_delay, n_train=1800)
        mcs.append(m)
    mcs = np.asarray(mcs)
    peak_r = float(radii[int(np.argmax(mcs))])

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=BG)
    for a in ax:
        a.set_facecolor(BG)
        a.tick_params(colors=FG)
        for sp in a.spines.values():
            sp.set_color("#d8dce0")
        a.title.set_color(FG)
        a.xaxis.label.set_color(FG)
        a.yaxis.label.set_color(FG)

    ax[0].plot(range(1, len(curve) + 1), curve, color=A1, lw=1.6)
    ax[0].set_xlabel("delay k")
    ax[0].set_ylabel("MC_k")
    ax[0].set_title(f"Forgetting curve  (total MC = {mc:.1f} / N={n})")

    ax[1].plot(radii, mcs, "o-", color=A2, lw=1.6)
    ax[1].axvline(1.0, ls="--", color="#7a8290", alpha=0.7, label="ρ = 1 (edge of chaos)")
    ax[1].set_xlabel("spectral radius ρ")
    ax[1].set_ylabel("total memory capacity")
    ax[1].set_title("Memory capacity vs ρ")
    ax[1].legend(facecolor=BG, edgecolor="#d8dce0", labelcolor=FG)

    fig.tight_layout()
    fig.savefig("figures/phase8_reservoir.png", dpi=140, facecolor=BG)
    print(f"MC = {mc:.1f} / N={n}; peak MC vs spectral radius at rho = {peak_r:.2f}")


if __name__ == "__main__":
    main()
