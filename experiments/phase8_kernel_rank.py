"""Phase 8 experiment — Legenstein–Maass kernel & generalization rank vs spectral radius.

Complements `phase8_reservoir.py` (memory capacity) with the other axis of reservoir
quality: linear-separation power. The figure shows

  * kernel rank — independent inputs the reservoir separates (rises with ρ, saturates),
  * generalization rank — sensitivity to input noise (stays low, then climbs in chaos),
  * computational capability = kernel − generalization (low in the ordered regime,
    largest around the edge of chaos).

    python -m experiments.phase8_kernel_rank
"""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from drift.reservoir import IsingReservoir, generalization_rank, kernel_rank  # noqa: E402

# Light theme: published figures are read on white. Accents are dark
# enough to stay legible there and in greyscale print.
BG, FG, A1, A2, A3 = "#ffffff", "#1a1d21", "#0b6ea8", "#6a3d9a", "#d95f02"


def main() -> None:
    n, m = 200, 100
    radii = np.linspace(0.2, 2.0, 16)
    kw = dict(n_streams=m, length=120, washout=40)

    kr, gr = [], []
    for r in radii:
        res = IsingReservoir(n=n, spectral_radius=float(r), leak=0.3, seed=0)
        kr.append(kernel_rank(res, **kw))
        gr.append(generalization_rank(res, noise=0.01, **kw))
    kr, gr = np.asarray(kr), np.asarray(gr)
    power = kr - gr
    peak_r = float(radii[int(np.argmax(power))])

    fig, ax = plt.subplots(figsize=(7.5, 4.8), facecolor=BG)
    ax.set_facecolor(BG)
    ax.tick_params(colors=FG)
    for sp in ax.spines.values():
        sp.set_color("#d8dce0")
    ax.title.set_color(FG)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)

    ax.plot(radii, kr, "o-", color=A1, lw=1.6, label="kernel rank (separation power)")
    ax.plot(radii, gr, "s-", color=A3, lw=1.6, label="generalization rank (noise sensitivity)")
    ax.plot(radii, power, "^-", color=A2, lw=1.8, label="computational capability (kernel − gen)")
    ax.axvline(1.0, ls="--", color="#7a8290", alpha=0.7, label="ρ = 1 (edge of chaos)")
    ax.set_xlabel("spectral radius ρ")
    ax.set_ylabel(f"effective rank (M = {m} streams)")
    ax.set_title("Reservoir linear-separation power vs ρ")
    ax.legend(facecolor=BG, edgecolor="#d8dce0", labelcolor=FG, fontsize=8)

    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    fig.savefig("figures/reservoir_kernel_rank.png", dpi=140, facecolor=BG)
    print(f"kernel rank rises {kr[0]}→{kr[-1]} (cap M={m}); "
          f"computational capability peaks near ρ = {peak_r:.2f}")


if __name__ == "__main__":
    main()
