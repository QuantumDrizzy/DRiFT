"""Phase 12 experiment — universal computation as an Ising ground state.

Composes logic gates into a 1-bit full adder and lets DRIFT's exact Ising engine compute its
whole truth table (matter doing arithmetic by relaxing to a ground state), then shows the same
honest scaling wall as factoring.

Produces figures/phase12_universal.png:
  (a) the full-adder truth table the ground state computes — inputs (a, b, cin) → outputs
      (sum, cout) — every row correct;
  (b) circuit variable count vs ripple-adder width k (~14·k), against the exact engine's
      22-variable ceiling: a 1-bit adder fits, wider ones are the principle, not solved here.

Run:  python experiments/phase12_universal.py
"""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from drift.circuits import Circuit, adder_truth_table, full_adder  # noqa: E402

BG, FG, A1, A2 = "#0f1117", "#e2e8f0", "#5ac4b8", "#aa96f0"


def _ripple_vars(k: int) -> int:
    c = Circuit()
    carry = "c0"
    for i in range(k):
        full_adder(c, f"a{i}", f"b{i}", carry, f"s{i}", f"c{i + 1}")
        carry = f"c{i + 1}"
    return c.n


def main() -> None:
    c = Circuit()
    full_adder(c, "a", "b", "cin", "sum", "cout")
    rows = adder_truth_table(c, "a", "b", "cin", "sum", "cout")

    print(f"1-bit full adder as an Ising ground state ({c.n} variables):")
    print("  a b cin | sum cout | a+b+cin  ok")
    for r in rows:
        print(f"  {r['a']} {r['b']}  {r['cin']}  |  {r['sum']}   {r['cout']}   |   "
              f"{r['expected']}      {r['ok']}")

    grid = np.array([[r["a"], r["b"], r["cin"], r["sum"], r["cout"]] for r in rows], float)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=BG)

    ax1.imshow(grid, cmap="cividis", aspect="auto", vmin=0, vmax=1)
    ax1.axvline(2.5, color=FG, lw=2)  # divider between inputs and outputs
    ax1.set_xticks(range(5))
    ax1.set_xticklabels(["a", "b", "cin", "sum", "cout"], color=FG)
    ax1.set_yticks(range(8))
    ax1.set_yticklabels([f"{r['a']}{r['b']}{r['cin']}" for r in rows], color=FG)
    ax1.set_ylabel("input (a b cin)", color=FG)
    for i, r in enumerate(rows):
        for j, v in enumerate(grid[i]):
            ax1.text(j, i, int(v), ha="center", va="center",
                     color=(FG if v < 0.5 else BG), fontsize=10)
    ax1.set_title("(a) full-adder truth table — computed by the ground state", color=FG)

    ks = np.arange(1, 7)
    nv = np.array([_ripple_vars(int(k)) for k in ks])
    ax2.plot(ks, nv, "o-", color=A2, label="ripple-adder variables")
    ax2.axhline(22, color=A1, lw=1.2, ls="--", label="exact engine ceiling (22)")
    ax2.set_xlabel("ripple-adder width  k (bits)", color=FG)
    ax2.set_ylabel("# circuit variables", color=FG)
    ax2.set_title("(b) wider circuits hit the same wall", color=FG)
    ax2.set_facecolor(BG)
    ax2.tick_params(colors=FG)
    for sp in ax2.spines.values():
        sp.set_color(FG)
    leg = ax2.legend(facecolor=BG, edgecolor=FG, labelcolor=FG, fontsize=9)
    leg.get_frame().set_alpha(0.6)

    fig.suptitle("DRIFT P12 — universal computation: any Boolean function as a ground state",
                 color=FG, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(os.path.dirname(__file__), "..", "figures", "phase12_universal.png")
    fig.savefig(out, dpi=130, facecolor=BG, bbox_inches="tight")
    print(f"wrote {os.path.normpath(out)}")


if __name__ == "__main__":
    main()
