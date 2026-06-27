"""Phase 11 experiment — integer factorization as an Ising ground state.

Factors a handful of semiprimes by finding the ground state of their factorization QUBO with
DRIFT's exact engine (matter computing arithmetic), then plots the honest reason it stops there.

Produces figures/phase11_factoring.png:
  (a) QUBO variable count vs the semiprime N — the problem size grows only ~ (log N)²;
  (b) the exact-solver search space 2^(#vars) vs N (log axis), with the engine's 2²² ceiling —
      the exponential wall that keeps this a demonstration, not an attack on RSA.

Run:  python experiments/phase11_factoring.py
"""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from drift.factoring import factor  # noqa: E402

BG, FG, A1, A2 = "#0f1117", "#e2e8f0", "#5ac4b8", "#aa96f0"

# (N, p, q) semiprimes; the small ones are actually solved, all are used for the scaling curve.
SEMIPRIMES = [
    (15, 3, 5), (21, 3, 7), (35, 5, 7), (77, 7, 11), (143, 11, 13), (221, 13, 17),
    (323, 17, 19), (437, 19, 23), (667, 23, 29), (1147, 31, 37), (1763, 41, 43),
    (3127, 53, 59), (5183, 71, 73), (10403, 101, 103), (39203, 191, 199),
]


def _nvars(p: int, q: int) -> int:
    pb, qb = p.bit_length(), q.bit_length()
    return (pb - 1) + (qb - 1) + (pb - 1) * (qb - 1)


def main() -> None:
    print("DRIFT factoring by ground state:")
    for (N, p, q) in SEMIPRIMES:
        if _nvars(p, q) <= 22:
            r = factor(N, p.bit_length(), q.bit_length())
            print(f"  N={N:<6} -> {r['p']} x {r['q']}   ok={r['ok']}  energy={r['energy']:.1f}")
        else:
            print(f"  N={N:<6} -> needs {_nvars(p, q)} vars (> 22): past the exact engine's wall")

    Ns = np.array([N for (N, _, _) in SEMIPRIMES])
    nv = np.array([_nvars(p, q) for (_, p, q) in SEMIPRIMES])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=BG)
    ax1.plot(Ns, nv, "o-", color=A1)
    ax1.set_title("(a) QUBO variables grow only ~ (log N)²", color=FG)
    ax1.set_ylabel("# QUBO variables", color=FG)

    ax2.semilogy(Ns, 2.0 ** nv, "o-", color=A2, label="search space 2^(#vars)")
    ax2.axhline(2 ** 22, color=A1, lw=1.2, ls="--", label="exact engine ceiling (2²²)")
    ax2.set_title("(b) …but the search space explodes — the wall", color=FG)
    ax2.set_ylabel("configurations to search", color=FG)
    leg = ax2.legend(facecolor=BG, edgecolor=FG, labelcolor=FG, fontsize=9)
    leg.get_frame().set_alpha(0.6)

    for ax in (ax1, ax2):
        ax.set_facecolor(BG)
        ax.set_xscale("log")
        ax.set_xlabel("semiprime N", color=FG)
        ax.tick_params(colors=FG)
        for sp in ax.spines.values():
            sp.set_color(FG)

    fig.suptitle("DRIFT P11 — factoring as a ground state: it works, and here is why it won't scale",
                 color=FG, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(os.path.dirname(__file__), "..", "figures", "phase11_factoring.png")
    fig.savefig(out, dpi=130, facecolor=BG, bbox_inches="tight")
    print(f"wrote {os.path.normpath(out)}")


if __name__ == "__main__":
    main()
