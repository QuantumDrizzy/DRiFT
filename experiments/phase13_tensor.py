"""Experiment: Phase 13 — the tensor-network ground state (χ as the engine).

Three windows onto the claim that DRIFT's microscope is now its own engine:

  (a) Correctness — the MPS ground energy lands exactly on the exact-Lanczos value across the
      phase diagram, for every chain the exact engine can still reach.
  (b) The χ peak — at a chain far past the exact wall, the effective bond dimension (the SAME
      Phase-3 ruler) rises into the quantum-critical region and collapses in the disordered
      phase. The cost of computation, produced by the solver itself.
  (c) Past the wall — the critical energy density marches toward the thermodynamic value −4/π
      as n grows, on chains (n = 48) with 2⁴⁸ ≈ 2.8×10¹⁴ configurations no exact engine touches.

    python -m experiments.phase13_tensor
"""

from __future__ import annotations

import numpy as np

from drift import mps
from drift.quantum import ising_chain_1d, tfim_terms
from drift.quantum import ground_state as exact_ground_state


def _exact(n: int, gamma: float) -> float:
    h_zz, h_x = tfim_terms(ising_chain_1d(n, 1.0))
    e0, _ = exact_ground_state(h_zz + gamma * h_x)
    return e0


def main(outdir: str = "figures") -> None:
    e_inf = -4.0 / np.pi

    # (a) correctness vs exact ------------------------------------------------
    print("(a) MPS vs exact Lanczos:")
    va_exact, va_mps = [], []
    worst = 0.0
    for n in (8, 10, 12, 14):
        for g in (0.5, 1.0, 2.0):
            e0 = _exact(n, g)
            em = mps.ground_state(n, gamma=g, chi_max=32).energy
            va_exact.append(e0)
            va_mps.append(em)
            worst = max(worst, abs(e0 - em))
    print(f"    worst |E_mps - E_exact| = {worst:.2e}  over {len(va_exact)} points\n")

    # (b) the χ peak, past the wall ------------------------------------------
    print("(b) effective χ and entropy vs Γ  (n=32, past the exact wall):")
    gammas = np.array([0.2, 0.4, 0.6, 0.77, 0.9, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0])
    chi_b, ent_b = [], []
    for g in gammas:
        r = mps.ground_state(32, gamma=float(g), chi_max=24)
        chi_b.append(r.chi)
        ent_b.append(r.max_entropy)
        print(f"    Γ={g:4.2f}  χ_eff={r.chi:2d}  S_max={r.max_entropy:.3f}")
    g_peak = float(gammas[int(np.argmax(chi_b))])
    print(f"    χ peaks at Γ ≈ {g_peak:.2f}  (finite-size shift below Γ_c=1)\n")

    # (c) finite-size toward −4/π --------------------------------------------
    print("(c) critical energy density vs n  (Γ=1):")
    ns = [12, 16, 24, 32, 48]
    ed = []
    for n in ns:
        r = mps.ground_state(n, gamma=1.0, chi_max=24)
        ed.append(r.energy_per_spin)
        print(f"    n={n:3d}  E/n={r.energy_per_spin:.5f}  (Δ to −4/π = {r.energy_per_spin - e_inf:+.5f})")
    print(f"    thermodynamic limit  −4/π = {e_inf:.5f}\n")

    # ── figure ───────────────────────────────────────────────────────────────
    try:
        from drift.viz import _save, _style, AMBER, CYAN, GRID, LIME, MAGENTA, MUTED, PANEL
    except Exception as exc:  # matplotlib missing
        print(f"  (skipping figure: {exc})")
        return

    _style()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    ax = axes[0]
    lo, hi = min(va_exact) - 1, max(va_exact) + 1
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1, ls="--")
    ax.scatter(va_exact, va_mps, s=42, color=CYAN, edgecolor="none", alpha=0.9)
    ax.set_xlabel("exact ground energy  (Lanczos)")
    ax.set_ylabel("MPS ground energy  (TEBD)")
    ax.set_title(f"(a) correct: max error {worst:.0e}")

    ax = axes[1]
    ln_chi, = ax.plot(gammas, chi_b, "o-", color=MAGENTA, lw=2, label=r"effective $\chi$ (left)")
    ax.axvline(1.0, color=GRID, ls="--", lw=1.2)
    ax.axvline(g_peak, color=AMBER, ls=":", lw=1.5)
    ax.annotate(f"χ peak\nΓ≈{g_peak:.2f}", (g_peak, max(chi_b)), color=AMBER,
                fontsize=8, ha="center", va="bottom")
    ax2 = ax.twinx()
    ln_ent, = ax2.plot(gammas, ent_b, "s-", color=LIME, lw=1.4, alpha=0.8,
                       label="entropy, bits (right)")
    ax2.tick_params(axis="y", colors=LIME)
    ax2.grid(False)
    ax.set_xlabel(r"transverse field  $\Gamma / J$")
    ax.set_ylabel(r"effective bond dimension  $\chi$", color=MAGENTA)
    ax.set_title("(b) χ peaks at criticality  (n=32)")
    ax.legend(handles=[ln_chi, ln_ent], loc="upper right", fontsize=8,
              facecolor=PANEL, edgecolor=GRID, labelcolor="#c8d6e0")

    ax = axes[2]
    inv = [1.0 / n for n in ns]
    ax.plot(inv, ed, "o-", color=CYAN, lw=2, label="MPS  E/n")
    ax.axhline(e_inf, color=AMBER, ls="--", lw=1.5)
    ax.annotate(r"$-4/\pi$", (max(inv) * 0.5, e_inf), color=AMBER, fontsize=10, va="bottom")
    for x, y, n in zip(inv, ed, ns):
        ax.annotate(f"n={n}", (x, y), color=MUTED, fontsize=7, xytext=(3, -8),
                    textcoords="offset points")
    ax.set_xlabel(r"$1/n$")
    ax.set_ylabel("critical energy density  E/n")
    ax.set_title("(c) past the wall → thermodynamic limit")

    fig.subplots_adjust(wspace=0.45)
    path = _save(fig, f"{outdir}/phase13_tensor.png")
    plt.close(fig)
    print(f"  saved {path}")


if __name__ == "__main__":
    main()
