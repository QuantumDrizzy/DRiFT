"""
Phase 7 — The microscope (synthesis).

Two figures that close DRIFT's argument:

  (1) four-faces panel — optimization, self-assembly, self-replication and memory built on
      one engine. Each panel is a ground state of a different (J, h), nothing else changes.

  (2) cosmic roofline — energy per operation for real systems against the Landauer floor.
      The headroom toward computronium, with honest numbers.

Understanding goal: see at a glance that one Hamiltonian framework already covers four
distinct phenomena, and where physical hardware sits against the ultimate physical limits.

Run:  python experiments/phase7_microscope.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from drift.builders import (  # noqa: E402
    maxcut_ising, random_graph, cut_value,
    jigsaw, tiles_qubo, decode_tiling, count_bonds, render, qubo_to_ising,
    crystal_2d, column_period,
    hopfield_model, recall, add_noise, overlap,
)
from drift.solvers.exact import exact_ground_state  # noqa: E402
from drift.solvers.annealing import simulated_annealing  # noqa: E402
from drift import viz  # noqa: E402


def face_optimization():
    W = random_graph(n=14, p=0.5, seed=2)
    model = maxcut_ising(W)
    s, e, _ = exact_ground_state(model, max_n=18)
    cv = cut_value(W, s)
    return {"name": "(1) optimization  -  MaxCut",
            "metric": f"cut = {cv:.0f} edges  |  E = {e:.1f}  |  n = {model.n} spins",
            "kind": "graph", "data": (W, s)}


def face_assembly():
    L = 3
    image = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=float)
    tiles, target = jigsaw(L, L, image)
    K = len(tiles)
    Q, _ = tiles_qubo(tiles, L, L)
    model, _ = qubo_to_ising(Q)
    max_b = count_bonds(tiles, target)
    best_e, best_s = np.inf, None
    for seed in range(8):
        s, e, _, _ = simulated_annealing(model, n_sweeps=20000, T0=5.0, T1=0.005, seed=seed)
        if e < best_e:
            best_e, best_s = e, s
    grid, _ = decode_tiling(best_s, L, L, K)
    b = count_bonds(tiles, grid)
    return {"name": "(2) self-assembly  -  Wang tiles",
            "metric": f"{b}/{max_b} bonds  |  n = {model.n} spins",
            "kind": "grid", "data": render(tiles, grid), "vrange": (0, 1)}


def face_replication():
    R = C = 12
    model = crystal_2d(R, C)
    best_e, best_s = np.inf, None
    for seed in range(6):
        s, e, _, _ = simulated_annealing(model, n_sweeps=8000, T0=4.0, T1=0.01, seed=seed)
        if e < best_e:
            best_e, best_s = e, s
    p = column_period(best_s.reshape(R, C))
    return {"name": "(3) self-replication  -  crystal",
            "metric": f"unit cell period = {p}  |  E = {best_e:.0f}  |  n = {model.n} spins",
            "kind": "grid", "data": best_s.reshape(R, C), "vrange": (-1, 1)}


def face_memory():
    L = 10
    x = -np.ones((L, L))
    for i in range(L):
        x[i, i] = 1.0
        x[i, L - 1 - i] = 1.0
    fr = -np.ones((L, L)); fr[0, :] = fr[-1, :] = fr[:, 0] = fr[:, -1] = 1.0
    t = -np.ones((L, L)); t[0:2, :] = 1.0; t[:, L // 2 - 1:L // 2 + 1] = 1.0
    patterns = np.array([x.ravel(), fr.ravel(), t.ravel()])
    model = hopfield_model(patterns)
    cue = add_noise(patterns[0], flip_frac=0.25, seed=2)
    rec = recall(model, cue, seed=1)
    m = overlap(rec, patterns[0])
    return {"name": "(4) memory  -  Hopfield",
            "metric": f"recovered overlap m = {m:+.2f}  |  n = {model.n} spins",
            "kind": "grid", "data": rec.reshape(L, L), "vrange": (-1, 1)}


def cosmic_roofline_systems():
    """Honest order-of-magnitude landmarks, log J/op. Sources: textbook physics
    (Landauer kT ln 2; Margolus-Levitin), public-domain device characterisations,
    and standard estimates for biology. Numbers are landmarks, not precision claims."""
    AMBER = "#d95f02"; CYAN = "#0b6ea8"; MAGENTA = "#a3327d"; LIME = "#2e7d32"
    return [
        ("Margolus-Levitin @ 1 GHz",      1.65e-25, LIME),
        ("Margolus-Levitin @ 1 THz",      1.65e-22, LIME),
        ("Photonic / superconducting",    1e-19,    CYAN),
        ("DNA polymerase (per bp)",       1e-19,    MAGENTA),
        ("Synapse (per event)",           1e-14,    MAGENTA),
        ("Modern CMOS logic gate",        1e-15,    CYAN),
        ("GPU FP16 op (RTX 5060 Ti, est.)", 5e-12,  CYAN),
        ("DDR5 DRAM access",              5e-11,    AMBER),
    ]


def main():
    print("Phase 7 - the microscope: assembling four faces under the same engine")
    panels = [
        face_optimization(),
        face_assembly(),
        face_replication(),
        face_memory(),
    ]
    for p in panels:
        print(f"  {p['name']:<35}  {p['metric']}")

    viz.plot_four_faces(panels,
                        title="DRIFT - four faces, one Ising engine",
                        out="figures/phase7_four_faces.png")
    print("  figure -> figures/phase7_four_faces.png")

    systems = cosmic_roofline_systems()
    viz.plot_cosmic_roofline(systems,
                             title="Cosmic roofline - real systems vs. the Landauer floor",
                             out="figures/phase7_roofline.png")
    print("  figure -> figures/phase7_roofline.png")


if __name__ == "__main__":
    main()
