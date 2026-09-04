"""
drift.viz — visualization. Dark 'DRIFT' palette, consistent with the iNFAMØUS look.

Every figure is a window onto the process, not a result to publish: you watch the
energy drift down, watch the spins order, see where the ground state sits in the
spectrum. Saved as PNG (Agg backend) so experiments run headless.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ── palette ───────────────────────────────────────────────────────────────────
# Light theme: these figures are published in the README and in PDFs, both
# read on white. The accents are dark enough to stay legible there and to
# survive greyscale printing. Names kept for continuity with call sites.
BG, PANEL = "#ffffff", "#f6f7f9"
CYAN, MAGENTA, AMBER, LIME = "#0b6ea8", "#a3327d", "#d95f02", "#2e7d32"
TEXT, MUTED, GRID = "#1a1d21", "#5b6672", "#d8dce0"
FOOTER = "DRIFT · a microscope for physical computation"


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT, "text.color": TEXT,
        "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID,
        "axes.grid": True, "grid.alpha": 0.5, "grid.linewidth": 0.7,
        "axes.titlecolor": TEXT, "axes.titlesize": 13, "axes.titleweight": "bold",
        "font.size": 11, "font.family": "DejaVu Sans Mono", "figure.dpi": 140,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def _save(fig, out: str | Path) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.text(0.99, 0.01, FOOTER, ha="right", va="bottom", color=MUTED, fontsize=7.2, style="italic")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_relaxation(traj, *, e_ground=None, Ts=None, title="Relaxation", out="figures/relaxation.png"):
    """Energy vs. sweep — the system drifting downhill. Optional ground-state line and
    a temperature curve on a twin axis."""
    _style()
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    steps = np.arange(len(traj))
    ax.plot(steps, traj, color=CYAN, lw=1.8, label="energy (annealing)")
    if e_ground is not None:
        ax.axhline(e_ground, color=LIME, lw=1.6, ls="--", label=f"exact ground state ({e_ground:.3f})")
    ax.set_xlabel("sweep")
    ax.set_ylabel("energy  E(s)")
    ax.set_title(title)
    if Ts is not None:
        axT = ax.twinx()
        axT.plot(steps, Ts, color=AMBER, lw=1.2, alpha=0.7, label="temperature")
        axT.set_ylabel("temperature  T", color=AMBER)
        axT.tick_params(axis="y", colors=AMBER)
        axT.grid(False)
        axT.set_yscale("log")
    ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=9, loc="upper right")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return _save(fig, out)


def plot_spins(s, *, grid_shape=None, title="Ground state", out="figures/spins.png"):
    """Show a spin configuration. 1-D → signed bars; 2-D → a colored lattice."""
    _style()
    s = np.asarray(s)
    if grid_shape is not None:
        fig, ax = plt.subplots(figsize=(4.6, 4.6))
        ax.imshow(s.reshape(grid_shape), cmap=plt.matplotlib.colors.ListedColormap([MAGENTA, CYAN]),
                  vmin=-1, vmax=1, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        ax.set_title(title)
    else:
        fig, ax = plt.subplots(figsize=(8.0, 2.4))
        ax.bar(np.arange(len(s)), s, color=[CYAN if v > 0 else MAGENTA for v in s], width=0.9)
        ax.set_ylim(-1.2, 1.2); ax.set_xlabel("spin index"); ax.set_yticks([-1, 0, 1])
        ax.set_title(title)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return _save(fig, out)


def plot_landscape(all_energies, *, e_ground=None, title="Energy landscape", out="figures/landscape.png"):
    """Histogram of the full 2ⁿ energy spectrum — shows how rare the ground state is."""
    _style()
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.hist(all_energies, bins=80, color=CYAN, alpha=0.8)
    if e_ground is not None:
        ax.axvline(e_ground, color=LIME, lw=1.8, ls="--", label=f"ground state ({e_ground:.3f})")
        ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=9)
    ax.set_xlabel("energy  E(s)"); ax.set_ylabel("number of configurations")
    ax.set_title(title)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return _save(fig, out)


def plot_graph_cut(W, s, *, title="MaxCut partition", out="figures/cut.png"):
    """A weighted graph with its partition: nodes colored by side (±1), cut edges
    highlighted, uncut edges faded. Uses a spring layout if networkx is available."""
    _style()
    W = np.asarray(W)
    s = np.asarray(s)
    n = len(s)

    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if W[i, j] != 0]
    try:
        import networkx as nx
        G = nx.Graph()
        G.add_nodes_from(range(n))
        G.add_edges_from(edges)
        pos = {k: np.asarray(v) for k, v in nx.spring_layout(G, seed=7).items()}
    except Exception:
        ang = 2 * np.pi * np.arange(n) / max(n, 1)
        pos = {i: np.array([np.cos(a), np.sin(a)]) for i, a in enumerate(ang)}

    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    for i, j in edges:
        cut = s[i] != s[j]
        ax.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]],
                color=(LIME if cut else MUTED), lw=(1.9 if cut else 0.6),
                alpha=(0.9 if cut else 0.3), zorder=1)
    for i in range(n):
        ax.scatter(*pos[i], s=200, color=(CYAN if s[i] > 0 else MAGENTA),
                   edgecolor="white", linewidth=0.7, zorder=2)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False); ax.set_aspect("equal")
    ax.set_title(title)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return _save(fig, out)


def plot_recall(stored, noisy, recovered, L, *, overlaps=None,
                title="Associative recall", out="figures/recall.png"):
    """Three grids side by side: the stored memory, a noisy cue, and what the Hopfield
    dynamics recovered. Memory recall = relaxing to the nearest attractor."""
    _style()
    cmap = plt.matplotlib.colors.ListedColormap([MAGENTA, CYAN])
    labels = ["stored memory", "noisy cue", "recovered"]
    if overlaps is not None:
        labels = ["stored memory", f"noisy cue  (m={overlaps[0]:+.2f})",
                  f"recovered  (m={overlaps[1]:+.2f})"]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.7))
    for ax, g, lab in zip(axes, (stored, noisy, recovered), labels):
        ax.imshow(np.asarray(g).reshape(L, L), cmap=cmap, vmin=-1, vmax=1, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        ax.set_title(lab, color=TEXT, fontsize=10)
    fig.suptitle(title, color=CYAN, fontweight="bold", fontsize=13)
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    return _save(fig, out)


def plot_assembly(disordered, assembled, L, *, bonds=None, max_bonds=None,
                  title="Self-assembly", out="figures/assembly.png"):
    """Two grids: a hot, disordered placement and the assembled ground state. The tiles
    bind by glue affinity; the target structure is the arrangement with the most bonds."""
    _style()
    cmap = plt.matplotlib.colors.ListedColormap([MAGENTA, CYAN])
    right = "assembled"
    if bonds is not None and max_bonds is not None:
        right = f"assembled  ({bonds}/{max_bonds} bonds)"
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.9))
    for ax, g, lab in zip(axes, (disordered, assembled), ("disordered (hot)", right)):
        ax.imshow(np.asarray(g).reshape(L, L), cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        ax.set_title(lab, color=TEXT, fontsize=10)
    fig.suptitle(title, color=CYAN, fontweight="bold", fontsize=13)
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    return _save(fig, out)


def plot_crystal(disordered, crystal, rows, cols, *, period=None,
                 title="Self-replication", out="figures/crystal.png"):
    """Two lattices: a hot disordered melt and the crystallized periodic ground state.
    A box marks the replicated unit cell — the motif that copies itself across the lattice."""
    _style()
    cmap = plt.matplotlib.colors.ListedColormap([MAGENTA, CYAN])
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.3))
    for ax, g, lab in zip(axes, (disordered, crystal),
                          ("disordered melt (hot)", "crystal (periodic ground state)")):
        ax.imshow(np.asarray(g).reshape(rows, cols), cmap=cmap, vmin=-1, vmax=1,
                  interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        ax.set_title(lab, color=TEXT, fontsize=10)
    if period is not None:
        rect = plt.Rectangle((-0.5, -0.5), period, min(rows, 2), fill=False,
                             edgecolor=AMBER, lw=2.2)
        axes[1].add_patch(rect)
        axes[1].text(period / 2 - 0.5, min(rows, 2) - 0.4, "unit cell", color=AMBER,
                     fontsize=8.5, ha="center", va="top")
    fig.suptitle(title, color=CYAN, fontweight="bold", fontsize=13)
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    return _save(fig, out)


def plot_four_faces(panels, *, title="Four faces, one engine",
                    out="figures/four_faces.png"):
    """One image with the four faces in a 2x2 grid. Each panel: a small lattice or graph
    showing the ground state of that face. Each is just an (J, h) into the same Ising
    engine; only the shape of the coupling matrix changes from face to face.

    `panels` is a list of 4 dicts, each:
        {"name": str, "metric": str, "kind": "grid"|"graph",
         "data": np.ndarray  (grid: 2-D) or  (graph: (W, s)),
         "vrange": (lo, hi)   for grid only (defaults to data range)}
    """
    _style()
    cmap = plt.matplotlib.colors.ListedColormap([MAGENTA, CYAN])
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 8.4))
    axes = axes.ravel()
    for ax, p in zip(axes, panels):
        if p["kind"] == "grid":
            g = np.asarray(p["data"])
            lo, hi = p.get("vrange", (g.min(), g.max()))
            cm = cmap if (lo == -1 and hi == 1) else "viridis"
            ax.imshow(g, cmap=cm, vmin=lo, vmax=hi, interpolation="nearest")
        else:  # graph
            W, s = p["data"]
            n = len(s)
            try:
                import networkx as nx
                G = nx.Graph()
                G.add_nodes_from(range(n))
                G.add_edges_from([(i, j) for i in range(n) for j in range(i + 1, n) if W[i, j] != 0])
                pos = {k: np.asarray(v) for k, v in nx.spring_layout(G, seed=7).items()}
            except Exception:
                ang = 2 * np.pi * np.arange(n) / max(n, 1)
                pos = {i: np.array([np.cos(a), np.sin(a)]) for i, a in enumerate(ang)}
            for i in range(n):
                for j in range(i + 1, n):
                    if W[i, j] == 0:
                        continue
                    cut = s[i] != s[j]
                    ax.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]],
                            color=(LIME if cut else MUTED), lw=(1.8 if cut else 0.5),
                            alpha=(0.85 if cut else 0.3), zorder=1)
            for i in range(n):
                ax.scatter(*pos[i], s=120, color=(CYAN if s[i] > 0 else MAGENTA),
                           edgecolor="white", linewidth=0.6, zorder=2)
            ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        ax.set_title(p["name"], color=CYAN, fontsize=11, fontweight="bold", loc="left")
        ax.text(0.02, -0.05, p["metric"], color=AMBER, fontsize=9,
                transform=ax.transAxes, va="top")
    fig.suptitle(title, color=CYAN, fontweight="bold", fontsize=14)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    return _save(fig, out)


def plot_cosmic_roofline(systems, *, landauer_300k=2.87e-21, landauer_4k=3.83e-23,
                         title="Cosmic roofline - energy per operation",
                         out="figures/cosmic_roofline.png"):
    """Real and idealized systems on a log J/op axis, with the Landauer floor drawn as a
    physical wall. `systems` is a list of (label, joules_per_op, color) tuples, sorted by
    cost. The headroom between a system and the Landauer line is the gap toward
    computronium."""
    _style()
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    labels = [s[0] for s in systems]
    vals = np.array([s[1] for s in systems], dtype=float)
    colors = [s[2] for s in systems]

    y = np.arange(len(systems))
    ax.barh(y, vals, color=colors, edgecolor="white", linewidth=0.6, height=0.6, log=True)
    ax.set_yticks(y); ax.set_yticklabels(labels, color=TEXT, fontsize=10)
    ax.set_xscale("log")
    ax.set_xlabel("energy per operation  (J/op)")
    ax.set_title(title)

    ax.axvline(landauer_300k, color=LIME, lw=1.8, ls="--", alpha=0.9)
    ax.axvline(landauer_4k, color=AMBER, lw=1.4, ls=":", alpha=0.9)
    ax.text(landauer_300k, -0.85, "Landauer\n300 K", color=LIME,
            fontsize=8.5, ha="center", va="top")
    ax.text(landauer_4k, -0.85, "Landauer\n4 K", color=AMBER,
            fontsize=8.5, ha="center", va="top")

    xmax = max(vals) * 50
    for yi, v in zip(y, vals):
        ax.text(v * 1.6, yi, f"{v:.1e} J/op", color=MUTED, fontsize=8.5, va="center")

    ax.set_xlim(min(vals) * 0.05, xmax)
    ax.set_ylim(-1.7, len(systems) - 0.3)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return _save(fig, out)


def plot_chi_sweep(gammas, entropies, chis, *, gamma_c=None,
                   title="χ — how much the state computes", out="figures/chi.png"):
    """Entanglement entropy (cyan) and effective bond dimension χ (magenta) vs. the
    transverse field Γ. Both peak at the quantum phase transition — the state is most
    entangled, and hardest to compress, exactly there."""
    _style()
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    gammas = np.asarray(gammas)
    ax.plot(gammas, entropies, color=CYAN, lw=2.0, marker="o", ms=3.5,
            label="entanglement entropy S (bits)")
    ax.set_xlabel("transverse field  Γ  (Γ/J)")
    ax.set_ylabel("entropy S (bits)", color=CYAN)
    ax.tick_params(axis="y", colors=CYAN)
    ax.set_title(title)

    axC = ax.twinx()
    axC.plot(gammas, chis, color=MAGENTA, lw=2.0, marker="s", ms=3.5, label="effective χ")
    axC.set_ylabel("effective bond dimension  χ", color=MAGENTA)
    axC.tick_params(axis="y", colors=MAGENTA)
    axC.grid(False)

    if gamma_c is not None:
        ax.axvline(gamma_c, color=AMBER, lw=1.6, ls="--", alpha=0.8)
        ax.text(gamma_c, ax.get_ylim()[1] * 0.96, "  quantum phase transition",
                color=AMBER, fontsize=8.5, va="top")

    # only the data series, not the helper axvline (whose auto-label is "_child…")
    lines = [ln for ln in (ax.get_lines() + axC.get_lines()) if not ln.get_label().startswith("_")]
    ax.legend(lines, [ln.get_label() for ln in lines],
              facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=8.5, loc="upper right")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return _save(fig, out)
