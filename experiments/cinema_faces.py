"""Cinema: three faces of one Ising engine, filmed while they anneal.

Each film is simulated annealing on a different (J, h), built by DRiFT's own builders,
with the same Metropolis rule as drift.solvers.annealing (same RNG draws, so the final
frame is exactly what `simulated_annealing(model, seed=...)` ends in). Nothing is drawn
that the spins did not do:

    optimization   MaxCut on a 14-node random graph; the cut is checked against the
                   exact ground state (all 2^14 configurations).
    self-assembly  a 3x3 jigsaw of Wang tiles; every internal edge has its own glue, so
                   the only tiling with all 12 bonds is the intended picture.
    replication    a frustrated 2D crystal (ferro J1, antiferro J2 along x); the ground
                   state is period-4 stripes, the unit cell copied across the lattice.

The fourth face, memory, is cinema_recall.py (a Hopfield recall of the word "DRiFT").

Writes figures/cinema_maxcut.gif, figures/cinema_tiles.gif, figures/cinema_crystal.gif.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Polygon, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from drift.builders import (  # noqa: E402
    column_period, count_bonds, crystal_2d, cut_value, decode_tiling, jigsaw,
    maxcut_ising, qubo_to_ising, random_graph, tiles_qubo,
)
from drift.solvers.annealing import geometric_schedule  # noqa: E402
from drift.solvers.exact import exact_ground_state  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BG, FG, DIM = "#050806", "#d9f2e3", "#6f8f78"
CYAN, MAGENTA, LIME = "#2ab0e8", "#c2418f", "#5ef0a0"


def anneal_film(model, n_sweeps, T0, T1, seed, n_frames):
    """Metropolis annealing identical to drift.solvers.annealing.simulated_annealing,
    returning snapshots (sweep, T, s, E) at geometrically spaced sweeps plus the last."""
    rng = np.random.default_rng(seed)
    n = model.n
    s = (rng.integers(0, 2, n) * 2 - 1).astype(np.float64)
    Ts = geometric_schedule(n_sweeps, T0, T1)
    marks = set(np.unique(np.geomspace(1, n_sweeps, n_frames).astype(int) - 1).tolist())
    E = model.energy(s)
    frames = [(0, Ts[0], s.copy(), E)]
    for t in range(n_sweeps):
        inv_T = 1.0 / max(Ts[t], 1e-12)
        for _ in range(n):
            i = int(rng.integers(n))
            dE = model.delta_energy_flip(s, i)
            if dE <= 0.0 or rng.random() < np.exp(-dE * inv_T):
                s[i] = -s[i]
                E += dE
        if t in marks:
            frames.append((t + 1, Ts[t], s.copy(), E))
    return frames


def grab(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=BG)
    plt.close(fig)
    return Image.open(buf).convert("RGB").quantize(colors=64, method=Image.MEDIANCUT)


def to_gif(imgs, out, duration, hold):
    imgs = imgs + [imgs[-1]] * hold
    imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=duration, loop=0,
                 optimize=True)
    print(f"wrote {out} ({out.stat().st_size / 2**20:.2f} MiB, {len(imgs)} frames)")


def canvas(w=6.4, h=6.0):
    fig = plt.figure(figsize=(w, h), dpi=90)
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0.04, 0.10, 0.92, 0.86])
    ax.set_facecolor(BG)
    ax.axis("off")
    return fig, ax


def caption(fig, text):
    fig.text(0.5, 0.035, text, color=DIM, ha="center", fontsize=10, family="monospace")


# ── optimization: MaxCut ────────────────────────────────────────────────────────────
def film_maxcut():
    W = random_graph(n=14, p=0.5, seed=2)
    model = maxcut_ising(W)
    s_exact, e_exact, _ = exact_ground_state(model, max_n=18)
    best_cut = cut_value(W, s_exact)
    for seed in range(32):
        frames = anneal_film(model, 400, 5.0, 0.01, seed, 70)
        if np.isclose(frames[-1][3], e_exact):
            break
    else:
        raise RuntimeError("no seed ended in the exact optimum")
    n = model.n
    try:
        import networkx as nx
        G = nx.Graph()
        G.add_edges_from([(i, j) for i in range(n) for j in range(i + 1, n) if W[i, j]])
        pos = {k: np.asarray(v) for k, v in nx.spring_layout(G, seed=7).items()}
    except ImportError:
        ang = 2 * np.pi * np.arange(n) / n
        pos = {i: np.array([np.cos(a), np.sin(a)]) for i, a in enumerate(ang)}
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if W[i, j]]
    figs = []
    for _, T, s, _ in frames:
        fig, ax = canvas()
        for i, j in edges:
            cut = s[i] != s[j]
            ax.plot(*zip(pos[i], pos[j]), color=LIME if cut else "#2a3a30",
                    lw=2.2 if cut else 0.8, alpha=0.95 if cut else 0.6, zorder=1)
        for i in range(n):
            ax.scatter(*pos[i], s=260, color=CYAN if s[i] > 0 else MAGENTA,
                       edgecolor=BG, linewidth=1.5, zorder=2)
        ax.set_aspect("equal")
        caption(fig, f"T = {T:5.2f}   cut = {cut_value(W, s):2.0f} / {best_cut:.0f} edges "
                     f"(exact optimum)")
        figs.append(grab(fig))
    assert cut_value(W, frames[-1][2]) == best_cut
    to_gif(figs, ROOT / "figures" / "cinema_maxcut.gif", 90, 16)
    return best_cut, n, seed


# ── self-assembly: Wang tiles ───────────────────────────────────────────────────────
def draw_tiles(ax, tiles, bits, L, K):
    x = ((np.asarray(bits) + 1.0) / 2.0).reshape(L * L, K)
    grid = np.argmax(x, axis=1).reshape(L, L)
    onehot = (x.sum(axis=1) == 1).reshape(L, L)
    glue_cmap = plt.get_cmap("tab20")
    for r in range(L):
        for c in range(L):
            x0, y0 = c, L - 1 - r
            if not onehot[r, c]:
                ax.add_patch(Rectangle((x0 + 0.06, y0 + 0.06), 0.88, 0.88, fc="#11181a",
                                       ec="#2a3a30", lw=1.0, ls="--"))
                continue
            t = tiles[grid[r, c]]
            cx, cy = x0 + 0.5, y0 + 0.5
            quads = {"n": [(x0, y0 + 1), (x0 + 1, y0 + 1)], "e": [(x0 + 1, y0 + 1), (x0 + 1, y0)],
                     "s": [(x0 + 1, y0), (x0, y0)], "w": [(x0, y0), (x0, y0 + 1)]}
            for side, (a, b) in quads.items():
                g = getattr(t, side)
                fc = "#1b2320" if g == 0 else glue_cmap((g - 1) % 20)
                ax.add_patch(Polygon([a, b, (cx, cy)], closed=True, fc=fc, ec=BG, lw=1.2))
            ax.add_patch(Rectangle((cx - 0.17, cy - 0.17), 0.34, 0.34,
                                   fc="#f5d90a" if t.color > 0.5 else "#3b1f5c", ec=BG))
    # satisfied bonds: a bright seam on the shared edge
    for r in range(L):
        for c in range(L):
            if not onehot[r, c]:
                continue
            tk = tiles[grid[r, c]]
            if c + 1 < L and onehot[r, c + 1] and tk.e and tk.e == tiles[grid[r, c + 1]].w:
                ax.plot([c + 1, c + 1], [L - 1 - r + 0.12, L - r - 0.12], color=LIME, lw=3)
            if r + 1 < L and onehot[r + 1, c] and tk.s and tk.s == tiles[grid[r + 1, c]].n:
                ax.plot([c + 0.12, c + 0.88], [L - 1 - r, L - 1 - r], color=LIME, lw=3)
    ax.set_xlim(-0.1, L + 0.1)
    ax.set_ylim(-0.1, L + 0.1)
    ax.set_aspect("equal")
    return grid, onehot


def placed_bonds(tiles, grid, onehot):
    """count_bonds, restricted to cells that hold exactly one tile."""
    L = grid.shape[0]
    b = 0
    for r in range(L):
        for c in range(L):
            if not onehot[r, c]:
                continue
            tk = tiles[grid[r, c]]
            if c + 1 < L and onehot[r, c + 1] and tk.e and tk.e == tiles[grid[r, c + 1]].w:
                b += 1
            if r + 1 < L and onehot[r + 1, c] and tk.s and tk.s == tiles[grid[r + 1, c]].n:
                b += 1
    return b


def film_tiles():
    L = 3
    image = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=float)
    tiles, target = jigsaw(L, L, image)
    K = len(tiles)
    Q, _ = tiles_qubo(tiles, L, L)
    model, _ = qubo_to_ising(Q)
    max_b = count_bonds(tiles, target)
    # The phase-7 schedule (T 5 -> 0.005) touches the picture near T ~ 0.9 and then leaves
    # it: its final state was the picture in 0 of 40 seeds (the panel shows the best state
    # seen). Starting colder, T 2 -> 0.05, the final state is the picture in 3 of 12 seeds.
    sweeps = 20000
    for seed in range(12):
        frames = anneal_film(model, sweeps, 2.0, 0.05, seed, 80)
        grid, valid = decode_tiling(frames[-1][2], L, L, K)
        if valid and count_bonds(tiles, grid) == max_b:
            break
    else:
        raise RuntimeError("no seed froze into the picture")
    figs = []
    for _, T, s, _ in frames:
        fig, ax = canvas(5.6, 6.0)
        grid, onehot = draw_tiles(ax, tiles, s, L, K)
        b = placed_bonds(tiles, grid, onehot)
        caption(fig, f"T = {T:6.3f}   bonds {b:2d} / {max_b}   placed {int(onehot.sum())}/9")
        figs.append(grab(fig))
    to_gif(figs, ROOT / "figures" / "cinema_tiles.gif", 80, 18)
    return max_b, model.n, seed


# ── self-replication: the crystal ───────────────────────────────────────────────────
def film_crystal():
    R = C = 16
    model = crystal_2d(R, C)
    sweeps = 6000
    for seed in range(12):
        frames = anneal_film(model, sweeps, 4.0, 0.01, seed, 80)
        if column_period(frames[-1][2].reshape(R, C)) == 4:
            break
    else:
        raise RuntimeError("no seed froze into the period-4 crystal")
    figs = []
    cmap = matplotlib.colors.ListedColormap([MAGENTA, CYAN])
    for _, T, s, E in frames:
        fig, ax = canvas(6.0, 6.2)
        ax.imshow(s.reshape(R, C), cmap=cmap, vmin=-1, vmax=1, interpolation="nearest")
        p = column_period(s.reshape(R, C))
        caption(fig, f"T = {T:5.2f}   E = {E:6.0f}   unit cell period = "
                     f"{p if p < C else '-'}")
        figs.append(grab(fig))
    to_gif(figs, ROOT / "figures" / "cinema_crystal.gif", 80, 18)
    return frames[-1][3], model.n, seed


def main():
    cut, n1, s1 = film_maxcut()
    print(f"maxcut: {n1} spins, final cut {cut:.0f} = exact optimum (seed {s1})")
    bonds, n2, s2 = film_tiles()
    print(f"tiles: {n2} spins, final state {bonds}/{bonds} bonds (seed {s2})")
    e, n3, s3 = film_crystal()
    print(f"crystal: {n3} spins, final period 4, E = {e:.0f} (seed {s3})")


if __name__ == "__main__":
    main()
