"""Cinema: a memory recalls a mycelium out of static.

The pattern is a grown mycelium: hyphal tips advance with a persistent random heading
and branch at random, from a few spores. DRiFT's Hopfield builder stores it as a ground
state of one Ising model (Hebbian couplings, h = 0), next to two random patterns. The cue
is the mycelium with 40 % of its spins flipped, and asynchronous zero-temperature
updates recall it, one spin at a time, each flip lowering the energy.

Every frame is the spin state. The glow is rendering only: a blurred copy of the same
spins added for light, nothing the dynamics did not do. The background is GitHub's dark
page colour, so the film sits on the page without a frame.

Writes figures/cinema_mycelium.gif.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from drift.builders.hopfield import add_noise, hopfield_model, overlap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
W, H, PX = 160, 40, 6                       # 6400 spins, drawn 960 x 240
PAGE = np.array([13, 17, 23], float)        # #0d1117, GitHub dark
HYPHA = np.array([126, 240, 200], float)    # the living threads
CORE = np.array([236, 255, 246], float)


def grow_mycelium(seed: int = 5, density: float = 0.15) -> np.ndarray:
    rng = np.random.default_rng(seed)
    grid = np.zeros((H, W), bool)
    spores = [(0.12 * W, 0.50 * H), (0.38 * W, 0.35 * H), (0.63 * W, 0.62 * H), (0.88 * W, 0.42 * H)]
    tips = [[x, y, a, 0] for x, y in spores for a in rng.uniform(0, 2 * np.pi, 4)]
    while tips and grid.mean() < density:
        new = []
        for x, y, a, age in tips:
            a += rng.normal(0.0, 0.22)
            x, y = x + np.cos(a) * 0.8, y + np.sin(a) * 0.8
            if not (0 <= x < W and 0 <= y < H) or age > 260:
                continue
            grid[int(y), int(x)] = True
            new.append([x, y, a, age + 1])
            if rng.random() < 0.028:
                new.append([x, y, a + rng.choice([-1, 1]) * rng.uniform(0.5, 1.1), age + 1])
        tips = new
    return np.where(grid, 1.0, -1.0).ravel()


def paint(s: np.ndarray) -> Image.Image:
    on = Image.fromarray(((s.reshape(H, W) > 0) * 255).astype(np.uint8)).resize(
        (W * PX, H * PX), Image.NEAREST)
    base = np.asarray(on.filter(ImageFilter.GaussianBlur(1.2)), float) / 255.0
    glow = np.asarray(on.filter(ImageFilter.GaussianBlur(7)), float) / 255.0
    light = np.clip(0.8 * base + 0.45 * glow, 0, 1)[..., None]
    rgb = PAGE + (HYPHA - PAGE) * np.clip(light * 1.15, 0, 1) + (CORE - HYPHA) * np.clip(
        base[..., None] - 0.6, 0, 1)
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))


def main() -> None:
    target = grow_mycelium()
    rng_p = np.random.default_rng(11)
    patterns = np.stack([target, rng_p.choice([-1.0, 1.0], W * H), rng_p.choice([-1.0, 1.0], W * H)])
    model = hopfield_model(patterns)
    s = add_noise(target, flip_frac=0.40, seed=7)
    n = s.size
    per_frame = n // 30
    rng = np.random.default_rng(3)
    energies, overlaps, states = [model.energy(s)], [overlap(s, target)], [s.copy()]
    for k, i in enumerate(np.concatenate([rng.permutation(n) for _ in range(3)]), start=1):
        s[i] = 1.0 if model.J[i] @ s + model.h[i] >= 0 else -1.0
        if k % per_frame == 0:
            energies.append(model.energy(s))
            overlaps.append(overlap(s, target))
            states.append(s.copy())
    frames = [paint(st).quantize(colors=48, method=Image.MEDIANCUT) for st in states]
    frames = [frames[0]] * 8 + frames + [frames[-1]] * 30
    out = ROOT / "figures" / "cinema_mycelium.gif"
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=80, loop=0, optimize=True)
    e = np.array(energies)
    print(f"wrote {out} ({out.stat().st_size / 2**20:.1f} MiB); {n} spins, "
          f"density {np.mean(target > 0):.2f}; overlap {overlaps[0]:+.3f} -> {overlaps[-1]:+.3f}; "
          f"energy {energies[0]:.0f} -> {energies[-1]:.0f}; monotone: {bool(np.all(np.diff(e) <= 1e-9))}")


if __name__ == "__main__":
    main()
