"""Cinema: a Hopfield memory recalls a logo out of static.

Any logo becomes a memory. The logo's coloured pixels are thresholded into spins (+1 on
the mark, -1 on the background), fitted into a wide 4:1 frame, and stored as a ground state
of one Ising model by DRiFT's Hopfield builder, next to two random patterns. The cue is the
logo with 40 % of its spins flipped; zero-temperature asynchronous updates recall it, one
spin at a time, each flip lowering the energy.

Every frame is the spin state. The glow is rendering only (a blurred copy of the same spins).
The background is GitHub's dark page colour, so the film sits on the page without a frame.

Usage:
    python experiments/cinema_logo.py LOGO.png OUT.gif [--colour FBBA02] [--cols 192]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from drift.builders.hopfield import add_noise, hopfield_model, overlap  # noqa: E402

PAGE = np.array([13, 17, 23], float)  # #0d1117, GitHub dark


def logo_pattern(path: Path, cols: int, rows: int, fill: float = 0.86) -> np.ndarray:
    """The logo's saturated, bright pixels as +1, fitted (aspect kept) into cols x rows."""
    a = np.asarray(Image.open(path).convert("RGB")).astype(float)
    mx, mn = a.max(axis=2), a.min(axis=2)
    mark = (mx > 120) & ((mx - mn) > 90)  # bright and saturated: the mark, not the black
    ys, xs = np.nonzero(mark)
    crop = mark[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape
    scale = min(rows * fill / h, cols * fill / w)
    tw, th = max(1, round(w * scale)), max(1, round(h * scale))
    small = Image.fromarray((crop * 255).astype(np.uint8)).resize((tw, th), Image.LANCZOS)
    grid = np.zeros((rows, cols), bool)
    oy, ox = (rows - th) // 2, (cols - tw) // 2
    grid[oy:oy + th, ox:ox + tw] = np.asarray(small) > 127
    return np.where(grid, 1.0, -1.0).ravel()


def paint(s: np.ndarray, cols: int, rows: int, px: int, colour: np.ndarray) -> Image.Image:
    on = Image.fromarray(((s.reshape(rows, cols) > 0) * 255).astype(np.uint8)).resize(
        (cols * px, rows * px), Image.NEAREST)
    base = np.asarray(on.filter(ImageFilter.GaussianBlur(0.8)), float) / 255.0
    glow = np.asarray(on.filter(ImageFilter.GaussianBlur(6)), float) / 255.0
    light = np.clip(0.9 * base + 0.35 * glow, 0, 1)[..., None]
    rgb = PAGE + (colour - PAGE) * light
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("logo", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--colour", default="FBBA02")
    ap.add_argument("--cols", type=int, default=192)
    args = ap.parse_args()
    cols, rows = args.cols, args.cols // 4
    px = max(1, 960 // cols)
    colour = np.array([int(args.colour[i:i + 2], 16) for i in (0, 2, 4)], float)

    target = logo_pattern(args.logo, cols, rows)
    rng_p = np.random.default_rng(11)
    patterns = np.stack([target, rng_p.choice([-1.0, 1.0], target.size),
                         rng_p.choice([-1.0, 1.0], target.size)])
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
    frames = [paint(st, cols, rows, px, colour).quantize(colors=48, method=Image.MEDIANCUT)
              for st in states]
    frames = [frames[0]] * 8 + frames + [frames[-1]] * 30
    frames[0].save(args.out, save_all=True, append_images=frames[1:], duration=80, loop=0,
                   optimize=True)
    e = np.array(energies)
    print(f"wrote {args.out} ({args.out.stat().st_size / 2**20:.1f} MiB); {n} spins; "
          f"overlap {overlaps[0]:+.3f} -> {overlaps[-1]:+.3f}; energy {energies[0]:.0f} -> "
          f"{energies[-1]:.0f}; monotone: {bool(np.all(np.diff(e) <= 1e-9))}")


if __name__ == "__main__":
    main()
