"""Cinema: a memory made of spins remembers, by lowering its energy.

DRiFT's Hopfield builder (drift.builders.hopfield) stores three memories as
the ground states of one Ising model (Hebbian couplings, h = 0): the word
"DRiFT" and two random patterns. The cue is "DRiFT" with 40 % of its spins
flipped.

Why random companions, not three words: three words share most of their dark
background, so they are strongly correlated, and the dynamics fell into a
*spurious mixture* of all three. That is the classic Hopfield failure, and it
was measured (overlap 0.848, three words superposed). Random patterns are
uncorrelated with the word, so the memory the cue belongs to is recovered. Asynchronous dynamics then flip one spin at a
time toward its local field. Every accepted flip lowers the energy, measured
with the model's own `energy`, until the pattern is recalled.

This is not an animation of an idea: every frame is the state of the spins,
and the energy and overlap are DRiFT's own numbers.

Writes figures/cinema_recall.gif.
"""

from __future__ import annotations

import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from drift.builders.hopfield import add_noise, hopfield_model, overlap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
W, H = 96, 24
BG = "#050806"


def word(text: str) -> np.ndarray:
    img = Image.new("L", (W, H), 0)
    font = ImageFont.truetype(font_manager.findfont("DejaVu Sans:bold"), 20)
    draw = ImageDraw.Draw(img)
    box = draw.textbbox((0, 0), text, font=font)
    draw.text(((W - (box[2] - box[0])) / 2 - box[0], (H - (box[3] - box[1])) / 2 - box[1]),
              text, fill=255, font=font)
    return np.where(np.asarray(img) > 100, 1.0, -1.0).ravel()


def main() -> None:
    rng_p = np.random.default_rng(11)
    patterns = np.stack([word("DRiFT"), rng_p.choice([-1.0, 1.0], W * H),
                         rng_p.choice([-1.0, 1.0], W * H)])
    model = hopfield_model(patterns)
    target = patterns[0]
    s = add_noise(target, flip_frac=0.40, seed=7)
    rng = np.random.default_rng(3)
    n = s.size
    per_frame = n // 24
    energies, overlaps, frames_s = [model.energy(s)], [overlap(s, target)], [s.copy()]
    order = np.concatenate([rng.permutation(n) for _ in range(3)])
    for k, i in enumerate(order, start=1):
        field = model.J[i] @ s + model.h[i]
        s[i] = 1.0 if field >= 0 else -1.0
        if k % per_frame == 0:
            energies.append(model.energy(s))
            overlaps.append(overlap(s, target))
            frames_s.append(s.copy())
    frames_s += [frames_s[-1]] * 18
    energies += [energies[-1]] * 18
    overlaps += [overlaps[-1]] * 18

    frames = []
    e = np.array(energies)
    for state in frames_s:
        # only the spins: the numbers (overlap, energy) are printed, and live in the README
        fig = plt.figure(figsize=(12, 3.2), dpi=80)
        fig.patch.set_facecolor(BG)
        ax = fig.add_axes([0.01, 0.02, 0.98, 0.96])
        ax.imshow(np.where(state.reshape(H, W) > 0, 1.0, 0.08), cmap="viridis", vmin=0, vmax=1,
                  interpolation="nearest")
        ax.axis("off")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", facecolor=BG)
        plt.close(fig)
        frames.append(Image.open(buf).convert("RGB").quantize(colors=32, method=Image.MEDIANCUT))
    out = ROOT / "figures" / "cinema_recall.gif"
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=90, loop=0, optimize=True)
    print(f"wrote {out} ({out.stat().st_size / 2**20:.1f} MiB); overlap {overlaps[0]:+.3f} -> "
          f"{overlaps[-1]:+.3f}; energy {energies[0]:.0f} -> {energies[-1]:.0f}; "
          f"monotone: {bool(np.all(np.diff(e) <= 1e-9))}")


if __name__ == "__main__":
    main()
