"""Scale sweep: wall-clock, solution quality, and χ vs n on the versioned instance bank.

Scientific question (falsifiable)
---------------------------------
How do wall-clock time and solution quality (and χ where an MPS/tensor path applies)
scale with system size n for fixed instance families, when solving via drift.solve
(exact → GPU-PT → CPU-PT)?

Classical families go through ``drift.solve``. χ is recorded only on the 1-D TFIM
chain via the existing MPS/TEBD solver, and only for n ≤ the documented cutoff
(default 16 on CPU CI; 24 is feasible locally; Phase 13 showed 48). A heuristic
minimum is never marked certified.

    python -m experiments.scale_sweep
    python -m experiments.scale_sweep --ci          # n≤10, no MPS, CI-safe
    python -m experiments.scale_sweep --n-max 32    # local / GPU ladder

GPU is optional: if ``cuda/ising_pt`` is missing, large n continue on CPU-PT.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from drift import gpu, mps, solve
from drift.benchmarks import (
    MPS_FAMILIES,
    MPS_N_MAX_SWEEP,
    BuiltInstance,
    iter_instances,
    load_manifest,
    scientific_question,
)
from drift.viz import AMBER, CYAN, GRID, LIME, MAGENTA, TEXT, _save, _style

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class SweepRow:
    id: str
    family: str
    n: int
    seed: int
    method: str
    certified: bool
    energy: float | None
    known_energy: float | None
    energy_error: float | None
    wall_s: float
    chi: int | None
    chi_kept: int | None
    notes: str

    def to_json(self) -> dict:
        return asdict(self)


def _honesty_check(method: str, certified: bool) -> None:
    """A heuristic (cpu-pt / gpu-pt / mps) is never a certified optimum."""
    if certified and method != "exact":
        raise RuntimeError(
            f"honesty contract violated: method={method!r} marked certified=True"
        )
    if method == "exact" and not certified:
        raise RuntimeError("exact solve must be certified")


def _energy_error(energy: float, known: float | None) -> float | None:
    if known is None:
        return None
    return float(energy - known)


def sweep_instance(
    inst: BuiltInstance,
    *,
    exact_max: int = 18,
    use_gpu: bool = True,
    n_replicas: int | None = None,
    n_rounds: int = 400,
    sweeps_per_round: int = 4,
    mps_n_max: int = MPS_N_MAX_SWEEP,
    mps_chi_max: int = 16,
    mps_max_sweeps: int = 12,
) -> SweepRow:
    """Solve one bank instance and record provenance, energy, wall time, optional χ."""
    spec = inst.spec
    notes: list[str] = []

    if spec.family in MPS_FAMILIES:
        if spec.n > mps_n_max:
            return SweepRow(
                id=spec.id, family=spec.family, n=spec.n, seed=spec.seed,
                method="skipped", certified=False, energy=None,
                known_energy=spec.known_energy, energy_error=None,
                wall_s=0.0, chi=None, chi_kept=None,
                notes=f"MPS skipped: n={spec.n} > mps_n_max={mps_n_max}",
            )
        gamma = float(spec.params.get("gamma", 1.0))
        j = float(spec.params.get("j", 1.0))
        t0 = time.perf_counter()
        result = mps.ground_state(
            spec.n, j=j, gamma=gamma, chi_max=mps_chi_max,
            max_sweeps=mps_max_sweeps, dt_schedule=[0.1, 0.02, 0.005],
        )
        wall = time.perf_counter() - t0
        _honesty_check("mps", False)
        notes.append("MPS/TEBD (variational; never certified)")
        return SweepRow(
            id=spec.id, family=spec.family, n=spec.n, seed=spec.seed,
            method="mps", certified=False, energy=float(result.energy),
            known_energy=spec.known_energy,
            energy_error=_energy_error(float(result.energy), spec.known_energy),
            wall_s=wall, chi=int(result.chi), chi_kept=int(result.chi_kept),
            notes="; ".join(notes),
        )

    if use_gpu and not gpu.gpu_available() and spec.n > exact_max:
        notes.append("GPU binary absent; continuing with CPU-PT")

    t0 = time.perf_counter()
    sol = solve(
        inst.model,
        exact_max=exact_max,
        use_gpu=use_gpu,
        n_replicas=n_replicas,
        n_rounds=n_rounds,
        sweeps_per_round=sweeps_per_round,
        seed=spec.seed,
    )
    wall = time.perf_counter() - t0
    _honesty_check(sol.method, sol.certified)
    if sol.certified:
        notes.append("certified exact enumeration")
    else:
        notes.append("heuristic minimum; not certified")
    return SweepRow(
        id=spec.id, family=spec.family, n=spec.n, seed=spec.seed,
        method=sol.method, certified=bool(sol.certified), energy=float(sol.energy),
        known_energy=spec.known_energy,
        energy_error=_energy_error(float(sol.energy), spec.known_energy),
        wall_s=wall, chi=None, chi_kept=None,
        notes="; ".join(notes),
    )


def run_sweep(
    *,
    n_min: int = 1,
    n_max: int | None = 32,
    families: tuple[str, ...] | None = None,
    exact_max: int = 18,
    use_gpu: bool = True,
    n_replicas: int | None = None,
    n_rounds: int = 400,
    sweeps_per_round: int = 4,
    mps_n_max: int = MPS_N_MAX_SWEEP,
    skip_mps: bool = False,
) -> list[SweepRow]:
    """Sweep the catalog. ``skip_mps`` drops the TFIM family entirely."""
    specs = load_manifest()
    if skip_mps:
        specs = [s for s in specs if s.family not in MPS_FAMILIES]
    rows = []
    for inst in iter_instances(n_min=n_min, n_max=n_max, families=families, specs=specs):
        rows.append(sweep_instance(
            inst, exact_max=exact_max, use_gpu=use_gpu,
            n_replicas=n_replicas, n_rounds=n_rounds,
            sweeps_per_round=sweeps_per_round, mps_n_max=mps_n_max,
        ))
    return rows


def write_results(rows: list[SweepRow], results_dir: Path) -> tuple[Path, Path]:
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / "scale_sweep.json"
    csv_path = results_dir / "scale_sweep.csv"
    payload = {
        "question": scientific_question(),
        "n_rows": len(rows),
        "rows": [r.to_json() for r in rows],
    }
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    fieldnames = list(asdict(rows[0]).keys()) if rows else [
        "id", "family", "n", "seed", "method", "certified", "energy",
        "known_energy", "energy_error", "wall_s", "chi", "chi_kept", "notes",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            rec = row.to_json()
            writer.writerow(rec)
    return csv_path, json_path


def plot_scale(rows: list[SweepRow], fig_dir: Path) -> Path:
    """n vs time; n vs energy error (where known); method vs n; χ vs n (MPS)."""
    _style()
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.6))
    family_color = {
        "maxcut-er": CYAN,
        "pmj-glass": MAGENTA,
        "bipartite-maxcut": AMBER,
        "ferro-chain": LIME,
        "crystal": TEXT,
        "tfim-chain": "#3d5a80",
    }
    family_index = {fam: i for i, fam in enumerate(family_color)}
    n_fam = len(family_color)
    dodge = 0.28
    method_marker = {"exact": "o", "cpu-pt": "s", "gpu-pt": "D", "mps": "^", "skipped": "x"}
    solved = [r for r in rows if r.method != "skipped"]

    def _x(row: SweepRow) -> float:
        return row.n + (family_index[row.family] - (n_fam - 1) / 2) * dodge

    ax = axes[0, 0]
    for fam, color in family_color.items():
        pts = [r for r in solved if r.family == fam]
        if not pts:
            continue
        ax.plot([r.n for r in pts], [r.wall_s for r in pts], "-", color=color, lw=1.0, alpha=0.45)
        for r in pts:
            ax.scatter(
                r.n, r.wall_s, color=color, marker=method_marker.get(r.method, "o"),
                s=36, zorder=3, edgecolors="white", linewidths=0.4,
            )
    ax.axvline(18, color=GRID, ls=":", lw=1.0)
    ax.set_xlabel("system size  n")
    ax.set_ylabel("wall-clock  (s)")
    ax.set_yscale("log")
    ax.set_title("(a)  n vs wall time")

    ax = axes[0, 1]
    for fam, color in family_color.items():
        pts = [r for r in solved if r.family == fam and r.energy_error is not None]
        if not pts:
            continue
        for r in pts:
            err = max(float(r.energy_error), 0.0)
            ax.scatter(
                _x(r), err, color=color, marker=method_marker.get(r.method, "o"),
                s=36, zorder=3, edgecolors="white", linewidths=0.4,
            )
    ax.set_xlabel("system size  n")
    ax.set_ylabel("E − E_known   (≥0)")
    ax.set_title("(b)  n vs energy error (where known)")

    ax = axes[1, 0]
    method_y = {"exact": 3, "gpu-pt": 2, "cpu-pt": 1, "mps": 0}
    for fam, color in family_color.items():
        pts = [r for r in solved if r.family == fam and r.method in method_y]
        for r in pts:
            ax.scatter(
                _x(r), method_y[r.method], color=color,
                marker=method_marker.get(r.method, "o"), s=42, zorder=3,
                edgecolors="white", linewidths=0.4,
            )
    ax.axvline(18, color=GRID, ls=":", lw=1.0)
    ax.set_yticks(list(method_y.values()), list(method_y.keys()))
    ax.set_xlabel("system size  n")
    ax.set_ylabel("method")
    ax.set_title("(c)  method used vs n")

    ax = axes[1, 1]
    mps_rows = [r for r in solved if r.family == "tfim-chain" and r.chi is not None]
    if mps_rows:
        ax.plot(
            [r.n for r in mps_rows], [r.chi for r in mps_rows],
            "o-", color=CYAN, lw=1.8, markersize=7, label="χ_eff (TFIM Γ=1)",
        )
        ax.plot(
            [r.n for r in mps_rows], [r.chi_kept for r in mps_rows],
            "s--", color=AMBER, lw=1.2, markersize=6, label="χ_kept (budget)",
        )
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "no MPS rows (n > cutoff or --no-mps)",
                ha="center", va="center", transform=ax.transAxes, color=TEXT)
    ax.set_xlabel("system size  n")
    ax.set_ylabel("bond dimension  χ")
    ax.set_title("(d)  χ vs n  (MPS / TFIM only)")

    handles = [
        Line2D([0], [0], color=c, marker="o", lw=0, label=f)
        for f, c in family_color.items()
        if any(r.family == f for r in solved)
    ]
    axes[0, 0].legend(handles=handles, fontsize=7, loc="upper left", framealpha=0.9)

    fig.suptitle("DRiFT scale path — drift.solve vs n  (exact → GPU-PT → CPU-PT)", fontsize=13)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / "scale_sweep.png"
    return _save(fig, out)


def _print_table(rows: list[SweepRow]) -> None:
    print(f"    {'id':<32} {'method':>7} {'cert':>4} {'E':>10} {'dE':>8} {'t(s)':>8} {'χ':>4}")
    for r in rows:
        de = f"{r.energy_error:8.3g}" if r.energy_error is not None else f"{'—':>8}"
        chi = f"{r.chi:4d}" if r.chi is not None else f"{'—':>4}"
        e = f"{r.energy:10.3f}" if r.energy is not None else f"{'skip':>10}"
        print(f"    {r.id:<32} {r.method:>7} {str(r.certified):>4} {e} {de} {r.wall_s:8.4f} {chi}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ci", action="store_true", help="tiny n≤10 sweep; skip MPS; skip figure")
    p.add_argument("--n-min", type=int, default=1)
    p.add_argument("--n-max", type=int, default=32)
    p.add_argument("--families", nargs="*", default=None)
    p.add_argument("--exact-max", type=int, default=18)
    p.add_argument("--n-rounds", type=int, default=400)
    p.add_argument("--n-replicas", type=int, default=None)
    p.add_argument("--sweeps-per-round", type=int, default=4)
    p.add_argument("--no-gpu", action="store_true", help="force CPU-PT even if a GPU binary exists")
    p.add_argument("--no-mps", action="store_true")
    p.add_argument("--mps-n-max", type=int, default=MPS_N_MAX_SWEEP)
    p.add_argument("--no-figure", action="store_true")
    p.add_argument("--results-dir", type=Path, default=REPO_ROOT / "docs" / "results")
    p.add_argument("--fig-dir", type=Path, default=REPO_ROOT / "figures" / "scale")
    args = p.parse_args(argv)

    n_max = 10 if args.ci else args.n_max
    skip_mps = True if args.ci else args.no_mps
    n_rounds = 80 if args.ci else args.n_rounds
    no_figure = True if args.ci else args.no_figure
    families = tuple(args.families) if args.families else None

    print(scientific_question())
    print(f"  n≤{n_max}  exact_max={args.exact_max}  gpu={not args.no_gpu}  mps={not skip_mps}")
    rows = run_sweep(
        n_min=args.n_min, n_max=n_max, families=families,
        exact_max=args.exact_max, use_gpu=not args.no_gpu,
        n_replicas=args.n_replicas, n_rounds=n_rounds,
        sweeps_per_round=args.sweeps_per_round,
        mps_n_max=args.mps_n_max, skip_mps=skip_mps,
    )
    _print_table(rows)
    csv_path, json_path = write_results(rows, args.results_dir)
    print(f"  wrote {csv_path}")
    print(f"  wrote {json_path}")
    if not no_figure:
        fig_path = plot_scale(rows, args.fig_dir)
        print(f"  wrote {fig_path}")
    n_certified = sum(1 for r in rows if r.certified)
    n_heuristic = sum(1 for r in rows if r.method in ("cpu-pt", "gpu-pt", "mps"))
    print(f"  certified={n_certified}  heuristic(uncertified)={n_heuristic}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
