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

    python -m experiments.scale_sweep --ci                # n≤10, no MPS — CI-safe
    python -m experiments.scale_sweep --profile local     # CPU ladder n≤40 + report
    python -m experiments.scale_sweep --profile gpu       # same IDs; gpu-pt iff binary exists

GPU is optional: if ``cuda/ising_pt`` is missing, large n continue on CPU-PT.
The runner **never invents gpu-pt rows**. ``--ci`` does not overwrite published
``docs/results/scale_sweep.*``.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, NoReturn

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from drift import gpu, mps, solve
from drift.benchmarks import (
    MPS_FAMILIES,
    MPS_N_MAX_SWEEP,
    PRIMARY_SEEDS,
    BuiltInstance,
    iter_instances,
    load_manifest,
    scientific_question,
)
from drift.viz import AMBER, CYAN, GRID, LIME, MAGENTA, TEXT, _save, _style

REPO_ROOT = Path(__file__).resolve().parent.parent
ProfileName = Literal["ci", "local", "gpu"]

# Published artifacts. ``--ci`` must not clobber these unless the caller
# passes an explicit ``--results-dir``.
DEFAULT_RESULTS_DIR = REPO_ROOT / "docs" / "results"
DEFAULT_FIG_DIR = REPO_ROOT / "figures" / "scale"
DEFAULT_N_MAX = 40
CI_N_MAX = 10


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


@dataclass
class SweepMeta:
    """Provenance for a sweep — what ran, on what, with which flags."""

    profile: str
    n_min: int
    n_max: int
    exact_max: int
    n_rounds: int
    use_gpu_flag: bool
    gpu_available: bool
    skip_mps: bool
    mps_n_max: int
    command: str
    python: str = field(default_factory=lambda: sys.version.split()[0])
    platform: str = field(default_factory=lambda: platform.platform())
    timestamp_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    )

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    def hardware_label(self, rows: list[SweepRow]) -> str:
        """One-line label derived from *rows*, never from a hoped-for GPU."""
        methods = {r.method for r in rows}
        if "gpu-pt" in methods:
            return (
                f"GPU-PT recorded ({sum(r.method == 'gpu-pt' for r in rows)} rows); "
                f"gpu_available={self.gpu_available}"
            )
        if "cpu-pt" in methods:
            why = "use_gpu=False" if not self.use_gpu_flag else "GPU binary absent"
            return f"CPU ({why}); gpu_available={self.gpu_available}; no gpu-pt rows"
        return f"exact/MPS only; gpu_available={self.gpu_available}; no gpu-pt rows"


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


def _is_primary(row: SweepRow) -> bool:
    if row.family in PRIMARY_SEEDS:
        return row.seed == PRIMARY_SEEDS[row.family]
    return row.seed == 0


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
    if sol.method == "gpu-pt" and not gpu.gpu_available():
        raise RuntimeError("honesty contract violated: gpu-pt recorded without a GPU binary")
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
    n_max: int | None = DEFAULT_N_MAX,
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


def write_results(
    rows: list[SweepRow],
    results_dir: Path,
    *,
    meta: SweepMeta | dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / "scale_sweep.json"
    csv_path = results_dir / "scale_sweep.csv"
    hardware = meta.to_json() if isinstance(meta, SweepMeta) else (meta or {})
    payload = {
        "question": scientific_question(),
        "n_rows": len(rows),
        "hardware": hardware,
        "methods_present": sorted({r.method for r in rows}),
        "gpu_pt_rows": sum(1 for r in rows if r.method == "gpu-pt"),
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


def _fmt_time(seconds: float) -> str:
    if seconds < 1e-3:
        return f"{seconds * 1e6:.2f} µs"
    if seconds < 1.0:
        return f"{seconds * 1e3:.2f} ms"
    return f"{seconds:.3f} s"


def _fmt_err(err: float | None) -> str:
    if err is None:
        return "—"
    if abs(err) < 1e-12:
        return "0"
    return f"{err:.3g}"


def write_markdown_report(
    rows: list[SweepRow],
    path: Path,
    *,
    meta: SweepMeta,
    fig_rel: str = "../../figures/scale/scale_sweep.png",
) -> Path:
    """Regenerate ``SCALE-sweep.md`` from measured rows. No gpu-pt claims without gpu-pt rows."""
    methods = sorted({r.method for r in rows})
    n_gpu = sum(1 for r in rows if r.method == "gpu-pt")
    n_cpu = sum(1 for r in rows if r.method == "cpu-pt")
    n_exact = sum(1 for r in rows if r.method == "exact")
    n_mps = sum(1 for r in rows if r.method == "mps")
    n_skip = sum(1 for r in rows if r.method == "skipped")
    n_cert = sum(1 for r in rows if r.certified)
    primary = [r for r in rows if _is_primary(r)]
    extras = [r for r in rows if not _is_primary(r)]
    families = ("maxcut-er", "pmj-glass", "bipartite-maxcut", "ferro-chain")
    fam_short = {
        "maxcut-er": "maxcut-er",
        "pmj-glass": "pmj-glass",
        "bipartite-maxcut": "bipartite",
        "ferro-chain": "ferro-chain",
    }

    def _primary_at(family: str, n: int) -> SweepRow | None:
        hits = [r for r in primary if r.family == family and r.n == n]
        return hits[0] if hits else None

    hw = meta.hardware_label(rows)
    gpu_banner = (
        f"**gpu-pt rows in this file: {n_gpu}.**"
        if n_gpu
        else (
            "**gpu-pt rows in this file: none.** This run did not execute the CUDA engine. "
            "CPU-PT times are not GPU throughput. Phase 14 Gflips/s live in "
            "[`PHASE14-results.md`](PHASE14-results.md) and are a different experiment."
        )
    )

    exact_ns = sorted({r.n for r in primary if r.method == "exact" and r.family in PRIMARY_SEEDS})
    pt_ns = sorted({r.n for r in primary if r.method in ("cpu-pt", "gpu-pt") and r.family in PRIMARY_SEEDS})

    lines: list[str] = [
        "# Scale path — wall time, quality, and χ vs n via `drift.solve`",
        "",
        "**Question (falsifiable):** How do wall-clock time and solution quality (and χ where an",
        "MPS/tensor path applies) scale with system size n for fixed instance families, when",
        "solving via `drift.solve` (exact → GPU-PT → CPU-PT)?",
        "",
        f"**Status:** {hw}.",
        "",
        gpu_banner,
        "",
        f"![scale sweep]({fig_rel})",
        "",
        "## What was measured",
        "",
        "A **versioned instance bank** (`drift/benchmarks/instances/manifest.json`, bank `v1`)",
        "with stable IDs such as `v1.maxcut-er.n008.s001`. Generators are seeded; n≤8 classical",
        "instances are also checked in as JSON fixtures so a generator change is loud. Families:",
        "",
        "| Family | What it is | Known energy |",
        "|--------|------------|--------------|",
        "| `maxcut-er` | Erdős–Rényi MaxCut G(n, 1/2) | exact enum, n≤20 |",
        "| `pmj-glass` | complete ±J spin glass, h=0 | exact enum, n≤20 |",
        "| `bipartite-maxcut` | random bipartite MaxCut | analytic: E = −n_edges |",
        "| `ferro-chain` | open 1-D ferromagnet | analytic: E = −(n−1) |",
        "| `crystal` | period-4 stripe crystal (4×4 / 6×4 / 4×8 / 10×4) | analytic: E = −2n |",
        "| `tfim-chain` | open TFIM at Γ/J = 1 | Lanczos, n≤14; χ via MPS |",
        "",
        "Classical families go through **`drift.solve`**. χ is recorded **only** on `tfim-chain`",
        "through the existing MPS/TEBD solver (`drift.mps`). That path is 1-D nearest-neighbour",
        "TFIM: it does **not** apply to dense MaxCut, spin glasses, or 2-D crystals.",
        "",
        f"**MPS cutoff (documented):** default sweep χ ladder is **n≤{meta.mps_n_max}**",
        "(`MPS_N_MAX_SWEEP`). n=20 and n=24 sit in the bank for local runs (`--mps-n-max 24`).",
        "Phase 13 has shown n=48 with a larger budget; that is out of scope for this CPU curve.",
        "",
        "A heuristic minimum is **never** marked `certified`. GPU is optional: if the CUDA",
        "binary is missing, n > `exact_max` (18) continues on CPU-PT.",
        "",
        f"This file has **{len(rows)} rows** ({n_exact} exact, {n_cpu} cpu-pt, {n_gpu} gpu-pt, "
        f"{n_mps} mps, {n_skip} skipped); **{n_cert} certified**. "
        f"{len(primary)} primary-seed rows, {len(extras)} extra-seed rows. "
        f"Methods present: `{', '.join(methods) or '—'}`.",
        "",
        "## How to regenerate",
        "",
        "```bash",
        "# After catalog_rows() changes (fingerprints + known energies + n≤8 fixtures):",
        "python -m drift.benchmarks",
        "",
        "# CI smoke (does not overwrite docs/results/scale_sweep.*):",
        "python -m experiments.scale_sweep --ci",
        "",
        "# Published CPU ladder (this document's default):",
        "python -m experiments.scale_sweep --profile local --write-report",
        "",
        "# Same IDs on a machine with cuda/ising_pt (Windows/sm_120). Records gpu-pt only",
        "# if the binary actually ran — never by renaming cpu-pt rows:",
        "python -m experiments.scale_sweep --profile gpu --write-report",
        "```",
        "",
        "Results: `docs/results/scale_sweep.csv`, `docs/results/scale_sweep.json`,",
        "`figures/scale/scale_sweep.png`. JSON `hardware` and `gpu_pt_rows` pin provenance.",
        "",
        f"This file was produced at `{meta.timestamp_utc}` with `{meta.command}`",
        f"(profile={meta.profile}, n≤{meta.n_max}, exact_max={meta.exact_max},",
        f"n_rounds={meta.n_rounds}, python {meta.python}, {meta.platform}).",
        "",
        "## Curves",
        "",
        f"Hardware: {hw}. `exact_max={meta.exact_max}`. CPU-PT uses `drift.solve` defaults",
        "(32 replicas, 400 rounds unless `--n-rounds` / `--ci` override, 4 sweeps/round).",
        "MPS uses χ_max=16, 12 sweeps, a short imaginary-time schedule (a scale probe, not",
        "a high-precision Phase-13 rerun).",
        "",
        "### (a) Wall time vs n",
        "",
        "Primary seed per family (lines in the figure). Extra seeds are plotted as faint",
        "markers at the same n; they are not in this table.",
        "",
    ]

    if exact_ns:
        lines.append("Exact enumeration grows exponentially, as it must (2ⁿ configurations):")
        lines.append("")
        header = "| n | " + " | ".join(fam_short[f] for f in families) + " |"
        sep = "|--:|" + "|".join("----------:" for _ in families) + "|"
        lines += [header, sep]
        for n in exact_ns:
            cells = [str(n)]
            for fam in families:
                row = _primary_at(fam, n)
                cells.append(_fmt_time(row.wall_s) if row else "—")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    if pt_ns:
        pt_name = "gpu-pt" if n_gpu else "cpu-pt"
        lines.append(
            f"At n>{meta.exact_max} the dispatcher switches to **{pt_name}** "
            f"({'GPU binary present' if n_gpu else 'no GPU binary in this environment'})."
        )
        lines.append("")
        header = "| n | method | " + " | ".join(fam_short[f] for f in families) + " |"
        sep = "|--:|--------|" + "|".join("----------:" for _ in families) + "|"
        lines += [header, sep]
        for n in pt_ns:
            cells = [str(n)]
            sample = next(
                (r for r in primary if r.n == n and r.family in PRIMARY_SEEDS
                 and r.method in ("cpu-pt", "gpu-pt")),
                None,
            )
            cells.append(sample.method if sample else "—")
            for fam in families:
                row = _primary_at(fam, n)
                cells.append(_fmt_time(row.wall_s) if row else "—")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    crystal_rows = [r for r in rows if r.family == "crystal"]
    if crystal_rows:
        lines.append("Crystal size steps (analytic E = −2n):")
        lines.append("")
        lines.append("| id | n | method | certified | E | dE | wall |")
        lines.append("|----|--:|--------|-----------:|--:|----|------|")
        for r in crystal_rows:
            e = f"{r.energy:.1f}" if r.energy is not None else "—"
            lines.append(
                f"| `{r.id}` | {r.n} | {r.method} | {r.certified} | {e} | "
                f"{_fmt_err(r.energy_error)} | {_fmt_time(r.wall_s)} |"
            )
        lines.append("")

    lines += [
        "### (b) Energy error vs known optimum",
        "",
    ]
    known = [r for r in rows if r.energy_error is not None and r.method != "skipped"]
    zero = [
        r for r in known
        if r.method != "mps" and abs(float(r.energy_error)) < 1e-9
    ]
    nonzero_class = [
        r for r in known
        if r.method in ("exact", "cpu-pt", "gpu-pt") and abs(float(r.energy_error)) >= 1e-9
    ]
    no_oracle = [
        r for r in rows
        if r.method in ("cpu-pt", "gpu-pt") and r.known_energy is None
    ]
    lines.append(
        f"Rows with an oracle: **{len(known)}**. Classical rows with dE = 0: "
        f"**{len(zero)}**. Classical rows with dE ≠ 0: **{len(nonzero_class)}**."
    )
    lines.append("")
    if nonzero_class:
        lines.append("Non-zero classical errors (heuristic miss, not hidden):")
        lines.append("")
        lines.append("| id | method | dE |")
        lines.append("|----|--------|----|")
        for r in nonzero_class:
            lines.append(f"| `{r.id}` | {r.method} | {_fmt_err(r.energy_error)} |")
        lines.append("")
    else:
        lines.append(
            "Wherever an oracle exists on this ladder, **error is 0** for exact and PT rows "
            "(certified only when method=`exact`)."
        )
        lines.append("")
    if no_oracle:
        lines.append(
            f"ER MaxCut and ±J glass have **no oracle** on {len(no_oracle)} rows "
            "(n past enumeration). Those rows report energy and time only — strong minima, "
            "not certified."
        )
        lines.append("")
    mps_known = [r for r in rows if r.method == "mps" and r.energy_error is not None]
    if mps_known:
        lines.append("MPS (not a `drift.solve` path) is variational:")
        lines.append("")
        lines.append("| n | χ_eff | χ_kept | dE vs Lanczos | wall |")
        lines.append("|--:|------:|-------:|---------------|------|")
        for r in mps_known:
            lines.append(
                f"| {r.n} | {r.chi} | {r.chi_kept} | {_fmt_err(r.energy_error)} | "
                f"{_fmt_time(r.wall_s)} |"
            )
        lines.append("")
        lines.append("Never certified.")
        lines.append("")

    if extras:
        extra_known = [r for r in extras if r.energy_error is not None]
        extra_zero = [
            r for r in extra_known if abs(float(r.energy_error)) < 1e-9
        ]
        lines.append(
            f"### Extra seeds ({len(extras)} rows)"
        )
        lines.append("")
        lines.append(
            "Stochastic families (`maxcut-er`, `pmj-glass`, `bipartite-maxcut`) carry extra "
            "seeds at cheap n (exact-reachable, plus one seed at the n=20 dispatcher wall). "
            "`ferro-chain` is seed-invariant and is **not** duplicated."
        )
        lines.append("")
        lines.append(
            f"Extra-seed rows with an oracle: {len(extra_known)}; dE = 0 on "
            f"{len(extra_zero)} of those."
        )
        lines.append("")

    lines += [
        "### (c) Method vs n",
        "",
        "The dispatcher does what it says:",
        "",
        f"- n ≤ {meta.exact_max} → `exact`, `certified=True` ({n_exact} rows)",
        f"- n > {meta.exact_max}, GPU binary present → `gpu-pt`, `certified=False` ({n_gpu} rows)",
        f"- n > {meta.exact_max}, no GPU / `--no-gpu` → `cpu-pt`, `certified=False` ({n_cpu} rows)",
        f"- `tfim-chain` → `mps`, `certified=False` ({n_mps} rows); skipped above the χ cutoff "
        f"({n_skip} rows)",
        "",
        "No row has `certified=True` with a heuristic method. An empty `gpu-pt` band is a",
        "missing binary (or `--profile local`), not a silent fallback pretending to be GPU.",
        "",
        "### (d) χ vs n (TFIM / MPS only)",
        "",
    ]
    mps_rows = [r for r in rows if r.method == "mps"]
    skipped_tfim = [r for r in rows if r.family == "tfim-chain" and r.method == "skipped"]
    if mps_rows:
        lines.append(f"Open TFIM at Γ/J = 1, χ_max=16, n≤{meta.mps_n_max}:")
        lines.append("")
        lines.append("| n | χ_eff | χ_kept | wall |")
        lines.append("|--:|------:|-------:|------|")
        for r in mps_rows:
            lines.append(
                f"| {r.n} | {r.chi} | {r.chi_kept} | {_fmt_time(r.wall_s)} |"
            )
        lines.append("")
        lines.append(
            "Effective χ stays small on this short schedule (a scale-path probe). "
            "Treat χ here as that probe, not a replacement for "
            "`docs/results/PHASE13-results.md`."
        )
        lines.append("")
    if skipped_tfim:
        ids = ", ".join(f"`{r.id}`" for r in skipped_tfim)
        lines.append(
            f"Skipped at the documented cutoff: {ids}. Raise `--mps-n-max` locally; "
            "Phase 13 already measured n=48."
        )
        lines.append("")

    lines += [
        "## Answer to the question",
        "",
        f"On these fixed families, **`drift.solve` has two time regimes**. Below n={meta.exact_max}",
        "the certified exact engine's wall time tracks 2ⁿ. Above that, "
        + ("GPU-PT " if n_gpu else "CPU-PT (no GPU) ")
        + "grows with the PT loop (replicas × rounds × sweeps × work per sweep), "
        "not with 2ⁿ. **Solution quality matches every available oracle** on this run "
        + ("except the rows listed in (b)" if nonzero_class else "for exact and PT rows")
        + "; PT rows stay `certified=False`. χ is a tensor-network number: it is defined on",
        "the TFIM chain and is not claimed for MaxCut or spin glasses. GPU-PT is the same",
        "experiment with a binary; it is not inferred from the CPU curve.",
        "",
        "## Honesty",
        "",
        "- Heuristic (cpu-pt / gpu-pt / mps) is never `certified`.",
        "- ER MaxCut and ±J glass at n>20 have no known energy here.",
        "- A CPU PT curve is not a GPU throughput claim (see Phase 14 for Gflips/s).",
        "- This file records only methods that ran. Zero `gpu-pt` rows means the CUDA",
        "  engine did not run, not that GPU time equals CPU time.",
        "- The MPS ladder uses a reduced schedule so the sweep stays cheap; treat χ as a",
        "  scale-path probe, not a replacement for `docs/results/PHASE13-results.md`.",
        "- Extra ferro-chain seeds are omitted on purpose: that generator ignores `seed`.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


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
    seed_ranks: dict[str, dict[int, int]] = {}
    for fam in family_color:
        seeds = sorted({r.seed for r in solved if r.family == fam})
        seed_ranks[fam] = {s: i for i, s in enumerate(seeds)}

    def _x(row: SweepRow) -> float:
        fam_off = (family_index[row.family] - (n_fam - 1) / 2) * dodge
        ranks = seed_ranks.get(row.family, {row.seed: 0})
        n_seeds = max(len(ranks), 1)
        seed_off = (ranks.get(row.seed, 0) - (n_seeds - 1) / 2) * 0.10
        return row.n + fam_off + seed_off

    ax = axes[0, 0]
    for fam, color in family_color.items():
        pts = [r for r in solved if r.family == fam]
        if not pts:
            continue
        prim = sorted((r for r in pts if _is_primary(r)), key=lambda r: r.n)
        if prim:
            ax.plot(
                [r.n for r in prim], [r.wall_s for r in prim],
                "-", color=color, lw=1.0, alpha=0.45,
            )
        for r in pts:
            ax.scatter(
                r.n, r.wall_s, color=color, marker=method_marker.get(r.method, "o"),
                s=36 if _is_primary(r) else 22, zorder=3, edgecolors="white",
                linewidths=0.4, alpha=1.0 if _is_primary(r) else 0.55,
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
                s=36 if _is_primary(r) else 22, zorder=3, edgecolors="white",
                linewidths=0.4, alpha=1.0 if _is_primary(r) else 0.55,
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
                marker=method_marker.get(r.method, "o"),
                s=42 if _is_primary(r) else 24, zorder=3,
                edgecolors="white", linewidths=0.4,
                alpha=1.0 if _is_primary(r) else 0.55,
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

    gpu_note = "gpu-pt" if any(r.method == "gpu-pt" for r in rows) else "CPU-PT (no gpu-pt rows)"
    fig.suptitle(
        f"DRiFT scale path — drift.solve vs n  (exact → {gpu_note})",
        fontsize=13,
    )
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


def _unknown_profile(profile: str) -> NoReturn:
    raise ValueError(f"unknown profile {profile!r}; expected one of ('ci', 'local', 'gpu')")


def _resolve_profile(profile: ProfileName) -> tuple[int, bool, bool, int, bool]:
    """Return (n_max, skip_mps, use_gpu, n_rounds, no_figure) for a named profile."""
    if profile == "ci":
        return CI_N_MAX, True, False, 80, True
    if profile == "local":
        return DEFAULT_N_MAX, False, False, 400, False
    if profile == "gpu":
        return DEFAULT_N_MAX, False, True, 400, False
    _unknown_profile(profile)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ci", action="store_true", help="alias for --profile ci: n≤10, no MPS, no published artifacts")
    p.add_argument(
        "--profile", choices=("ci", "local", "gpu"), default=None,
        help="ci = smoke; local = CPU n≤40 + report; gpu = same IDs, gpu-pt only if binary exists",
    )
    p.add_argument("--n-min", type=int, default=1)
    p.add_argument("--n-max", type=int, default=None, help=f"default {DEFAULT_N_MAX} (or profile default)")
    p.add_argument("--families", nargs="*", default=None)
    p.add_argument("--exact-max", type=int, default=18)
    p.add_argument("--n-rounds", type=int, default=None)
    p.add_argument("--n-replicas", type=int, default=None)
    p.add_argument("--sweeps-per-round", type=int, default=4)
    p.add_argument("--no-gpu", action="store_true", help="force CPU-PT even if a GPU binary exists")
    p.add_argument("--no-mps", action="store_true")
    p.add_argument("--mps-n-max", type=int, default=MPS_N_MAX_SWEEP)
    p.add_argument("--no-figure", action="store_true")
    p.add_argument("--write-report", action="store_true", help="regenerate docs/results/SCALE-sweep.md from this run")
    p.add_argument("--results-dir", type=Path, default=None)
    p.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    args = p.parse_args(argv)

    if args.ci and args.profile and args.profile != "ci":
        p.error("--ci and --profile disagree")
    explicit_profile = args.ci or args.profile is not None
    profile: ProfileName
    if args.ci:
        profile = "ci"
    elif args.profile is not None:
        profile = args.profile
    else:
        profile = "local" if args.no_gpu else "gpu"

    n_max_p, skip_mps_p, use_gpu_p, n_rounds_p, no_figure_p = _resolve_profile(profile)
    n_max = args.n_max if args.n_max is not None else n_max_p
    skip_mps = True if (profile == "ci" or args.no_mps) else skip_mps_p
    n_rounds = args.n_rounds if args.n_rounds is not None else n_rounds_p
    no_figure = True if (profile == "ci" or args.no_figure) else no_figure_p
    use_gpu = False if (args.no_gpu or profile == "local" or profile == "ci") else use_gpu_p
    families = tuple(args.families) if args.families else None

    if profile == "gpu" and use_gpu and not gpu.gpu_available():
        print(
            "profile=gpu requested but cuda/ising_pt is missing; "
            "recording cpu-pt. Not faking gpu-pt rows.",
            file=sys.stderr,
        )

    write_artifacts = True
    results_dir = args.results_dir
    if profile == "ci":
        # Smoke test: do not clobber the published ladder unless the caller
        # explicitly chose a results directory.
        if results_dir is None:
            write_artifacts = False
            results_dir = DEFAULT_RESULTS_DIR
    elif results_dir is None:
        results_dir = DEFAULT_RESULTS_DIR

    command = "python -m experiments.scale_sweep " + " ".join(argv)
    meta = SweepMeta(
        profile=profile,
        n_min=args.n_min,
        n_max=n_max,
        exact_max=args.exact_max,
        n_rounds=n_rounds,
        use_gpu_flag=use_gpu,
        gpu_available=gpu.gpu_available(),
        skip_mps=skip_mps,
        mps_n_max=args.mps_n_max,
        command=command.strip(),
    )

    print(scientific_question())
    print(
        f"  profile={profile}  n≤{n_max}  exact_max={args.exact_max}  "
        f"gpu={use_gpu} (available={meta.gpu_available})  mps={not skip_mps}"
    )
    rows = run_sweep(
        n_min=args.n_min, n_max=n_max, families=families,
        exact_max=args.exact_max, use_gpu=use_gpu,
        n_replicas=args.n_replicas, n_rounds=n_rounds,
        sweeps_per_round=args.sweeps_per_round,
        mps_n_max=args.mps_n_max, skip_mps=skip_mps,
    )
    if any(r.method == "gpu-pt" for r in rows) and not meta.gpu_available:
        raise RuntimeError("honesty contract violated: gpu-pt rows without a GPU binary")
    _print_table(rows)
    if write_artifacts:
        csv_path, json_path = write_results(rows, results_dir, meta=meta)
        print(f"  wrote {csv_path}")
        print(f"  wrote {json_path}")
        write_md = args.write_report or (explicit_profile and profile in ("local", "gpu"))
        if write_md:
            md_path = write_markdown_report(rows, results_dir / "SCALE-sweep.md", meta=meta)
            print(f"  wrote {md_path}")
    else:
        print("  --ci: not writing docs/results/scale_sweep.* (pass --results-dir to override)")
    if not no_figure:
        fig_path = plot_scale(rows, args.fig_dir)
        print(f"  wrote {fig_path}")
    n_certified = sum(1 for r in rows if r.certified)
    n_heuristic = sum(1 for r in rows if r.method in ("cpu-pt", "gpu-pt", "mps"))
    n_gpu = sum(1 for r in rows if r.method == "gpu-pt")
    print(
        f"  certified={n_certified}  heuristic(uncertified)={n_heuristic}  "
        f"gpu-pt={n_gpu}  hardware={meta.hardware_label(rows)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
