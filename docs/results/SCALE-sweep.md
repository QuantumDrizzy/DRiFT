# Scale path — wall time, quality, and χ vs n via `drift.solve`

**Question (falsifiable):** How do wall-clock time and solution quality (and χ where an
MPS/tensor path applies) scale with system size n for fixed instance families, when
solving via `drift.solve` (exact → GPU-PT → CPU-PT)?

**Status:** CPU (use_gpu=False); gpu_available=False; no gpu-pt rows.

**gpu-pt rows in this file: none.** This run did not execute the CUDA engine. CPU-PT times are not GPU throughput. Phase 14 Gflips/s live in [`PHASE14-results.md`](PHASE14-results.md) and are a different experiment.

![scale sweep](../../figures/scale/scale_sweep.png)

## What was measured

A **versioned instance bank** (`drift/benchmarks/instances/manifest.json`, bank `v1`)
with stable IDs such as `v1.maxcut-er.n008.s001`. Generators are seeded; n≤8 classical
instances are also checked in as JSON fixtures so a generator change is loud. Families:

| Family | What it is | Known energy |
|--------|------------|--------------|
| `maxcut-er` | Erdős–Rényi MaxCut G(n, 1/2) | exact enum, n≤20 |
| `pmj-glass` | complete ±J spin glass, h=0 | exact enum, n≤20 |
| `bipartite-maxcut` | random bipartite MaxCut | analytic: E = −n_edges |
| `ferro-chain` | open 1-D ferromagnet | analytic: E = −(n−1) |
| `crystal` | period-4 stripe crystal (4×4 / 6×4 / 4×8 / 10×4) | analytic: E = −2n |
| `tfim-chain` | open TFIM at Γ/J = 1 | Lanczos, n≤14; χ via MPS |

Classical families go through **`drift.solve`**. χ is recorded **only** on `tfim-chain`
through the existing MPS/TEBD solver (`drift.mps`). That path is 1-D nearest-neighbour
TFIM: it does **not** apply to dense MaxCut, spin glasses, or 2-D crystals.

**MPS cutoff (documented):** default sweep χ ladder is **n≤16**
(`MPS_N_MAX_SWEEP`). n=20 and n=24 sit in the bank for local runs (`--mps-n-max 24`).
Phase 13 has shown n=48 with a larger budget; that is out of scope for this CPU curve.

A heuristic minimum is **never** marked `certified`. GPU is optional: if the CUDA
binary is missing, n > `exact_max` (18) continues on CPU-PT.

This file has **102 rows** (61 exact, 34 cpu-pt, 0 gpu-pt, 5 mps, 2 skipped); **61 certified**. 63 primary-seed rows, 39 extra-seed rows. Methods present: `cpu-pt, exact, mps, skipped`.

## How to regenerate

```bash
# After catalog_rows() changes (fingerprints + known energies + n≤8 fixtures):
python -m drift.benchmarks

# CI smoke (does not overwrite docs/results/scale_sweep.*):
python -m experiments.scale_sweep --ci

# Published CPU ladder (this document's default):
python -m experiments.scale_sweep --profile local --write-report

# Same IDs on a machine with cuda/ising_pt (Windows/sm_120). Records gpu-pt only
# if the binary actually ran — never by renaming cpu-pt rows:
python -m experiments.scale_sweep --profile gpu --write-report
```

Results: `docs/results/scale_sweep.csv`, `docs/results/scale_sweep.json`,
`figures/scale/scale_sweep.png`. JSON `hardware` and `gpu_pt_rows` pin provenance.

This file was produced at `2026-09-21T00:15:46Z` with `python -m experiments.scale_sweep --profile local --write-report`
(profile=local, n≤40, exact_max=18,
n_rounds=400, python 3.12.3, Linux-6.12.94+-x86_64-with-glibc2.39).

## Curves

Hardware: CPU (use_gpu=False); gpu_available=False; no gpu-pt rows. `exact_max=18`. CPU-PT uses `drift.solve` defaults
(32 replicas, 400 rounds unless `--n-rounds` / `--ci` override, 4 sweeps/round).
MPS uses χ_max=16, 12 sweeps, a short imaginary-time schedule (a scale probe, not
a high-precision Phase-13 rerun).

### (a) Wall time vs n

Primary seed per family (lines in the figure). Extra seeds are plotted as faint
markers at the same n; they are not in this table.

Exact enumeration grows exponentially, as it must (2ⁿ configurations):

| n | maxcut-er | pmj-glass | bipartite | ferro-chain |
|--:|----------:|----------:|----------:|----------:|
| 8 | 251.72 µs | 109.06 µs | 92.00 µs | 76.86 µs |
| 10 | 284.30 µs | 109.70 µs | 99.59 µs | 92.70 µs |
| 12 | 722.34 µs | 533.88 µs | 525.78 µs | 476.99 µs |
| 14 | 3.48 ms | 1.94 ms | 1.89 ms | 1.82 ms |
| 16 | 9.41 ms | 5.96 ms | 5.80 ms | 5.59 ms |
| 18 | 41.79 ms | 23.51 ms | 24.30 ms | 22.44 ms |

At n>18 the dispatcher switches to **cpu-pt** (no GPU binary in this environment).

| n | method | maxcut-er | pmj-glass | bipartite | ferro-chain |
|--:|--------|----------:|----------:|----------:|----------:|
| 20 | cpu-pt | 2.588 s | 2.486 s | 2.465 s | 2.417 s |
| 22 | cpu-pt | 2.738 s | 2.720 s | 2.712 s | 2.652 s |
| 24 | cpu-pt | 2.946 s | 2.962 s | 2.953 s | 2.877 s |
| 28 | cpu-pt | 3.431 s | 3.453 s | 3.449 s | 3.363 s |
| 32 | cpu-pt | 3.914 s | 3.953 s | 3.972 s | 3.877 s |
| 36 | cpu-pt | 4.506 s | 4.489 s | 4.448 s | 4.301 s |
| 40 | cpu-pt | 4.886 s | 4.897 s | 4.912 s | 4.766 s |

Crystal size steps (analytic E = −2n):

| id | n | method | certified | E | dE | wall |
|----|--:|--------|-----------:|--:|----|------|
| `v1.crystal.n016.s000` | 16 | exact | True | -32.0 | 0 | 10.86 ms |
| `v1.crystal.n024.s000` | 24 | cpu-pt | False | -48.0 | 0 | 2.951 s |
| `v1.crystal.n032.s000` | 32 | cpu-pt | False | -64.0 | 0 | 3.881 s |
| `v1.crystal.n040.s000` | 40 | cpu-pt | False | -80.0 | 0 | 4.834 s |

### (b) Energy error vs known optimum

Rows with an oracle: **87**. Classical rows with dE = 0: **83**. Classical rows with dE ≠ 0: **0**.

Wherever an oracle exists on this ladder, **error is 0** for exact and PT rows (certified only when method=`exact`).

ER MaxCut and ±J glass have **no oracle** on 12 rows (n past enumeration). Those rows report energy and time only — strong minima, not certified.

MPS (not a `drift.solve` path) is variational:

| n | χ_eff | χ_kept | dE vs Lanczos | wall |
|--:|------:|-------:|---------------|------|
| 8 | 4 | 13 | 7.91e-05 | 20.29 ms |
| 10 | 4 | 16 | 0.000287 | 27.42 ms |
| 12 | 4 | 16 | 0.000706 | 40.22 ms |
| 14 | 4 | 16 | 0.00112 | 76.74 ms |

Never certified.

### Extra seeds (39 rows)

Stochastic families (`maxcut-er`, `pmj-glass`, `bipartite-maxcut`) carry extra seeds at cheap n (exact-reachable, plus one seed at the n=20 dispatcher wall). `ferro-chain` is seed-invariant and is **not** duplicated.

Extra-seed rows with an oracle: 39; dE = 0 on 39 of those.

### (c) Method vs n

The dispatcher does what it says:

- n ≤ 18 → `exact`, `certified=True` (61 rows)
- n > 18, GPU binary present → `gpu-pt`, `certified=False` (0 rows)
- n > 18, no GPU / `--no-gpu` → `cpu-pt`, `certified=False` (34 rows)
- `tfim-chain` → `mps`, `certified=False` (5 rows); skipped above the χ cutoff (2 rows)

No row has `certified=True` with a heuristic method. An empty `gpu-pt` band is a
missing binary (or `--profile local`), not a silent fallback pretending to be GPU.

### (d) χ vs n (TFIM / MPS only)

Open TFIM at Γ/J = 1, χ_max=16, n≤16:

| n | χ_eff | χ_kept | wall |
|--:|------:|-------:|------|
| 8 | 4 | 13 | 20.29 ms |
| 10 | 4 | 16 | 27.42 ms |
| 12 | 4 | 16 | 40.22 ms |
| 14 | 4 | 16 | 76.74 ms |
| 16 | 5 | 16 | 94.56 ms |

Effective χ stays small on this short schedule (a scale-path probe). Treat χ here as that probe, not a replacement for `docs/results/PHASE13-results.md`.

Skipped at the documented cutoff: `v1.tfim-chain.n020.s000`, `v1.tfim-chain.n024.s000`. Raise `--mps-n-max` locally; Phase 13 already measured n=48.

## Answer to the question

On these fixed families, **`drift.solve` has two time regimes**. Below n=18
the certified exact engine's wall time tracks 2ⁿ. Above that, CPU-PT (no GPU) grows with the PT loop (replicas × rounds × sweeps × work per sweep), not with 2ⁿ. **Solution quality matches every available oracle** on this run for exact and PT rows; PT rows stay `certified=False`. χ is a tensor-network number: it is defined on
the TFIM chain and is not claimed for MaxCut or spin glasses. GPU-PT is the same
experiment with a binary; it is not inferred from the CPU curve.

## Honesty

- Heuristic (cpu-pt / gpu-pt / mps) is never `certified`.
- ER MaxCut and ±J glass at n>20 have no known energy here.
- A CPU PT curve is not a GPU throughput claim (see Phase 14 for Gflips/s).
- This file records only methods that ran. Zero `gpu-pt` rows means the CUDA
  engine did not run, not that GPU time equals CPU time.
- The MPS ladder uses a reduced schedule so the sweep stays cheap; treat χ as a
  scale-path probe, not a replacement for `docs/results/PHASE13-results.md`.
- Extra ferro-chain seeds are omitted on purpose: that generator ignores `seed`.
