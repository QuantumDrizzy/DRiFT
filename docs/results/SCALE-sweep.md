# Scale path — wall time, quality, and χ vs n via `drift.solve`

**Question (falsifiable):** How do wall-clock time and solution quality (and χ where an
MPS/tensor path applies) scale with system size n for fixed instance families, when
solving via `drift.solve` (exact → GPU-PT → CPU-PT)?

**Status:** measured on CPU (no GPU binary in this environment). Larger-n / GPU-PT is the
same runner with a compiled `cuda/ising_pt` binary.

![scale sweep](../../figures/scale/scale_sweep.png)

## What was measured

A **versioned instance bank** (`drift/benchmarks/instances/manifest.json`, bank `v1`) with
stable IDs such as `v1.maxcut-er.n008.s001`. Generators are seeded; n≤8 classical instances
are also checked in as JSON fixtures so a generator change is loud. Families:

| Family | What it is | Known energy |
|--------|------------|--------------|
| `maxcut-er` | Erdős–Rényi MaxCut G(n, 1/2) | exact enum, n≤20 |
| `pmj-glass` | complete ±J spin glass, h=0 | exact enum, n≤20 |
| `bipartite-maxcut` | random bipartite MaxCut | analytic: E = −n_edges |
| `ferro-chain` | open 1-D ferromagnet | analytic: E = −(n−1) |
| `crystal` | period-4 stripe crystal 4×4 / 4×8 | analytic: E = −2n |
| `tfim-chain` | open TFIM at Γ/J = 1 | Lanczos, n≤14; χ via MPS |

Classical families go through **`drift.solve`**. χ is recorded **only** on `tfim-chain`
through the existing MPS/TEBD solver (`drift.mps`). That path is 1-D nearest-neighbour TFIM:
it does **not** apply to dense MaxCut, spin glasses, or 2-D crystals.

**MPS cutoff (documented):** default sweep χ ladder is **n≤16** (`MPS_N_MAX_SWEEP`). n=20 and
n=24 sit in the bank for local runs (`--mps-n-max 24`). Phase 13 has shown n=48 with a larger
budget; that is out of scope for this CPU curve.

A heuristic minimum is **never** marked `certified`. GPU is optional: if the CUDA binary is
missing, n > `exact_max` (18) continues on CPU-PT.

## How to run

```bash
python -m experiments.scale_sweep              # documented CPU ladder (n≤32)
python -m experiments.scale_sweep --n-max 24   # stop before the n=32 PT rung
python -m experiments.scale_sweep --ci         # n≤10, no MPS — CI-safe
```

Results: `docs/results/scale_sweep.csv`, `docs/results/scale_sweep.json`,
`figures/scale/scale_sweep.png`.

## Curves (CPU, this commit)

Hardware: CPU, no `cuda/ising_pt` in this environment. `exact_max=18`. CPU-PT uses `drift.solve`
defaults (32 replicas, 400 rounds, 4 sweeps/round). MPS uses χ_max=16, 12 sweeps, a short
imaginary-time schedule (a scale probe, not a high-precision Phase-13 rerun).

### (a) Wall time vs n

Exact enumeration grows exponentially, as it must (2ⁿ configurations):

| n | maxcut-er | pmj-glass | bipartite | ferro-chain |
|--:|----------:|----------:|----------:|------------:|
| 8 | 0.27 ms | 0.10 ms | 0.09 ms | 0.07 ms |
| 12 | 0.83 ms | 0.55 ms | 0.53 ms | 0.52 ms |
| 16 | 16 ms | 6.2 ms | 5.9 ms | 6.2 ms |
| 18 | 58 ms | 28 ms | 24 ms | 24 ms |

At n=20 the dispatcher switches to **CPU-PT**. Wall time jumps to ~2.5 s and then grows
roughly with n (the Python PT loop is O(replicas × rounds × sweeps × n²) on dense J):

| n | method | wall (s), four families ≈ |
|--:|--------|---------------------------|
| 20 | cpu-pt | 2.46 – 2.51 |
| 24 | cpu-pt | 2.91 – 3.07 |
| 32 | cpu-pt | 3.91 – 4.25 |

Crystal 4×4 (n=16) is certified-exact in 6.7 ms; 4×8 (n=32) is CPU-PT in 4.17 s.
**gpu-pt did not run** — the binary is absent; the notes column says so on every
heuristic row. On a machine with `cuda/ising_pt`, the same IDs would record `gpu-pt`
for n>18 and a much flatter time curve (Phase 14).

### (b) Energy error vs known optimum

Wherever an oracle exists, **error is 0** on this ladder — including CPU-PT:

- exact rows (n≤18, every family): E − E_known = 0, **certified=True**
- CPU-PT vs stored exact at n=20 (MaxCut ER and ±J glass): error = 0, **certified=False**
- CPU-PT vs analytic through n=32 (bipartite, ferro, crystal 4×8): error = 0, **certified=False**

MaxCut ER and the spin glass have **no oracle at n=24, 32** (2²⁴ and 2³² are past
enumeration). Those rows report energy and time only — strong minima, not certified.
That is the honest gap, not a missing column.

MPS (not a `drift.solve` path) is variational: n=8 error 7.9×10⁻⁵ vs Lanczos; n=12
error 7.1×10⁻⁴. Never certified.

### (c) Method vs n

The dispatcher does what it says:

- n ≤ 18 → `exact`, `certified=True` (25 rows)
- n > 18, no GPU → `cpu-pt`, `certified=False`
- `tfim-chain` → `mps`, `certified=False` (n≤16); skipped above the χ cutoff

No row has `certified=True` with a heuristic method. The empty `gpu-pt` band is the
missing binary, not a silent fallback pretending to be GPU.

### (d) χ vs n (TFIM / MPS only)

Open TFIM at Γ/J = 1, χ_max=16:

| n | χ_eff | χ_kept | wall |
|--:|------:|-------:|-----:|
| 8 | 4 | 13 | 18 ms |
| 12 | 4 | 16 | 44 ms |
| 16 | 5 | 16 | 98 ms |

Effective χ stays small (the chain is not deeply entangled at this budget and Γ); the
truncation cap is hit (`χ_kept=16`) from n=12. Time grows with n, far slower than 2ⁿ.
n=20 and 24 are in the bank and were **skipped** at the documented cutoff
(`MPS skipped: n=20 > mps_n_max=16`). Raise `--mps-n-max` locally; Phase 13 already
measured n=48.

## Answer to the question

On these fixed families, **`drift.solve` has two time regimes**. Below n=18 the certified
exact engine's wall time tracks 2ⁿ (milliseconds → tens of milliseconds). Above n=18,
with no GPU, CPU-PT takes a few seconds and grows gently with n, not with 2ⁿ. **Solution
quality matches every available oracle** (stored exact at n=20; analytic bipartite / ferro
/ crystal through n=32) but those PT rows stay `certified=False`. χ is a tensor-network
number: it is defined on the TFIM chain, stays O(1) on n≤16 at Γ=1, and is not claimed
for MaxCut or spin glasses. GPU-PT is the same experiment with a binary; it is not
inferred from the CPU curve.

## Honesty

- Heuristic (cpu-pt / gpu-pt / mps) is never `certified`.
- ER MaxCut and ±J glass at n≥24 have no known energy here.
- This CPU PT curve is not a GPU throughput claim (see Phase 14 for Gflips/s).
- The MPS ladder uses a reduced schedule so the sweep stays cheap; treat χ as a
  scale-path probe, not a replacement for `docs/results/PHASE13-results.md`.
