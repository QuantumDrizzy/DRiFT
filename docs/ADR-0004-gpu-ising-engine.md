# ADR-0004: The GPU Ising engine — parallel tempering past the exact wall

**Status:** Accepted (CPU reference landed; CUDA kernel written, on-device benchmark pending)
**Date:** 2026-07-02
**Deciders:** Antonio (QuantumDrizzy)

## Context

Through Phase 13 DRIFT reads ground states two ways: the exact engine (brute force / full-matrix
Lanczos, ~22 spins) and the Phase-13 MPS solver (1-D quantum chains, χ-bounded). The **optimization
face** — MaxCut, factoring, universal circuits, crystals, Hopfield, anything that becomes an
`IsingModel` — is still capped at the exact wall or left to the single-walker CPU
`simulated_annealing`. The ROADMAP explicitly defers *"large-scale GPU solving to a Rust/CUDA port
if and when a phase needs it."* A phase needs it now, and the hardware is here: RTX 5060 Ti
(Blackwell, sm_120), CUDA 13.

## Decision

Build a GPU **parallel-tempering (replica-exchange Metropolis)** solver for a general Ising/QUBO —
the canonical GPU-Ising computation — with a **CPU reference as the oracle**:

- `drift/solvers/parallel_tempering.py` — the CPU reference (R replicas on a geometric temperature
  ladder, single-spin-flip sweeps, adjacent-rung swaps), validated against `exact_ground_state`.
- `cuda/ising_pt.cu` — the CUDA engine: one block per replica, local-field maintenance, coalesced
  field updates via J's symmetry, temperature-swap exchange, cuRAND. Compiled with `nvcc -arch=sm_120`.
- `drift/gpu.py` — Python glue: drop-in `parallel_tempering_gpu(model, …)` returning the same
  `PtResult`, so the CPU and GPU paths are interchangeable and cross-checkable.

**Language split (doctrine: right language per domain).** Kernel in **C++/CUDA** (least Windows
friction, directly benchmarkable), glue and faces in **Python**. A Rust/`cudarc` wrapper was the
alternative (aesthetic fit) but adds CUDA-on-Windows friction with no benefit for a benchmark
artifact.

## Options Considered

### Option A: GPU parallel tempering, general J  ⭐
| Dimension | Assessment |
|-----------|------------|
| Scales | the whole optimization face → thousands of spins |
| Honesty | strong minima, cross-checked vs exact on small n; not a SOTA claim |
| Fit | canonical GPU-Ising; matches Antonio's CUDA strength |

**Pros:** one engine lifts every face; clean CPU oracle; real throughput benchmark.
**Cons:** dense-J O(n²) memory; single-spin-flip is inherently sequential per replica (mitigated by
one-block-per-replica intra-replica parallelism).

### Option B: GPU MPS/TEBD (large χ)
**Pros:** extends Phase 13 to 2-D / high entanglement. **Cons:** useless in 1-D; needs batched
cuSOLVER SVD; heavy. → deferred to a later phase.

### Option C: GPU brute force (2ⁿ)
**Pros:** trivially correct. **Cons:** pushes the wall only a few spins; memory-bound. → rejected.

## Trade-off Analysis

A is the only option that changes the *scale* of what DRIFT can observe rather than nudging it. It
keeps the honesty contract intact: the exact engine remains the certified oracle on small n, the GPU
is the microscope at large n, and the two are pinned together by a shared falsifier (identical
ground energy where both run). It also matches the machine and the builder.

## Consequences

- **Easier:** every optimization-face problem can now be run at n ≫ 22; a genuine spin-flips/sec
  benchmark exists; the CPU reference is a better solver than plain SA even without a GPU.
- **Harder / to revisit:** dense-J memory limits very large n (sparse-J is future work); the CUDA
  path is verified on-device by Antonio, not in CI; fp32 J may need fp64 for pathological instances.
- **Honest status:** the CUDA source is written against the tested CPU oracle but **not yet compiled
  or benchmarked** — the acceptance test is the first `python -m experiments.phase14_gpu` run
  reproducing the exact ground energy on the small instances.

## Action Items

1. [x] CPU reference `parallel_tempering` + tests vs exact (5/5).
2. [x] CUDA engine `ising_pt.cu` + `build.bat` + `drift/gpu.py` glue + `experiments/phase14_gpu.py`.
3. [ ] `build.bat` on the x64 Native Tools prompt; confirm exact cross-check passes on-device.
4. [ ] Record real throughput (Gflips/s) vs n in `docs/results/PHASE14-results.md` + the figure.
5. [ ] Follow-ups: sparse-J path; wire `factor()` / large MaxCut through the GPU engine.
