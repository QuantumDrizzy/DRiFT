# Phase 14 — the GPU Ising engine (parallel tempering past the exact wall)

**Understood:** *the substrate, run at scale.* Phase 13 scaled the quantum 1-D chain; this scales
the **optimization face** — MaxCut, factoring, universal circuits, crystals, Hopfield, anything
that becomes an `IsingModel` — from the exact engine's ~22-spin wall to thousands of spins, on the
GPU. The method is the canonical GPU-Ising computation: **parallel tempering** (replica-exchange
Metropolis), R walkers held at a temperature ladder, swapping configurations between rungs so a
minimum found while hot flows down to the cold replica that reports it.

**Built:**

- `drift/solvers/parallel_tempering.py` — the **CPU reference** and algorithmic oracle: geometric
  temperature ladder, single-spin-flip sweeps with local-field maintenance, alternating even/odd
  adjacent-rung swaps with the Metropolis criterion `min(1, exp((β_r−β_{r+1})(E_r−E_{r+1})))`, and
  a tracked global best.
- `cuda/ising_pt.cu` — the **CUDA engine**: one block per replica; a replica keeps its spins and
  local field `f_i = (J·s)_i + h_i` so a flip is O(1) to score (`dE = 2 s_i f_i`) and O(n) to apply,
  with the block's threads doing that update in parallel and coalesced (J symmetric ⇒ read row i);
  temperature-swap exchange; cuRAND; fp64 energies. `build.bat` → `nvcc -O3 -arch=sm_120`.
- `drift/gpu.py` — Python glue, `parallel_tempering_gpu(model, …)`, drop-in with the CPU reference
  (same `PtResult`), so CPU and GPU are cross-checkable in one line.
- `experiments/phase14_gpu.py` — correctness cross-check + scaling benchmark (CPU-only if unbuilt).

**Validated — CPU reference** (`tests/test_parallel_tempering.py`, 5/5; full suite 63/63):

- Finds the **exact** ground energy on a random MaxCut (n=16), a frustrated ±J **spin glass**
  (n=14), and a **ferromagnet** (n=18, E=−(n−1)).
- **Replica exchange earns its keep:** on a frustrated glass PT reaches the exact minimum while a
  lone cold walker (a degenerate 1-rung ladder, no swaps) freezes strictly above it.
- The ladder is healthy: temperatures ascending, swap rate strictly in (0, 1).

The Phase-14 experiment confirms the CPU path matches exact on the acceptance instances:

```
    maxcut G(16, .5)      exact=  -22.000  CPU=  -22.000
    maxcut G(18, .4)      exact=  -33.000  CPU=  -33.000
    => all match exact
```

**Validated — GPU engine (built + measured, RTX 5060 Ti, sm_120, CUDA 13):** the acceptance test
**passes** — the GPU engine reproduces the exact ground energies bit-for-bit:

```
    maxcut G(16, .5)      exact=  -22.000  CPU=  -22.000  GPU=  -22.000
    maxcut G(18, .4)      exact=  -33.000  CPU=  -33.000  GPU=  -33.000
    => all match exact
```

It scales past the exact wall (n = 128 … 2048; 2²⁰⁴⁸ configurations), returning sane MaxCut
solutions.

### Phase 14 → 14b: measure, find the bottleneck, fix what's cheap

The first build (dense J, R = 32) measured **~0.05 Gflips/s, decaying with n** (0.059 → 0.044 from
n=128 to 2048). Low for a GPU — and the benchmark named *why*, so two levers followed:

1. **Sparse J (CSR).** A single-spin *flip* is not a unit of GPU work: each accepted flip updated an
   O(n) field over a **dense** J row, so throughput *decayed* with n. The benchmark graphs are sparse
   (~10 neighbours), so J is now stored as CSR and a flip touches only a spin's neighbours —
   **O(degree), not O(n)**. Throughput went **flat in n** (the decay is gone) and wall time dropped
   ~30 %. Correctness unchanged (same exact energies, same final E).
2. **Occupancy.** With CSR the field work is tiny, so the engine is now **latency-bound** by the
   serial flip (thread-0 decision + per-flip block syncs). R = 32 = 32 blocks left the SMs ~4× idle.
   Raising to **R = 128** (a finer PT ladder, so a *better* solver too) scales throughput ~4× at
   nearly constant wall time — measured occupancy curve at n=1024: 0.03 (R16) → 0.06 (R32) →
   0.23 (R128) Gflips/s, wall time flat to R≈128 then saturating.

**Result (sparse-J + R = 128, `figures/phase14_gpu.png`):**

| n | 128 | 256 | 512 | 1024 | 2048 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| Gflips/s | 0.23 | 0.23 | 0.23 | 0.23 | 0.22 |
| time (400 rounds) | 0.12 s | 0.23 s | 0.45 s | 0.93 s | 1.95 s |

**~4–5× over the first build, and now n-independent** — while still reproducing the exact −22 / −33.
The remaining ceiling was the *serial single-spin-flip* (one flip per block per step) — which
**Phase 14c** removes.

### Phase 14c — checkerboard / graph-colouring (the real leap)

Two spins that are **not adjacent** can flip at the same time. The graph is greedily coloured into
independent sets (no edge within a colour), and a whole colour flips in parallel: each of its spins
reads its neighbours' *current* spins (all other colours, so stable), scores dE, and decides
independently. A sweep is now **k colour-steps** (k = #colours, small for a sparse graph) instead of
n serial flips — the latency wall is gone. No stored field (dE recomputed from CSR neighbours);
energy is recomputed exactly once per sweep, so `best_E` never drifts and the exact-match test holds.

**Result (checkerboard + sparse-J + R = 128, `figures/phase14_gpu.png`):**

| n | 128 | 256 | 512 | 1024 | 2048 | 4096 | 8192 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Gflips/s | 0.53 | 0.51 | 0.67 | 0.89 | **0.93** | 0.90 | 0.64 |
| time (400 rounds) | 0.05 s | 0.10 s | 0.16 s | 0.24 s | 0.45 s | 0.93 s | — |

Throughput now **rises with n** (bigger graphs = more spins per colour = more parallel work), peaking
~**0.94 Gflips/s**, and the engine reaches **n = 8192** (2⁸¹⁹² configurations). Correctness is intact
— exact −22 / −33, same-quality minima.

### Phase 14d — warp per replica (breaking 1 Gflips/s)

Phase 14c used a whole 256-thread **block** per replica. Phase 14d drops to one **warp** (32 lanes)
per replica, 8 replicas per block. The per-colour barrier becomes a `__syncwarp` (implicit-lockstep,
near-free) instead of a block-wide `__syncthreads`, and the energy reduction is a warp shuffle
instead of a shared-memory reduction. A replica is now *cheap*, so many more run concurrently.

Honest trade-off: a warp gives less *intra*-replica parallelism, so at a low replica count 14d is
**slower** than 14c (0.51 vs 0.93 Gflips/s at R=128 — the GPU is under-filled). Its regime is **many
replicas** — which is what a fine PT ladder wants anyway. At **R = 256** (its sweet spot) it runs at
a **stable ~0.94 Gflips/s @ n=2048** (5-run median; an earlier single run read 1.03, but that was a
boost-clock outlier — the honest, repeatable number is 0.94). Correctness intact (exact −22 / −33).

### Recompute energy less often — the lever that *didn't* move the needle (and why that's the result)

The obvious next optimisation: the replica energy was recomputed from scratch (O(nnz)) once per
round for the swap + best-check. That's replaced with an **incremental** energy — each accepted flip
already computes its dE for the Metropolis test, and Σ dE telescopes to the *exact* energy change
(flips within a colour are independent), so no per-round recompute is needed. Verified exact over
600 rounds × 5 sweeps: `best_E` equals the true energy of `best_s` to fp precision, still matching the
exact ground energy — **no drift**.

But the **measured throughput did not change** (~0.94 Gflips/s, 5-run median, before and after). The
honest conclusion: the energy recompute was *not* the bottleneck. The kernel is **memory-bound on the
scattered CSR neighbour reads** — `neigh_sum` gathers `colIdx[t]`, `weight[t]`, and `sr[colIdx[t]]`
uncoalesced for every spin. Removing the recompute is kept (it's correct, cleaner, and less work),
but it named the *real* next lever: **coalescing / caching those neighbour reads** (reorder for
locality, or stage the replica's spins in shared memory). Distrust the pretty number; measure, and
let the measurement point at the truth.

### Phase 14e — shared-memory staging (the lever that *did* move it)

The fix the measurement pointed at: a replica's spins are read many times per sweep (once per
neighbour edge), so at the start of each round each warp **stages its replica's spins into shared
memory as int8 (±1)**, runs all the sweeps against shared memory, and writes back to global once at
the end. The hot scattered read `sr[colIdx[t]]` (global, uncoalesced) becomes a shared-memory access;
`WARPS_PER_BLOCK · n` bytes of dynamic shared per block, with a `MaxDynamicSharedMemorySize` opt-in
past 48 KB (so it still reaches n = 8192 = 64 KB on Blackwell).

**Measured, stable (5-run median), R = 256:**

| n | 128 | 256 | 512 | 1024 | 2048 | 4096 | 8192 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Gflips/s | 0.94 | 0.90 | 1.01 | 1.19 | **1.29** | 1.30 | 1.36 |

**~0.94 → ~1.29 Gflips/s at n=2048 (~1.37×), peak ~1.36 at n=8192** — exactly where the memory-bound
diagnosis said the gain was, correctness intact (exact −22 / −33, same minima). The wrong lever
(energy recompute) and the right one (shared staging) came from the *same* honest measurement.

### The whole arc, measured

| build | flips/s @ n=2048 | wall @ n=2048 |
|:---|:---:|:---:|
| Phase 14 — dense J, R=32 | 0.044 G | 2.36 s |
| Phase 14b — sparse-J (CSR) | 0.06 G | — |
| Phase 14b — + R=128 occupancy | 0.22 G | 1.95 s |
| Phase 14c — checkerboard (block/replica) | 0.93 G | 0.45 s |
| Phase 14d — warp/replica, R=256 | ~0.94 G (stable) | 0.89 s |
| **Phase 14e — shared-memory staging** | **~1.29 G** | 0.65 s |

**~0.044 → ~1.29 Gflips/s (~29×) over five measured steps — each one named the next bottleneck, and
none claimed a speed it hadn't shown** (including correcting a boost-clock outlier down to the stable
rate). The microscope, pointed at itself: dense-O(n) → sparse-J → checkerboard parallel updates →
warp-level replicas → spins staged in shared memory, and the exact −22 / −33 held at every step,
scaling to n = 8192. The serial-flip wall *and* the memory wall that made the first GPU build no
faster than a CPU are both gone; the next levers are multi-GPU and routing `factor()` / large MaxCut
through the engine.

**Honest scope.** Parallel tempering finds **strong minima, not certified optima** — DRIFT is a
microscope for computation at scale, not a SOTA solver; the certified answer stays the exact
engine's, on the small n where the two are pinned together. v1 is **dense-J** and modest replica
counts (16–64); sparse-J, multi-GPU, and routing `factor()` / large MaxCut through the engine are
future work. The scale is new; the honesty contract — cross-checked against exact, cost reported —
is the same one every DRIFT phase carries.
