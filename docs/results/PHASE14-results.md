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

**~4–5× over the first build, and now n-independent** — while still reproducing the exact −22 / −33
and the same final energies. Honest remaining ceiling: **~0.23 Gflips/s is still latency-bound** by
the *serial single-spin-flip* (one flip per block per step, three block syncs each). The real
GPU-Ising leap (billions of flips/s) needs **parallel spin updates** — checkerboard / graph-colouring
so a whole independent set flips at once — which is **Phase 14c**. The microscope, pointed at itself:
each measurement named the next bottleneck, and we fixed the two cheap ones without a false claim.

**Honest scope.** Parallel tempering finds **strong minima, not certified optima** — DRIFT is a
microscope for computation at scale, not a SOTA solver; the certified answer stays the exact
engine's, on the small n where the two are pinned together. v1 is **dense-J** and modest replica
counts (16–64); sparse-J, multi-GPU, and routing `factor()` / large MaxCut through the engine are
future work. The scale is new; the honesty contract — cross-checked against exact, cost reported —
is the same one every DRIFT phase carries.
