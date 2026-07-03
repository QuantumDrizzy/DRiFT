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
solutions. **Measured throughput (`figures/phase14_gpu.png`):**

| n | 128 | 256 | 512 | 1024 | 2048 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| Gflips/s | 0.059 | 0.059 | 0.053 | 0.048 | 0.044 |
| time (400 rounds) | 0.11 s | 0.22 s | 0.49 s | 1.08 s | 2.36 s |

**Honest reading of the number — it is low, and *why* is the useful result.** ~0.05 Gflips/s looks
poor for a GPU, but a single-spin *flip* is not a unit of GPU work here: each accepted flip triggers
an O(n) field update over a **dense** J row, so at n = 2048 the engine is actually pushing on the
order of 10¹¹ field-updates/s — real, memory-bound GPU work. Two design choices, not idle silicon,
cap the flips/s:

1. **Dense J ⇒ O(n) per flip.** The benchmark graphs are *sparse* (≈10 neighbours), yet a flip
   touches all n fields. A **sparse-J (CSR)** representation makes a flip O(degree), not O(n) — the
   single biggest win, and already the flagged next step.
2. **Serial single-spin-flip + per-flip block syncs.** Within a replica, flips are sequential
   (flipping i changes j's field), and rejected flips still pay a full block-wide sync doing nothing.
   With only R = 32 replicas = 32 blocks, the design is latency-bound. Real speedup needs a
   **parallel-update scheme** — checkerboard/graph-colouring (large independent sets exist for the
   sparse graphs) or **population annealing** (thousands of one-thread walkers) to hide the latency.

So Phase 14 delivers a **correct, honestly benchmarked** GPU port whose measurement *names its own
bottleneck*: dense-O(n) flips on a latency-bound layout. That is the microscope pointed at itself —
the next phase (sparse-J + parallel updates) is where the GPU actually earns its keep.

**Honest scope.** Parallel tempering finds **strong minima, not certified optima** — DRIFT is a
microscope for computation at scale, not a SOTA solver; the certified answer stays the exact
engine's, on the small n where the two are pinned together. v1 is **dense-J** and modest replica
counts (16–64); sparse-J, multi-GPU, and routing `factor()` / large MaxCut through the engine are
future work. The scale is new; the honesty contract — cross-checked against exact, cost reported —
is the same one every DRIFT phase carries.
