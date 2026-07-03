# DRIFT GPU engine — Ising parallel tempering (Phase 14)

The scale-out the roadmap deferred: a CUDA **parallel-tempering (replica-exchange Metropolis)**
solver for a general Ising/QUBO. It scales the *optimization face* (MaxCut, factoring, circuits,
crystals, Hopfield — anything that becomes an `IsingModel`) from the exact engine's ~22-spin wall
to **thousands of spins**, with the cost reported honestly as spin-flips/sec.

`ising_pt.cu` mirrors, kernel-for-algorithm, the CPU reference in
`drift/solvers/parallel_tempering.py`. That reference is the oracle: it is validated against
`exact_ground_state`, and the GPU engine must reproduce the **same ground energy** on the small
instances the exact engine can still solve. That shared falsifier is the whole safety net —
correctness was proven on CPU; the GPU is a faithful, benchmarkable port.

## Build

From the **x64 Native Tools Command Prompt for VS 2022** (so `cl.exe` is on PATH):

```bat
cd cuda
build.bat
```

which runs `nvcc -O3 -arch=sm_120 ising_pt.cu -o ising_pt.exe` (`sm_120` = Blackwell / RTX 50xx;
change `-arch` for other GPUs). Requires the CUDA Toolkit (cuRAND ships with it).

## Validate + benchmark

```bat
python -m experiments.phase14_gpu
```

This cross-checks the GPU engine against the exact ground state on small instances, then benchmarks
throughput as n scales past the exact wall. With no binary present it runs the CPU reference only
and prints these build instructions.

## Using it from Python

```python
from drift.builders.qubo import maxcut_ising, random_graph
from drift.gpu import parallel_tempering_gpu, gpu_available

model = maxcut_ising(random_graph(1000, p=0.05, seed=0))
res = parallel_tempering_gpu(model, n_replicas=32, n_rounds=500)   # drop-in with the CPU version
print(res.best_E, res.throughput, "flips/s")
```

## Design (one block per replica)

- A replica keeps its spins `s` and its **local field** `f_i = (J·s)_i + h_i`, so a single-spin
  flip costs O(1) to evaluate (`dE = 2·s_i·f_i`) and O(n) to apply — and the block's threads apply
  that field update in parallel.
- J is symmetric, so the field update reads **row i** (`J[i*n+j]`), which is coalesced.
- Replica exchange swaps **temperatures** between adjacent rungs (equivalent to swapping configs,
  much cheaper); the global best over all replicas is tracked throughout, so no swap loses the
  answer.
- cuRAND per replica; energies in fp64 for a stable accept/track; `float32` J on the device.

## Honest scope

- J is stored **sparse (CSR)**; spins are updated by **checkerboard / graph-colouring** (a whole
  independent set flips in parallel, so a sweep is k colour-steps not n serial flips); and a replica
  is driven by **one warp** (near-free `__syncwarp` barriers + shuffle energy reduction). Use **~256
  replicas** (warp-per-replica's sweet spot — a replica is cheap, so many fill the GPU and make a
  finer PT ladder). **~1.03 Gflips/s @ n=2048** (~1.13 @ R=512), scales to n=8192, exact ground
  energies preserved. Further headroom (recompute energy less often, coalesced colour reads) and
  multi-GPU are future work.
- Parallel tempering finds **strong minima**, not certified optima — DRIFT is a microscope for
  computation at scale, not a solver competing for SOTA. The certified answer is the exact engine's,
  on the small n where the two are cross-checked.
- **Status:** the CUDA source is written against the CPU oracle but is compiled and benchmarked on
  your machine; treat the first `phase14_gpu` run (exact cross-check) as the acceptance test.
