"""Experiment: Phase 14 — the GPU Ising engine (parallel tempering past the exact wall).

Two things to see:
  (1) Correctness — on instances the exact engine can still solve, both the CPU reference and the
      GPU engine reach the exact ground energy. This is the acceptance test for the CUDA port.
  (2) Scale — the GPU engine solves Ising problems with thousands of spins (2ⁿ utterly beyond
      brute force), reporting throughput in spin-flips/sec.

Runs the CPU reference always; runs the GPU engine and the scaling benchmark only if
`cuda/ising_pt.exe` has been built (else it prints the build instructions and skips).

    python -m experiments.phase14_gpu
"""

from __future__ import annotations

import numpy as np

from drift.builders.qubo import cut_value, maxcut_ising, random_graph
from drift.solvers.exact import exact_ground_state
from drift.solvers.parallel_tempering import parallel_tempering
from drift import gpu


def _random_bipartite(n: int, p: float, seed: int) -> np.ndarray:
    """A random bipartite graph on two equal halves — edges only cross the split. Its maximum cut is
    every edge (the planted 2-colouring cuts them all), so it is a *known optimum* at any size."""
    rng = np.random.default_rng(seed)
    w = np.zeros((n, n))
    half = n // 2
    for i in range(half):
        for j in range(half, n):
            if rng.random() < p:
                w[i, j] = w[j, i] = 1.0
    return w


def _small_instances():
    return {
        "maxcut G(16, .5)": maxcut_ising(random_graph(16, p=0.5, seed=3)),
        "maxcut G(18, .4)": maxcut_ising(random_graph(18, p=0.4, seed=8)),
    }


def main(outdir: str = "figures") -> None:
    # (1) correctness — CPU reference (and GPU if built) vs exact ---------------
    print("(1) correctness vs exact ground state:")
    ok = True
    rows = []
    for name, model in _small_instances().items():
        _, e_exact, _ = exact_ground_state(model)
        cpu = parallel_tempering(model, n_replicas=16, T_min=0.05, T_max=4.0,
                                 n_rounds=200, sweeps_per_round=4, seed=1)
        line = f"    {name:20s}  exact={e_exact:9.3f}  CPU={cpu.best_E:9.3f}"
        gpu_E = None
        if gpu.gpu_available():
            g = gpu.parallel_tempering_gpu(model, n_replicas=16, T_min=0.05, T_max=4.0,
                                           n_rounds=200, sweeps_per_round=4, seed=1)
            gpu_E = g.best_E
            line += f"  GPU={gpu_E:9.3f}"
            ok = ok and np.isclose(gpu_E, e_exact)
        ok = ok and np.isclose(cpu.best_E, e_exact)
        rows.append((name, e_exact, cpu.best_E, gpu_E))
        print(line)
    print(f"    => {'all match exact' if ok else 'MISMATCH — investigate'}\n")

    if not gpu.gpu_available():
        print("GPU engine not built — skipping the scaling benchmark.")
        print(f"  build it: cd cuda && build.bat   (expects {gpu.gpu_binary_path()})")
        return

    # (2) scale — throughput past the exact wall -------------------------------
    print("(2) scaling benchmark (n far past the 2²² exact wall):")
    ns = [128, 256, 512, 1024, 2048]
    thr = []
    for n in ns:
        model = maxcut_ising(random_graph(n, p=min(0.1, 20.0 / n), seed=0))
        # 256 replicas: with a warp per replica (Phase 14d), a replica is cheap, so more of them fill
        # the GPU better AND make a finer PT ladder (better mixing). 256 is the measured sweet spot.
        g = gpu.parallel_tempering_gpu(model, n_replicas=256, T_min=0.05, T_max=5.0,
                                       n_rounds=400, sweeps_per_round=4, seed=0)
        thr.append(g.throughput or 0.0)
        cut = cut_value(  # a concrete, human-readable readout of the solution's quality
            random_graph(n, p=min(0.1, 20.0 / n), seed=0), g.best_s)
        print(f"    n={n:5d}  E={g.best_E:12.1f}  cut={cut:8.1f}  "
              f"{(g.throughput or 0)/1e9:6.2f} Gflips/s  {g.seconds or 0:.3f}s")

    # (3) solution quality vs a KNOWN optimum ----------------------------------
    # Fast is not the same as good. A bipartite graph's maximum cut is *all* its edges (the planted
    # 2-colouring cuts every edge), so it is a known optimum at any n — and the engine must find it.
    print("\n(3) solution quality vs known optimum (bipartite: max cut = every edge):")
    for n in (128, 256, 512, 1024):
        w = _random_bipartite(n, p=min(0.08, 20.0 / n), seed=1)
        edges = int(np.count_nonzero(np.triu(w)))
        g = gpu.parallel_tempering_gpu(maxcut_ising(w), n_replicas=256, T_min=0.05, T_max=6.0,
                                       n_rounds=500, sweeps_per_round=5, seed=0)
        cut = int(round(cut_value(w, g.best_s)))
        print(f"    n={n:5d}  edges={edges:6d}  GPU cut={cut:6d}  ratio={cut / edges:.4f}  "
              f"{'OPTIMAL' if cut == edges else 'suboptimal'}")

    # ── figure ────────────────────────────────────────────────────────────────
    try:
        from drift.viz import _save, _style, CYAN
    except Exception as exc:
        print(f"  (skipping figure: {exc})")
        return
    _style()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(ns, np.array(thr) / 1e9, "o-", color=CYAN, lw=2)
    ax.set_xlabel("number of spins  n  (2ⁿ ≫ exact wall)")
    ax.set_ylabel("throughput  (Gflips/s)")
    ax.set_title("Phase 14 — GPU Ising parallel tempering: scale")
    ax.set_xscale("log", base=2)
    path = _save(fig, f"{outdir}/phase14_gpu.png")
    plt.close(fig)
    print(f"  saved {path}")


if __name__ == "__main__":
    main()
