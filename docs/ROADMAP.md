# DRIFT — Roadmap

Each phase has an **understanding goal** (what you'll be able to *see/measure* afterward)
and a **deliverable** (code + at least one figure + a `docs/results/PHASE{N}-results.md`
note). Phases are incremental: every one builds on the shared engine. No phase is about
proving anything — each one makes a piece of the process *observable*.

> Legend: ⬜ not started · 🔄 doing · ✅ done

---

### Phase 0 — Scaffolding ✅
- **Understand:** the shape of the whole thing before any code.
- **Deliverable:** README, ADR-0001, this roadmap, CONCEPTS glossary.

### Phase 1 — The engine + observability ✅  *(see [results](results/PHASE1-results.md))*
- **Understood:** *matter computes.* The relaxation figure shows a spin system exploring
  hot, then freezing into its exact ground state as it cools.
- **Built:** `IsingModel(J, h)`; solvers = **simulated annealing** + **exact** brute force;
  metrics (energy, magnetization, Landauer floor); dark-palette viz.
- **Validated:** 2D ferromagnet `L=4` → exact = annealing = `-32` (`= -2nJ`), SA reaches it.
- **Figures:** `figures/phase1_{relaxation,groundstate,landscape}.png`.

### Phase 2 — Face ①: Optimization (QUBO) ✅  *(see [results](results/PHASE2-results.md))*
- **Understood:** *optimization = ground state.* MaxCut as a purely antiferromagnetic
  Ising; the maximum cut is the lowest-energy spin configuration.
- **Built:** `qubo` builder (`qubo_to_ising`, `maxcut_ising`, `cut_value`, `random_graph`)
  + `plot_graph_cut`.
- **Validated:** `G(n=14, p=0.5)`, 43 edges → exact = annealing = **31-edge cut** (E = -19).
- **Figures:** `figures/phase2_{relaxation,maxcut}.png`.

### Phase 3 — Quantum ground state + χ (the thermometer) ✅  *(see [results](results/PHASE3-results.md))*
- **Understood:** *how much* a state computes. The transverse-field Ising chain's ground
  state, with χ = the bond dimension a tensor network needs. χ stays small in the ordered
  (cat) and disordered (polarized) phases and **peaks at the quantum phase transition**.
- **Built:** `drift/quantum.py` — sparse TFIM, Lanczos ground state, entanglement entropy
  (SVD), Blaze-style `effective_chi`; `plot_chi_sweep`.
- **Validated:** `n=14` chain; χ: 4 (cat) → 6 (critical) → 3 (polarized); peak at Γ≈0.77
  (Γ_c = 1, finite-size shift, stated honestly).
- **Figure:** `figures/phase3_chi.png`.  *The Blaze lesson, applied as a probe.*

### Phase 4 — Face ④: Neural memory (Hopfield) ✅  *(see [results](results/PHASE4-results.md))*
- **Understood:** *memory = the self-assembly of an attractor.* The neuroscience face — the
  same engine with Hebbian couplings; recall is relaxing to the nearest energy minimum.
- **Built:** `hopfield` builder (`hopfield_model`, `recall`, `add_noise`, `overlap`) +
  `plot_recall`.
- **Validated:** `n=100`, 3 memories (within capacity ≈14); a 25%-corrupted cue (overlap
  0.5) recovers to overlap **1.0** — perfect recall. (Hopfield, Nobel Physics 2024.)
- **Figure:** `figures/phase4_recall.png` — noisy X relaxing back to the clean stored X.

### Phase 5 — Face ②: Self-assembly (aTAM / tiles) ✅  *(see [results](results/PHASE5-results.md))*
- **Understood:** *constructive nanotech.* Wang tiles binding by glue affinity; the target
  structure is the arrangement with the most bonds = the ground state.
- **Built:** `tiles` builder (`jigsaw`, `tiles_qubo` one-hot + bond reward, `decode_tiling`,
  `count_bonds`, `render`) + `plot_assembly`.
- **Validated:** 2×2 jigsaw exact (4/4 bonds = target); 3×3 'plus' (81 spins) anneals to
  **12/12 bonds = exact target** (8 restarts — the rigid puzzle sits in a narrow basin,
  stated honestly).
- **Figure:** `figures/phase5_assembly.png` — hot disorder → assembled plus.

### Phase 6 — Face ③: Self-replication ✅  *(see [results](results/PHASE6-results.md))*
- **Understood:** *grey goo, contained.* Replication as crystallization — a periodic ground
  state from translation-invariant frustrated couplings (`h = 0`; the motif emerges, it is
  not painted in).
- **Built:** `crystal` builder (`crystal_2d` J1/J2/Jy, `column_period`, `is_striped`) +
  `plot_crystal`.
- **Validated:** 4×4 exact = period-4 stripe crystal (E = -2n); 12×12 anneals from a hot
  melt to the same crystal (E = -288 = -2n, period 4). Degeneracy → domains, stated honestly.
- **Figure:** `figures/phase6_crystal.png` — disordered melt → striped crystal, unit cell marked.

### Phase 7 — The microscope (synthesis) ✅  *(see [results](results/PHASE7-results.md))*
- **Understood:** one Ising engine + pluggable builders = four faces of "matter computes",
  and real hardware sits 6+ orders of magnitude above the Landauer floor — the headroom
  toward computronium.
- **Built:** `experiments/phase7_microscope.py` (runs all four faces back-to-back from the
  same engine), `plot_four_faces` (2×2 ground-state panel), `plot_cosmic_roofline` (log
  J/op axis with the Landauer wall and Margolus–Levitin floors).
- **Figures:** `figures/phase7_four_faces.png` (optimization, assembly, replication, memory
  side by side); `figures/phase7_roofline.png` (real systems vs. physical limits).
- **Honest scope:** the roofline uses textbook order-of-magnitude landmarks (Landauer and
  Margolus–Levitin are first-principles; the rest are device estimates).

### Phase 8 — The dynamical (reservoir) face ✅  *(see [results](results/PHASE8-results.md))*
- **Understood:** *matter computes in time, not just at rest.* The Ising substrate driven as a
  physical reservoir has a measurable compute capacity that peaks at the **edge of chaos**.
- **Built:** `drift/reservoir.py` — `IsingReservoir` (spectral radius set via the **Spectra**
  spine), Jaeger **memory capacity**, **separation**, and kernel/generalization rank.
- **Validated:** MC = 43.5 at N=200, peaking at spectral radius ρ ≈ 1.0; DRIFT's first
  automated suite (`tests/test_reservoir.py`, 5/5). **Figure:** `figures/phase8_reservoir.png`.

### Phase 9 — The optimization face, run quantum ✅  *(see [results](results/PHASE9-results.md))*
- **Understood:** *the Phase-2 ground state, reached by quantum adiabatic evolution — and its
  cost is the spectral gap, not cleverness.* Quantum annealing is not magic: when the gap
  closes, it fails too.
- **Built:** `drift/anneal.py` — `H(s) = (1−s)(−ΣXᵢ) + s·H_problem` with a diagonal
  `H_problem` built straight from `IsingModel.energy`; `spectral_gap_path`, real-time
  `quantum_anneal` (Krylov `expm_multiply`), `success_probability`.
- **Validated:** slow anneal → ground state (success ≈ 1.00), sudden quench fails (< 0.01);
  shrinking Δ_min 0.96 → 0.27 drops success 0.99 → 0.60 at fixed T (the honest gap-bounded
  limit). `tests/test_anneal.py`, 5/5. **Figure:** `figures/phase9_quantum_anneal.png`.
- **Honest scope:** the mechanism on small `n` (exact state-vector), **not** a speed claim over
  classical SA; the closing-gap problem is the deliberate falsifier.

### Phase 10 — Quantum vs simulated annealing, honestly ✅  *(see [results](results/PHASE10-results.md))*
- **Understood:** *the deferred question — is quantum better? — answered honestly: it depends on
  what is hard.* The quantum edge is specific (a thin tunnelable barrier), not general.
- **Built:** `drift/tunneling.py` — both annealers on the *same* landscape: `hamming_cost_energies`
  (a Hamming-weight funnel with an optional thin Farhi **spike**), `metropolis_sa` (single-spin-
  flip), and the Phase-9 `quantum_anneal` for the quantum side.
- **Validated:** on the spike, SA is walled out (success 0.17) while QA tunnels it (0.45 at T=40,
  ~2.6×); on the funnel neither has an edge (≈1.0); a taller spike costs QA too. `tests/
  test_tunneling.py`, 4/4 (deterministic). **Figure:** `figures/phase10_tunneling.png`.
- **Honest scope:** the well-understood tunnelling-vs-barrier effect on small `n` — *not* a
  "quantum supremacy" claim; wide barriers and closing gaps give the quantum route no edge.

### Phase 11 — Factoring as an Ising ground state ✅  *(see [results](results/PHASE11-results.md))*
- **Understood:** *matter computing arithmetic* — encode `p·q = N` so the energy minimum *is*
  the factorization. The optimization face (Phase 2) pointed at number theory.
- **Built:** `drift/factoring.py` — `factoring_qubo` (odd-factor binary encoding, products
  linearised by AND-gadget auxiliaries, objective `(N−p·q)²`) and `factor` (QUBO → Ising →
  exact ground state → decode). Energy is exactly 0 at a valid factorization.
- **Validated:** DRIFT's engine factors `15, 35, 143, … 221 = 13×17` (energy 0); the QUBO's own
  brute-force minimum agrees; and `factor(143)` with default widths *raises* (23 > 22 vars) —
  the construction is honest about its wall. `tests/test_factoring.py`, 5/5.
- **Honest scope:** a demonstration of the principle on small semiprimes, **not** a cryptographic
  attack — it scales exponentially, which is exactly why RSA is safe (`figures/phase11_factoring.png`).

### Phase 12 — Universal computation as a ground state ✅  *(see [results](results/PHASE12-results.md))*
- **Understood:** *matter computing any function* — logic gates synthesised as QUBO penalties and
  composed by sharing wires; AND/OR/NOT are complete, so any Boolean circuit is a ground state.
- **Built:** `drift/circuits.py` — primitive gates from `inverse_logic.synthesize`, a `Circuit`
  that sums gate penalties over named wires (`add`, `add_xor` by composition), and `evaluate`
  that clamps inputs and reads the output from DRIFT's exact ground state. `full_adder` wires
  (a, b, cin) → (sum, cout).
- **Validated:** the full adder computes all 8 inputs correctly (penalty 0); XOR by composition;
  a forced-wrong output costs energy; a 2-bit ripple adder exceeds the engine (honest wall).
  `tests/test_circuits.py`, 5/5. **Figure:** `figures/phase12_universal.png`.
- **Honest scope:** the principle is universal; the demonstration is a 1-bit adder because the
  state space is exponential in the variable count — the computronium thesis, measured, not oversold.

### Phase 13 — The tensor-network ground state ✅  *(see [results](results/PHASE13-results.md))*
- **Understood:** *the microscope's lens is finally the engine.* Through Phase 12 χ was only ever
  *measured*, after an exact 2ⁿ diagonalization (Phase 3). Here an MPS solver finds the ground
  state itself by imaginary-time TEBD, and the same χ that was the thermometer becomes the
  compute budget — the truncation *is* the physics. This delivers the README's "read with tensor
  networks" thesis, at last.
- **Built:** `drift/mps.py` — imaginary-time TEBD on a matrix-product state (nearest-neighbor
  two-site gates, SVD truncation to χ, robust center-moving canonical form, no Λ⁻¹). χ reported
  with Phase 3's `effective_chi` ruler. `experiments/phase13_tensor.py`, `tests/test_mps.py`.
- **Validated:** vs exact Lanczos across the phase diagram, worst error **5.8e-5** (target 1e-3);
  energy is a variational upper bound; **reproduces Phase 3's χ peak independently** (Γ≈0.77, χ=6
  at finite size); and runs **past the exact wall** — n=48 (2⁴⁸≈2.8e14 states) with E/n → −4/π as
  n grows. `tests/test_mps.py`, 7/7. **Figure:** `figures/phase13_tensor.png`.
- **Honest scope:** the CPU 1-D **reference** solver — exact while χ ≤ chi_max, degrading
  *measurably* (χ pinned, entropy rising) at criticality where entanglement outgrows the budget.
  Higher-D / large-χ / GPU is the still-deferred Rust/CUDA story; the thesis is no longer deferred.

### Phase 14 — The GPU Ising engine (parallel tempering) ✅  *(built + benchmarked; [results](results/PHASE14-results.md), [ADR-0004](ADR-0004-gpu-ising-engine.md))*
- **Understood:** *the substrate, run at scale.* Scales the **optimization face** (MaxCut, factoring,
  circuits, crystals, Hopfield — any `IsingModel`) from the ~22-spin exact wall to thousands of spins
  by **parallel tempering** (replica-exchange Metropolis) on the GPU.
- **Built:** `drift/solvers/parallel_tempering.py` (CPU reference + oracle), `cuda/ising_pt.cu`
  (one block per replica, local-field maintenance, coalesced updates via J-symmetry, temperature-swap
  exchange, cuRAND; `nvcc -arch=sm_120`), `drift/gpu.py` (drop-in glue), `experiments/phase14_gpu.py`.
- **Validated (CPU, 5/5; full suite 63/63):** finds the **exact** ground energy on MaxCut, a ±J spin
  glass, and a ferromagnet; replica exchange beats a lone cold walker; healthy ladder.
- **GPU built + benchmarked (RTX 5060 Ti, sm_120):** acceptance test **passes** (GPU reproduces the
  exact −22 / −33 energies); scales to n=2048.
- **Phase 14b done — sparse-J (CSR) + occupancy:** each flip is now O(degree) not O(n) (throughput
  flat in n, ~30 % faster), and R=32→128 (a finer PT ladder that also fills the SMs) adds ~4×.
  **~0.05 → ~0.23 Gflips/s, ~4–5× over the first build**, same exact energies. Measured occupancy
  curve confirmed 32 blocks left the GPU ~4× idle.
- **Phase 14c done — checkerboard / graph-colouring:** greedily colour the graph into independent
  sets and flip a whole colour in parallel (neighbours are other colours, so stable) — a sweep is k
  colour-steps, not n serial flips. The serial-flip latency wall is gone; throughput now **rises with
  n** and reaches **n=8192**. **~0.93 Gflips/s @ n=2048 — ~21× the first build, ~5× less wall time**,
  exact −22/−33 preserved. Full arc measured in [results](results/PHASE14-results.md).
- **Phase 14d done — warp per replica:** one warp (32 lanes) per replica instead of a 256-thread
  block, so the colour barrier is a near-free `__syncwarp` and the energy reduction is a shuffle, and
  many more replicas run at once. **Stable ~0.94 Gflips/s @ n=2048, R=256** (an earlier single-run
  1.03 was a boost-clock outlier — corrected). Exact −22/−33 preserved. Honest trade: worse than 14c
  at low R (under-fills the GPU), better in the many-replica regime PT wants. Net arc **~0.044 →
  ~0.94 Gflips/s (~21×)**.
- **Also done (neutral):** energy tracked **incrementally** (Σ dE telescopes to the exact change),
  removing the per-round O(nnz) recompute — verified exact (no drift), but throughput unchanged: the
  kernel is **memory-bound on the scattered CSR neighbour reads**, not the energy.
- **Next (the real lever):** **coalesce / cache the neighbour reads** (`neigh_sum` gathers colIdx /
  weight / spins uncoalesced) — stage the replica's spins in shared memory or reorder for locality.
  Then multi-GPU; route `factor()` / large MaxCut through the engine.
- **Honest scope:** strong minima, not certified optima (the exact engine stays the oracle on small n).

---

## Out of scope (on purpose)
- Beating quantum-annealing or DMRG SOTA — DRIFT is a microscope, not a competitor.
- Claims about consciousness, real nanotech, or imminent grey goo — see CONCEPTS honesty tags.
- Large-scale GPU solving — deferred to a Rust/CUDA port if and when a phase needs it.
