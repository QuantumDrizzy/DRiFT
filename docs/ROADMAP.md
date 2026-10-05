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
  `drift.solve` → decode). Energy is exactly 0 at a valid factorization.
- **Validated:** DRIFT's engine factors `15, 35, 143, … 221 = 13×17` (energy 0); the QUBO's own
  brute-force minimum agrees. Past exact reach, `factor()` goes through `drift.solve` (GPU-PT
  then CPU-PT) and returns `certified=False` rather than raising or pretending optimality.
  `tests/test_factoring.py`.
- **Honest scope:** a demonstration of the principle on small semiprimes, **not** a cryptographic
  attack — it scales exponentially, which is exactly why RSA is safe (`figures/phase11_factoring.png`).

### Phase 12 — Universal computation as a ground state ✅  *(see [results](results/PHASE12-results.md))*
- **Understood:** *matter computing any function* — logic gates synthesised as QUBO penalties and
  composed by sharing wires; AND/OR/NOT are complete, so any Boolean circuit is a ground state.
- **Built:** `drift/circuits.py` — primitive gates from `inverse_logic.synthesize`, a `Circuit`
  that sums gate penalties over named wires (`add`, `add_xor` by composition), and `evaluate`
  that clamps inputs and reads the output from `drift.solve`. `full_adder` wires
  (a, b, cin) → (sum, cout).
- **Validated:** the full adder computes all 8 inputs correctly (penalty 0, certified-exact);
  XOR by composition; a forced-wrong output costs energy. Wider circuits use the same solver
  path with `certified=False`. `tests/test_circuits.py`.
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
  Higher-D / large-χ **GPU MPS/TEBD** is still deferred; Phase 14's CUDA engine is Ising parallel
  tempering, not a tensor-network GPU. The "read with tensor networks" thesis is no longer deferred.

### Phase 14 — The GPU Ising engine (parallel tempering) ✅  *(built + benchmarked; [results](results/PHASE14-results.md), [ADR-0004](ADR-0004-gpu-ising-engine.md))*
- **Understood:** *the substrate, run at scale.* Scales the **optimization face** (MaxCut, factoring,
  circuits, crystals, Hopfield — any `IsingModel`) from the ~22-spin exact wall to thousands of spins
  by **parallel tempering** (replica-exchange Metropolis) on the GPU.
- **Built:** `drift/solvers/parallel_tempering.py` (CPU reference + oracle), `cuda/ising_pt.cu`
  (one block per replica, local-field maintenance, coalesced updates via J-symmetry, temperature-swap
  exchange, cuRAND; `nvcc -arch=sm_120`), `drift/gpu.py` (drop-in glue), `experiments/phase14_gpu.py`.
- **Validated (CPU PT, 5/5):** finds the **exact** ground energy on MaxCut, a ±J spin
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
- **Phase 14e done — shared-memory staging:** each warp stages its replica's spins into shared memory
  (int8 ±1) once per round, so the hot scattered read `sr[colIdx[t]]` becomes a shared access
  (dynamic shared, `MaxDynamicSharedMemorySize` opt-in past 48 KB → still reaches n=8192). **~0.94 →
  ~1.29 Gflips/s @ n=2048 (~1.37×), peak ~1.36 @ n=8192**, exact −22/−33 preserved. Vindicated the
  memory-bound diagnosis. **Net arc: ~0.044 → ~1.29 Gflips/s (~29×).**
- **Unified solver — `drift.solve`:** one `solve(model)` picks the best method by size/hardware —
  **exact (certified)** for n ≤ exact_max, the **GPU-PT engine** for large n, **CPU-PT** as fallback —
  and reports which ran and whether it is **certified** (a heuristic minimum is never passed off as
  the proven optimum). `tests/test_solve.py`. So every face gets exact-when-it-can, scale-when-it-
  must, from one call.
- **Faces share `drift.solve`:** `factor()`, `Circuit.evaluate`, and `minimise_qubo` route
  through the dispatcher. Small n stays certified-exact; past exact reach they return `method`
  + `certified=False` rather than raising or pretending optimality. `require_certified=True`
  restores the old fail-loud wall.
- **Solution quality validated:** fast ≠ good, so checked against a **known optimum at scale** — a
  bipartite graph's max cut is every edge, and the engine recovers it **exactly (ratio 1.0000)** at
  n = 128…1024. Certified two ways now: exact cross-check (n≤18) + known bipartite optimum (n≤1024).
- **Honest scope:** on arbitrary frustrated instances at scale there's no oracle, so those minima are
  strong-not-certified (standard for any heuristic) — stated plainly.

### Phase 15 — The drawing is the function ✅  *(see [results](results/PHASE15-results.md))*
- **Understood:** *Phase 12's coupling graph is a drawing that computes.* One deleted line breaks
  it; a look-alike with every graph statistic matched computes nothing; and the drawing only
  computes while its coupling beats kT.
- **Built:** `drift/drawing.py` — degeneracy-aware exhaustive scorer, unclamped ground set,
  Maslov–Sneppen rewiring, Boltzmann P(correct)/ΔS/heat bill. `tests/test_drawing.py`.
- **Measured:** rse-hpc-lab exercise 11 (ADR-004, predictions pre-registered).

### Phase 16 — Graphity: geometry from a graph Hamiltonian? ✅  *(see [results](results/PHASE16-results.md))*
- **Understood:** *cooling a complete graph under local terms gives an expander, not a lattice;*
  the lower states are disjoint dense blocks the anneal never reaches. Geometry has to be put in.
- **Built:** `drift/graphity.py` — bitset graphs, exact cycle counts and flip deltas, edge-flip
  Metropolis, random-regular null. `tests/test_graphity.py`, including the quadratic-valence
  design error as a regression.
- **Measured:** rse-hpc-lab exercise 12 (ADR-005); 4 of 7 predictions missed, on the record.

### Phase 17 — Scrambling ✅  *(see [results](results/PHASE17-results.md))*
- **Understood:** *level statistics separate chaos from integrability; the OTOC light cone and a
  decaying OTOC do not.*
- **Built:** `drift/scrambling.py` — mixed-field Ising, exact reflection sector, spacing ratio,
  exact OTOC. `tests/test_scrambling.py`.
- **Measured:** rse-hpc-lab exercise 13 (ADR-005).

### Phase 18 — The free-fermion oracle and entanglement scaling ✅  *(see [results](results/PHASE18-results.md))*
- **Understood:** *the critical chain's entanglement grows like (c/6)·log with c = ½* — the scaling
  MERA is built to reproduce. And **`drift.mps` under-converges at criticality with its default
  stopping rule** (c = 0.28 at n = 64; 0.515 when run to convergence).
- **Built:** `drift/freefermion.py` — exact Majorana solution of the open TFIM, block entropies,
  central-charge fit; the oracle for `drift.quantum` and `drift.mps`. `tests/test_freefermion.py`.
- **Measured:** rse-hpc-lab exercise 14 (ADR-005).
- **Next:** a stopping rule for `drift.mps.ground_state` that critical slowing down cannot fool,
  and a re-check of Phase 13's χ curve against this oracle.

### Phase 19 — Random tensor networks: the boundary reads the bulk ✅  *(see [results](results/PHASE19-results.md))*
- **Understood:** *the averaged Rényi-2 entropy of a random tensor network is an Ising free energy,
  and at large D it is the minimal cut* -- exactly, with **−log(number of minimal cuts)** as the
  correction (g = 2, 7, 23 reproduced to 3·10⁻⁴). The minimal cut reads the bulk: logarithmic on a
  hyperbolic disk (+2 per doubling), linear then turning over on a flat grid (the Manhattan geodesic),
  a pure volume law on a random graph. At D = 2 the ratio-of-averages map is 51 % off the true
  averaged entropy; at D = 3, 2.8 %.
- **Built:** `drift/holography.py` -- boundary graphs, the map onto `IsingModel`, exact Z, min cut by
  max-flow, direct contraction of Gaussian random tensors. `tests/test_holography.py`, 12/12.
- **Measured:** rse-hpc-lab exercise 22 (ADR-006); QuBLAR finds the D = ∞ wall on 78/78 large cases.
- **Next (done, phase 20):** phase 2 -- reconstruct the bulk graph from boundary entropies alone (open research).

### Phase 20 — The bulk from the boundary ✅  *(see [results](results/PHASE20-results.md))*
- **Understood:** *the all-interval entropy table, alone, separates the three bulks.* Its circular
  splits (conditional mutual informations, all ≥ 0) rebuild it exactly and are sparse: a random bulk
  leaves only diametric splits (a cycle, δ/diam ½, "nothing inside"); a flat grid one family of large
  nested splits; a hyperbolic disk a comb at five scales whose count halves per doubling (δ/diam 0.167,
  a 1/ℓ² density read post-hoc). δ-hyperbolic is **not** tree-like: the finest hyperbolic splits cross
  (treeness 0.167), as geodesics at every centre do.
- **Built:** `drift/holography.py` -- interval table, split weights and rebuild, four-point Gromov δ,
  kinematic density, the heaviest non-crossing family by exact O(n³) DP, nesting depth.
  `tests/test_holography.py`, 27/27; 7 mutants caught.
- **Measured:** rse-hpc-lab exercise 23 (ADR-007): P1, P2, P5 pass; P3's hyperbolic slope and P4's
  hyperbolic treeness and depth fail, kept.
- **Next:** finite D (the α blur), a {p,q} tiling instead of a ring lattice, and an estimator for
  kinematic density on a discrete hierarchy -- each pre-registered before it is run.

### Phase 21 — A bulk event seen from the boundary ✅  *(see [results](results/PHASE21-results.md))*
- **Understood:** *a change of one bulk bond is placed by the boundary only where geodesics are
  unique.* On the hyperbolic disk every interior bond (264, rings 1-5) is located exactly; the centre
  (twin geodesics around the core) and the rim (leg cuts tie with bulk cuts up to 8 legs) are seen but
  not placed: 47.4 % overall. A flat grid gives the row, never the column. A random bulk is unseen
  (98.8 %). The interval needed to see a ring halves per ring outward, to the leg floor.
- **Built:** `drift/holography.py` -- `cut_membership` (some / every minimal cut, Picard-Queyranne on
  one residual graph), `bond_edges`, `ring_of`. `tests/test_holography.py`, 35/35.
- **Measured:** rse-hpc-lab exercise 24 (ADR-008): Q1, Q2, Q4, Q5 pass; Q3 hyperbolic 0.474 < 0.5, kept.
- **Next (done, phase 22):** a {p,q} tiling to test whether the rim blindness is the leg regime or
  the lattice -- it was the lattice. Two simultaneous events remain open.

### Phase 22 — Regular hyperbolic tilings ✅  *(see [results](results/PHASE22-results.md))*
- **Understood:** *on a regular {p,q} tiling the boundary locates every bulk bond.* On {5,4}, {4,5}
  and {7,3} all 1107 bonds, centre and rim included, are on every minimal cut of some interval with a
  signature no other bond shares. Phase 21's blind centre and rim were the ring lattice's, not
  hyperbolic space's. The parity argument (ties need even p) was wrong and is refuted. The interval
  needed to see one layer deeper grows by the tiling's own growth rate (0.85-0.97 of ln λ).
- **Built:** `drift/holography.py` -- `pq_tiling` (Poincaré-disk reflections, dedup by centre, legs by
  angle), `hyperbolic_distance`, `hyperbolic_angle`. `tests/test_holography.py`, 43/43; 5/5 mutants.
- **Measured:** rse-hpc-lab exercise 25 (ADR-009): T1, T5, T6, T7 pass; T3/T4's {4,5} and T2's {7,3}
  miss, kept.
- **Next:** two simultaneous events (done, phase 23), finite D, and larger cut-offs.

### Phase 23 — Two bulk events at once ✅  *(see [results](results/PHASE23-results.md))*
- **Understood:** *two simultaneous bulk changes are both placed by the boundary* (12 890 / 12 890
  pairs on {5,4} and {4,5}). Their responses mostly add; where they do not, the bonds are close
  (≥ 97.7 % within two steps). The commonest interaction is a cover: degenerate geodesics through
  each bond, so the boundary sees the pair where it sees neither event alone. The even tiling never
  shows a detour.
- **Built:** `drift/holography.py` -- `strengthened`, `pair_response` (exact pruning: one max-flow
  only where both bonds lie on some minimal cut), `bond_distance`. `tests/test_holography.py`, 50/50.
- **Measured:** rse-hpc-lab exercise 26 (ADR-010): V1, V2, V3, V5, V6 pass; V4 (detours outnumber
  covers) fails on both tilings, kept.
- **Next:** why {4,5} has no detours; events of unknown number; finite D.

### Scale path — versioned instance bank + measurable sweeps ✅  *(see [results](results/SCALE-sweep.md), [tech report](TECH-REPORT.md))*
- **Question (falsifiable):** *How do wall-clock time and solution quality (and χ where an MPS/tensor
  path applies) scale with system size n for fixed instance families, when solving via `drift.solve`
  (exact → GPU-PT → CPU-PT)?*
- **Built:** `drift/benchmarks/` — v1 catalog (`instances/manifest.json`) with stable IDs and seeded
  generators for MaxCut Erdős–Rényi, ±J spin glass, bipartite MaxCut, ferro chain, crystal size
  ladder, and a TFIM chain for χ; `experiments.scale_sweep` records n, method, certified, energy,
  wall time, and χ (MPS only, n≤16 on CPU / n≤24 local). GPU binary optional (falls through to CPU-PT;
  gpu-pt rows are never invented).
- **Ladder (wider than the first scale-path PR):** even n = 8…40 (added 22, 28, 36, 40), extra seeds
  on ER / ±J / bipartite at cheap n (and one extra seed at the n=20 dispatcher wall), denser crystal
  (4×4 / 6×4 / 4×8 / 10×4) and TFIM (n=8,10,12,14,16) χ rungs. `ferro-chain` is seed-invariant and
  is not duplicated. `--ci` stays n≤10 and does not clobber `docs/results/scale_sweep.*`.
- **Validated:** bank loads and fingerprints match generators; tiny n≤10 sweep finishes; **certified
  is True only for exact**; published markdown reports `gpu-pt rows: none` on CPU
  (`tests/test_scale_sweep.py`).
- **Figure:** `figures/scale/scale_sweep.png` — n vs time, n vs energy error, method vs n, χ vs n.
- **Regenerate:** `python -m experiments.scale_sweep --profile local --write-report` (CPU) or
  `--profile gpu` (records gpu-pt only if `cuda/ising_pt` actually ran).
- **Next:** still later — multi-GPU; bit-pack spins / reduce shared-bank conflicts; GPU MPS/TEBD;
  more bank families (Hopfield / tiles / circuits) on the same dispatcher.

---

## Out of scope (on purpose)
- Beating quantum-annealing or DMRG SOTA — DRIFT is a microscope, not a competitor.
- Claims about consciousness, real nanotech, or imminent grey goo — see CONCEPTS honesty tags.
- Multi-GPU and GPU MPS/TEBD (higher-D / large-χ) — still later work. Phase 14 landed
  **single-GPU Ising parallel tempering**
  (local Windows/sm_120 binary, not CI); that is not the same as those deferred items.
