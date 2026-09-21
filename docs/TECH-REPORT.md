# DRIFT as a microscope for physical computation

**Ising ground states as one object, four faces**

Lab note / preprint skeleton · DRiFT (`github.com/QuantumDrizzy/DRiFT`) · 2026  
This note is a map of what the repository actually measures. It is not a claim that
the measurements settle a larger story.

---

## Abstract

DRIFT is a laboratory for *seeing* how a physical system computes by minimizing
energy. One object — the ground state of an Ising Hamiltonian
\(H = -\sum_{ij} J_{ij} s_i s_j - \sum_i h_i s_i\), \(s_i \in \{\pm 1\}\) — is
read as four faces of the same mathematics: combinatorial optimization, tile
self-assembly, crystalline self-replication, and Hopfield memory. The same
engine also supports a dynamical (reservoir) face, an adiabatic quantum path,
and a 1-D tensor-network solver whose bond dimension \(\chi\) is both
thermometer and compute budget.

The software is built as a microscope, not a competitor. Small instances are
solved by exact enumeration and marked `certified=True`. Larger instances fall
through GPU parallel tempering when a local CUDA binary exists, otherwise
CPU-PT, and are marked `certified=False`. \(\chi\) is recorded only where a
tensor-network path actually applies (open 1-D TFIM). This note states that
contract, sketches the architecture, points at measured results that already
live in `docs/results/`, and names the walls.

---

## Honesty contract

Tags follow [`docs/CONCEPTS.md`](CONCEPTS.md): **[ESTABLISHED]**, **[ACTIVE]**,
**[SPECULATIVE]**. DRIFT builds on the first two. The third motivates questions;
it is not a result.

**Established (this lab treats as fact, with method and size stated):**

- Ising model; QUBO \(\leftrightarrow\) Ising; MaxCut as an antiferromagnetic
  Ising; Hopfield networks *are* Ising models (Nobel Physics 2024).
- aTAM / Wang tiles as energy-minimizing assembly; period-4 crystals from
  translation-invariant frustrated couplings.
- Landauer's \(kT\ln 2\); Margolus–Levitin / Lloyd-style physical limits
  (order-of-magnitude roofline, not a precision metrology).
- Bond dimension \(\chi\) as an entanglement / compressibility measure for
  MPS states. Adiabatic theorem: quantum annealing pays the spectral gap.
- Parallel tempering is a heuristic. A heuristic minimum is not a proven
  optimum.

**Speculation (named so it is not smuggled in):**

- Computronium as a *material*. The *limits* it gestures at are established;
  the fabrication is fiction.
- Grey goo / imminent nanobots; consciousness; IIT-\(\Phi\) as experience.
  DRIFT does not measure those and does not claim them.

Every energy, \(\chi\), success probability, and Gflips/s figure in this note
is already written down with a method, a system size, and a figure or table in
`docs/results/`. If a number is not there, it is not here either.

---

## Architecture (one pager)

One engine, pluggable loads ([`ADR-0001`](ADR-0001-architecture.md)):

```
 builders (QUBO, tiles, crystal, Hopfield, circuits, factoring)
                    │
                    ▼
            IsingModel(J, h)
                    │
                    ▼
     drift.solve  ── exact (n ≤ exact_max, default 18)  → certified=True
                 ── GPU-PT  (cuda/ising_pt present)     → certified=False
                 ── CPU-PT  (portable fallback)         → certified=False

     χ path (separate):  drift.mps  TEBD on 1-D TFIM only
```

| Piece | Role |
|-------|------|
| `drift/ising.py` | Shared Hamiltonian; energy and single-spin \(\Delta E\). |
| Builders | Fill \(J,h\). A new face is a new builder, not a new project. |
| `drift.solve` | Dispatcher. Returns `Solution(s, energy, method, certified)`. |
| Exact engine | Brute force. Proven ground state. Dies ~22 spins; default wall is 18. |
| CPU-PT | Replica-exchange Metropolis; oracle for the CUDA kernel. |
| GPU-PT | `cuda/ising_pt.cu`, `nvcc -arch=sm_120`, glue in `drift/gpu.py`. Local Windows/Blackwell binary — **not in CI, never committed**. |
| MPS | Imaginary-time TEBD; \(\chi\) is the truncation. 1-D TFIM only. |
| Instance bank | `drift/benchmarks/` — versioned IDs, seeded generators, n≤8 fixtures. |
| Scale sweep | `python -m experiments.scale_sweep` — wall time, energy error, method, \(\chi\). |

`factor()`, `Circuit.evaluate`, and `minimise_qubo` all go through `drift.solve`.
Pass `require_certified=True` when a proven ground state is required (truth
tables, uniqueness): that restores the exact-engine wall instead of guessing.

**Certified flag.** `certified=True` if and only if `method == "exact"`.
CPU-PT, GPU-PT, and MPS are never certified, even when they match an oracle.
That is the whole honesty of the dispatcher.

---

## Measured results (already in the repo)

Cite the phase notes and figures. Do not treat this section as a new
measurement.

### Four faces, one engine

[`PHASE7-results.md`](results/PHASE7-results.md),
`figures/phase7_four_faces.png`:

| Face | What was run | Reported outcome |
|------|----------------|------------------|
| Optimization | MaxCut, n = 14 | cut = 30 edges, E = −17.0 |
| Self-assembly | Wang tiles, n = 81 | 12/12 bonds |
| Self-replication | crystal, n = 144 | period 4, E = −288 |
| Memory | Hopfield, n = 100 | recovered overlap +1.00 |

Same `IsingModel` and solvers; only \((J,h)\) changed. Supporting phase notes:

- P1 — 2-D ferro \(L=4\): exact = SA = −32 \(= -2nJ\). [`PHASE1-results.md`](results/PHASE1-results.md)
- P2 — \(G(14,0.5)\), 43 edges: exact cut 31, E = −19. [`PHASE2-results.md`](results/PHASE2-results.md)
- P4 — n = 100, cue overlap 0.50 → recall 1.00. [`PHASE4-results.md`](results/PHASE4-results.md)
- P5 — 3×3 plus: 12/12 bonds. [`PHASE5-results.md`](results/PHASE5-results.md)
- P6 — 12×12 crystal: E = −288, period 4. [`PHASE6-results.md`](results/PHASE6-results.md)

Roofline (`figures/phase7_roofline.png`): real hardware sits six or more orders
of magnitude above Landauer \(kT\ln 2\) at 300 K (\(2.87\times 10^{-21}\) J).
Those landmarks are textbook order-of-magnitude, as the phase note says.

### \(\chi\) as thermometer, then as engine

- P3 — TFIM n = 14, exact Lanczos: \(\chi\) 4 (ordered) → 6 (peak at
  \(\Gamma \approx 0.77\)) → 3 (polarized). Finite-size shift from
  \(\Gamma_c = 1\) is stated. [`PHASE3-results.md`](results/PHASE3-results.md),
  `figures/phase3_chi.png`.
- P13 — MPS/TEBD matches Lanczos to \(5.8\times 10^{-5}\); reproduces the
  P3 peak independently at n = 32 (\(\Gamma \approx 0.77\), \(\chi = 6\));
  n = 48 with \(E/n \to -4/\pi\). [`PHASE13-results.md`](results/PHASE13-results.md),
  `figures/phase13_tensor.png`.

\(\chi\) is not a property of a dense MaxCut graph in this codebase. The scale
sweep records it only on `tfim-chain`.

### Dynamics, quantum, arithmetic, universality

- P8 — reservoir MC = 43.5 at N = 200, peaking at \(\rho \approx 1.0\).
  [`PHASE8-results.md`](results/PHASE8-results.md)
- P9 — slow adiabatic anneal success \(\approx 1.00\); quench \(< 0.01\);
  shrinking \(\Delta_{\min}\) 0.96 → 0.27 drops success 0.99 → 0.60 at fixed T.
  [`PHASE9-results.md`](results/PHASE9-results.md)
- P10 — spike landscape: SA success 0.17, QA 0.45 at T = 40; funnel \(\approx 1\)
  for both. [`PHASE10-results.md`](results/PHASE10-results.md)
- P11 — factors \(15,\ldots,221=13\times 17\), energy 0; not an attack.
  [`PHASE11-results.md`](results/PHASE11-results.md)
- P12 — 1-bit full adder, all 8 inputs, penalty 0.
  [`PHASE12-results.md`](results/PHASE12-results.md)

### GPU engine (a different experiment than the scale sweep)

[`PHASE14-results.md`](results/PHASE14-results.md), [`ADR-0004`](ADR-0004-gpu-ising-engine.md),
`figures/phase14_gpu.png`. Measured on an RTX 5060 Ti (sm_120, CUDA 13):

- Acceptance: GPU reproduces exact −22 / −33 on the small MaxCut oracles.
- Throughput arc at n = 2048: ~0.044 → ~1.29 Gflips/s (~29×) through sparse-J,
  occupancy, checkerboard, warp-per-replica, shared-memory staging; n = 8192
  reachable.
- Bipartite MaxCut cut ratio 1.0000 at n = 128…1024 (known optimum = every edge).
- Frustrated instances at scale: strong-not-certified.

That Gflips/s curve is **not** the instance-bank sweep. Do not copy it into
`SCALE-sweep.md` as if `drift.solve` had run gpu-pt on the bank.

### Scale path (`drift.solve` vs n)

[`SCALE-sweep.md`](results/SCALE-sweep.md), `figures/scale/scale_sweep.png`.

Falsifiable question: how do wall-clock time and solution quality (and \(\chi\)
on TFIM) scale with n for fixed families, via `drift.solve`?

The bank is version `v1` with stable IDs (`v1.maxcut-er.n008.s001`, …). Families:
Erdős–Rényi MaxCut, complete ±J glass, bipartite MaxCut (analytic optimum),
open ferro chain (analytic \(E=-(n-1)\)), period-4 crystal (analytic \(E=-2n\)),
TFIM chain (MPS). The ladder is even n from 8 to 40 with extra seeds on the
stochastic families at cheap n. `--ci` stays n≤10 and does not overwrite
published CSV/JSON.

**How to read the published sweep.** The JSON field `gpu_pt_rows` and the banner
in `SCALE-sweep.md` say whether CUDA ran. A CPU-only environment records
`cpu-pt` for n > 18 and **zero gpu-pt rows**. That is not a GPU measurement.
Regenerate with:

```bash
python -m drift.benchmarks                              # after catalog edits
python -m experiments.scale_sweep --profile local --write-report   # CPU
python -m experiments.scale_sweep --profile gpu  --write-report    # gpu-pt iff binary exists
python -m experiments.scale_sweep --ci                             # n≤10 smoke
```

Numbers for the current commit live in that file (tables a–d). This note does
not duplicate them: wall times are machine-dependent, and inventing a gpu-pt
column on a CPU run would be a false claim.

---

## Limits

| Wall | What it is | What it is not |
|------|------------|----------------|
| Exact ~18 | `drift.solve` default `exact_max=18`. Enumeration is still feasible a few spins past that (~22); 2²⁴ is not precomputed in the bank. | A physics limit. It is an honesty limit: past here we do not stamp `certified`. |
| \(\chi\) scope | Open 1-D TFIM, CPU TEBD, sweep cutoff n≤16 (local n≤24; P13 showed 48). | A \(\chi\) for MaxCut, spin glasses, or 2-D crystals. Those graphs do not go through `drift.mps`. |
| Windows / sm_120 GPU | CUDA engine compiled `nvcc -arch=sm_120`, local RTX 50-class binary. CI is CPU-only; the binary is gitignored. | A portable GPU story. Other GPUs need a different `-arch`. Absent binary ⇒ CPU-PT, labeled as such. |
| PT quality | Matches oracles where they exist (analytic; exact enum n≤20; P14 bipartite n≤1024). | A certificate on arbitrary frustrated instances at n≥24. |
| MPS schedule in the sweep | Short imaginary-time ladder, \(\chi_{\max}=16\). | A replacement for the P13 high-precision curve. |
| Roofline | Order-of-magnitude landmarks. | A calorimeter. |
| Scope of the project | Microscope for established mappings. | SOTA solver; quantum supremacy; consciousness; grey goo. |

---

## Outlook

Gradual, measurable, same honesty contract:

1. **Wider n on the same IDs.** The bank now reaches n = 40 on the classical
   ladder. Next cheap steps are more seeds at PT rungs that still have an
   oracle (n = 20), then n = 48 / 64 on analytic families only, then GPU-PT
   on this bank (same JSON schema, `gpu_pt_rows > 0` only if the kernel ran).
2. **More families in the bank.** Hopfield recall, tile assemblies, and small
   circuits/factoring instances as catalog rows — still through `drift.solve`,
   still with an honest `certified` bit.
3. **Do not blur engines.** GPU Ising-PT (P14) is not GPU MPS/TEBD. Higher-D /
   large-\(\chi\) tensor networks remain deferred. Multi-GPU is deferred.
4. **Keep CI small.** `--ci` is a smoke test. Published curves are `--profile
   local` (CPU) or `--profile gpu` (binary present). Never rename cpu-pt to
   gpu-pt.

The thesis that can be defended from this repository is narrow and, we think,
true: **one Ising ground-state object, four faces, a dispatcher that says
which solver ran, and a \(\chi\) that is only claimed where a tensor network
actually ran.** Everything else is either already tagged speculation or future
work.

---

## References (in-repo)

- [`README.md`](../README.md) — thesis, honesty contract, figures.
- [`CONCEPTS.md`](CONCEPTS.md) — glossary with ESTABLISHED / ACTIVE / SPECULATIVE tags.
- [`ADR-0001-architecture.md`](ADR-0001-architecture.md), [`ADR-0004-gpu-ising-engine.md`](ADR-0004-gpu-ising-engine.md).
- [`ROADMAP.md`](ROADMAP.md) — phases P0–P14 and the scale path.
- [`results/`](results/) — `PHASE{1..14}-results.md`, `SCALE-sweep.md`.
