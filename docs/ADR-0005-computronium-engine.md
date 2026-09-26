# ADR-0005: DRiFT is computronium's engine, and the ecosystem's Ising oracle

**Status:** Accepted
**Date:** 2026-09-26
**Deciders:** Antonio

## Context

DRiFT studies one idea: matter computes by minimising energy. Its engine is an
`IsingModel(J, h)` with exact, annealing, parallel-tempering and tensor-network solvers.
It has 96 tests (1 skipped by design) and a CUDA parallel-tempering engine.

On 2026-09-26 the ecosystem took its shape, in pairs:
- SUBSTRATE ↔ QuBLAR (lab ↔ engine);
- **computronium ↔ DRiFT** (field ↔ engine);
- Unibit ↔ LYTH (processor ↔ compiler).

Blaze is the compressor that speaks each engine's format. DRiFT carries the same weight as
QuBLAR, as the engine of its own field.

## Decision

1. **DRiFT is the engine of computronium.** OSCILLON (physical Ising machines) and TRELLIS
   (embedding onto hardware graphs) are graded against DRiFT, as they are today.
2. **DRiFT is the ecosystem's Ising oracle.** Any Ising or QUBO solver elsewhere must match
   DRiFT's exact solver on small shared instances before its numbers are reported:
   - QuBLAR's annealer (its ADR-007 check already follows this rule);
   - a future U-QPU annealing mode in Unibit.
3. **Bridges**, each closed by a gate in the receiving repository:

| Link | What crosses it | Status |
|---|---|---|
| **QuBLAR ↔ DRiFT** | QuBLAR's exported ROI QUBO (h, J), solved by DRiFT's exact and parallel-tempering solvers as an independent cross-check | open; the format is a plain (h, J) |
| **DRiFT → Blaze** | MPS and low-lying states from `drift.mps`, compressed and compared in TT form | open; both sides are MPS |
| **LYTH ↔ DRiFT** | the CUDA parallel-tempering kernel (`cuda/ising_pt.cu`) as a LYTH kernel: movement declared, intensity derived and checked | open; the hot path is already isolated |

4. **Licence unchanged** (MIT). DRiFT stays the open reference that the others check against.

## Consequences

- The README states DRiFT's place in two sentences and links here.
- No code moves. Each bridge is its own change, with its own gate.
