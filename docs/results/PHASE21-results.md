# Phase 21 — A bulk event seen from the boundary: results

Pre-registered in rse-hpc-lab ADR-008 (`b3e991b`), measured by exercise 24 (`3d2ced3`) on this
module (`f969720`). The full table, with the miss, is in that exercise's `RESULTS.md`; the numbers
(D = ∞, one bond changed at a time, all intervals):

| | hyperbolic (759 bonds) | flat 24 × 24 (1104) | random 3-reg (600) |
|---|---|---|---|
| seen by some interval | 100 % | 100 % | 1.2 % |
| on every minimal cut of some interval | 61 % | 91 % | 0 |
| **located** (certain signature, unique) | **47.4 %** -- all 264 interior bonds; centre and rim not | 0 -- 42 classes of 24: the row, not the column | 0 |
| shortest interval that sees ring r = 1..5 | 54, 28, 14, 8, 8 (halves per ring to the leg floor) | | |
| QuBLAR on 100 doubled-bond instances | 100 / 100 | | |

`cut_membership` is checked against direct perturbation (0 mismatches). Five surviving mutants are
redundant clauses on undirected networks (four proven, one observed); the general clauses are kept.

Known limits: D = ∞, unit bonds, single events, one graph per class, a ring lattice.
