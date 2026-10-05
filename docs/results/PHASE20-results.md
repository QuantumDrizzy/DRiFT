# Phase 20 — The bulk from the boundary: results

Pre-registered in rse-hpc-lab ADR-007 (`bca131a`), measured by exercise 23 (`175fece`) on this
module (`a3cd425`). The full table, with the misses, is in that exercise's `RESULTS.md`; the numbers
(D = ∞, from the all-interval min-cut table only):

| | hyperbolic (192 legs) | flat 24 × 24 (92) | random 3-reg (96) |
|---|---|---|---|
| non-zero circular splits | 192, weight ½, five scales | 46, weight 1, ℓ ≥ 25 | 48, all diametric |
| rebuild of every γ from the splits | exact | exact | exact |
| δ / diameter (four-point, every 4th gap) | **0.167** | 0.455 | 0.500 (a cycle) |
| treeness (heaviest non-crossing family) | 0.167 | 0.478 | 0.021 |
| nesting depth of that family | 4 | 22 | 1 |
| QuBLAR on the 300 heaviest splits vs exact DP | 15.5 / 16 | exact | exact |

Misses kept: the hyperbolic A(ℓ) slope (−1.23, not in [−2.5, −1.5] -- a comb whose cumulative
weight falls as ℓ^−1.01, i.e. a 1/ℓ² density, read post-hoc), treeness 0.167 (not ≥ 0.7: the finest
splits sit at every other gap and cross, as geodesics do in continuous AdS) and depth 4 (not 5-8:
the outer rings are never cheaper than the legs).

Known limits: D = ∞; one graph per class; the hyperbolic graph is a ring lattice; a split network
recovers the boundary metric, not the bulk graph.
