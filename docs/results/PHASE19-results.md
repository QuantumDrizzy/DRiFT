# Phase 19 — Random tensor networks as Ising models: results

Pre-registered in rse-hpc-lab ADR-006 (`09028e8`), measured by exercise 22 (`f5b59d8`) on this
module (`7f69a7f`). The full table, with the misses, is in that exercise's `RESULTS.md`; the numbers:

| | measured |
|---|---|
| RT at D = 10⁴ | S₂ − \|γ\| log D = −log g exactly (g = 2, 7, 23 → −0.6933, −1.9462, −3.1358) |
| exact-size graphs (≤ 21 spins) | every minimal cut is the legs themselves: the bulk is too small to be cheaper |
| hyperbolic, 192 legs | γ(n) = n to 8, then +2 per doubling (log) |
| flat 24 × 24 | γ(n) = n to 35, then falls to 24 (the cut crosses the grid) |
| random 3-regular, 96 legs | γ(n) = n (volume law) |
| finite D, mean deficit vs \|γ\| log D | D = 2: 70.5 %, D = 3: 30.9 %, D = 10: 4.6 %, D = 100: 1.4 % |
| ratio of averages vs true average (4 × 5 grid) | D = 2: up to 51 % off; D = 3: 2.8 % |
| QuBLAR at D = ∞ (189-vertex disk, 16 × 16 grid) | 78/78 intervals exact |

Known limits: exact Z to 21 spins; large graphs are D = ∞; the hyperbolic graph is a ring lattice,
not a {p,q} tiling; one graph for the D = 2/3 contraction, 200 samples each.
