# Phase 22 — Regular hyperbolic tilings: results

Pre-registered in rse-hpc-lab ADR-009 (`b9d3e0b`), measured by exercise 25 (`a70978a`) on this
module (`434da6a`). The full table, with the misses, is in that exercise's `RESULTS.md`; the numbers
(D = ∞, every bond × every interval):

| | {5,4} · 4 layers | {4,5} · 5 layers | {7,3} · 4 layers | ring lattice (phase 21) |
|---|---|---|---|---|
| tiles / bonds / legs | 166 / 225 / 380 | 257 / 336 / 356 | 232 / 546 / 532 | 381 / 759 / 192 |
| bonds **located** by the boundary | **100 %** | **100 %** | **100 %** | 47.4 % |
| centre / rim located | 100 % / 100 % | 100 % / 100 % | 100 % / 100 % | 0 / ≤ 25 % |
| leg regime n* | 2 | 2 | 3 | 8 |
| ℓ_min per layer vs growth λ | ×2.6 vs 2.625 | ×2.1 vs 2.312 | ×2.4 vs 2.625 | ×1.66 vs 2 |
| QuBLAR, 100 doubled-bond instances | 100 / 100 | | | 100 / 100 |

Misses kept: H-parity is refuted (the even {4,5} is as located as the odd tilings), so {4,5} is not
the lowest; {7,3}'s leg regime is 3, not ≤ 2.

Known limits: D = ∞, unit bonds, single events, one cut-off per tiling; which feature of the ring
lattice blinded it is not isolated.
