# Phase 24 — Decoding an unknown number of bulk events: results

Pre-registered in rse-hpc-lab ADR-011 (`6a0b2f5`), measured by exercise 27 (`20edaff`) on this
module (`5995fe5`). The full table, with the misses, is in that exercise's `RESULTS.md`; the numbers
(D = ∞, 40 random event sets per k, the decoder not told k):

| recovery by k | 1 | 2 | 3 | 4 | 6 |
|---|---|---|---|---|---|
| {5,4} · 3 layers (80 bonds) | 40/40 | 40/40 | 37/40 | 40/40 | 35/40 |
| {4,5} · 4 layers (140 bonds) | 40/40 | 40/40 | 40/40 | 40/40 | 40/40 |

Every failure is an alias (another set explains the data strictly better under the additive model),
and every alias comes from interacting events. There is no solver miss. QuBLAR, as the QUBO referee,
matched or beat DRiFT's annealing on 88.5 % ({5,4}) and 99 % ({4,5}) of instances.

Misses kept: recovery is not monotone in k on {5,4} (a dip at k = 3); QuBLAR ≤ annealing falls short of
95 % on {5,4}; W5 is vacuous on {4,5} (no failures).

Known limits: D = ∞, unit bonds, noiseless data, λ = 1, no optimality certificate.
