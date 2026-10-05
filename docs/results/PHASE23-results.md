# Phase 23 — Two bulk events at once: results

Pre-registered in rse-hpc-lab ADR-010 (`66cf609`), measured by exercise 26 (`ec870f8`) on this
module (`b41ce00`). The full table, with the miss, is in that exercise's `RESULTS.md`; the numbers
(D = ∞, every pair of bonds strengthened together, every interval):

| | {5,4} · 3 layers | {4,5} · 4 layers |
|---|---|---|
| pairs identified (signature unique among pairs and singles) | 3160 / 3160 | 9730 / 9730 |
| interacting cells (both bonds on some minimal cut) | 246 640 | 1 072 586 |
| non-additive share of those | 29.2 % | 5.0 % |
| ... at bond distance ≤ 2 | 97.8 % | 97.7 % |
| sub-additive (detour) / super-additive (cover) | 27 635 / 44 450 | **0** / 53 312 |
| QuBLAR, 100 interacting cells | 100 / 100 | |

Miss kept: covers, not detours, dominate. The boundary often sees a pair on intervals that see
neither event alone. The even tiling has no detours at all `[TO DETERMINE]`.

Known limits: D = ∞, unit bonds, strengthening only, pairs only, 3-4 layers.
