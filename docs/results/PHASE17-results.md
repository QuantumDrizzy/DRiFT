# Phase 17 — scrambling: what chaos looks like in a spectrum, and what only looks like it

**Understood:** *level statistics tell chaos from integrability; the OTOC light cone does not,
and neither does a decaying OTOC.* What separates them in the OTOC is the late value of an
operator that is local in the integrable model's own quasiparticles.

**Built** (`drift/scrambling.py`, `tests/test_scrambling.py`):

- `mixed_field_ising(L, g, h)` — H = Σ ZᵢZᵢ₊₁ + g Σ Xᵢ + h Σ Zᵢ, sparse, open chain.
  Chaotic at (1.05, 0.5); free fermions at h = 0.
- `reflection_even_basis` — exact projection onto one symmetry sector (isometry and
  no-leak checked), because mixing sectors hides level repulsion.
- `spacing_ratio` — ⟨r⟩ over the central half of the sector's spectrum.
- `otoc` — F(t) = Tr(W(t) V W(t) V)/2ᴸ at infinite temperature, exact, by eigenbasis phases;
  checked against brute-force `expm` evolution to 1e-12.

**Measured** in rse-hpc-lab exercise 13 (ADR-005), predictions committed before the code.
Digits in that repository's `exercises/13-scrambling/RESULTS.md`.

| | chaotic (1.05, 0.5) | free fermions (1.05, 0) |
|---|---|---|
| ⟨r⟩, L = 14, reflection-even | **0.5387** (GOE 0.5307) | **0.4053** (Poisson 0.3863); 0.3408 at L = 12 |
| X–X OTOC, late mean (t 6–10, r = 4, L = 10) | **0.096** | **0.676** — X does not scramble |
| Z–Z OTOC, late mean | 0.025 | **−0.479**, swinging to −0.94 |
| first time F < ½, linear in r | R² 0.99998, v = 1.68 | R² 0.9997, v = 1.78 |

All six predictions held — two of them for the wrong reason, which is the result:

- **A decaying OTOC is not chaos.** On free fermions Z₀(t) carries a Jordan–Wigner string;
  when it reaches site r, Z₀(t) and Z_r nearly *anticommute* and F → −1. Coherent, not
  scrambled (`test_integrable_decay_is_not_chaos`).
- **A light cone is not chaos.** Both chains have the same ballistic front: it is locality
  (Lieb–Robinson), present with or without chaos.

**Honest scope.** L ≤ 14 for spectra and L = 10 for OTOCs, CPU-only exact diagonalisation.
The free-fermion ⟨r⟩ is non-monotonic in L, as expected for a spectrum with that much
structure. The GPU path (sector `eigvalsh` at L = 16) is listed, not built.
