# Phase 13 — the tensor-network ground state (χ as the engine, not just the thermometer)

**Understood:** *the microscope's lens is finally the engine.* Through Phase 12 every ground
state was read one of two ways — brute force over 2ⁿ configurations, or exact Lanczos on the
full 2ⁿ × 2ⁿ Hamiltonian — and both die at n ≈ 14–22, the wall each recent phase confesses.
Yet DRIFT's thesis is that ground states are *read with tensor networks*: in Phase 3 the bond
dimension χ was only ever **measured**, after an exact diagonalization. The lens was never the
microscope. Phase 13 closes that gap: an MPS solver finds the ground state itself, and the
same χ that was our thermometer becomes our compute budget.

**Built** (`drift/mps.py`, `experiments/phase13_tensor.py`, `tests/test_mps.py`):

- **Imaginary-time TEBD.** e^{-τH} projects onto the ground state; the 1-D transverse-field
  Ising Hamiltonian `H = -j Σ Z_iZ_{i+1} - Γ Σ X_i` is split into nearest-neighbor two-site
  bond gates (the single-site X shared ½/½ across an interior site's two bonds), and after each
  gate the MPS is re-compressed by an SVD truncated to bond dimension χ. **That truncation is
  the physics** — χ is exactly how much entanglement, how much computation, the state carries.
- **Robust canonical form.** The orthogonality center is moved by center-shifting SVDs (no Λ⁻¹
  inverses), so the scheme is numerically stable; all arithmetic is real.
- **χ reported with the Phase-3 ruler.** `MpsResult.chi` is `effective_chi` (tol 1e-3) applied to
  the center-bond Schmidt spectrum — the *same* function Phase 3 used on the exact state, now on
  the state the tensor network produced. `chi_kept` is the raw memory cost.

**Validated** (`tests/test_mps.py`, 7/7):

- **Correct, not merely plausible.** Against exact Lanczos across the phase diagram (n = 8…14,
  Γ ∈ {0.5, 1, 2}), the worst energy error is **5.8×10⁻⁵** — target was 1e-3.
- **Variational.** Imaginary time approaches from above; the MPS energy never falls below the
  true ground energy (checked to 1e-6).
- **Reproduces Phase 3's χ curve — independently.** At n = 32 (past the exact wall), the effective
  χ rises into the critical region and collapses in the disordered phase, peaking at **Γ ≈ 0.77**
  with **χ = 6** — the *same* finite-size peak location and height Phase 3 found by exact
  diagonalization at n = 14 (4 in the cat phase → 6 critical → 2 polarized):

  | Γ/J | 0.20 | 0.40 | 0.60 | **0.77** | 0.90 | 1.00 | 1.20 | 1.50 | 2.00 | 3.00 | 5.00 |
  |:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
  | χ_eff | 4 | 4 | 4 | **6** | 6 | 6 | 5 | 4 | 3 | 2 | 2 |

- **Past the wall.** n = 48 is 2⁴⁸ ≈ 2.8×10¹⁴ configurations — untouchable by brute force or
  full-matrix Lanczos. The critical energy density marches monotonically toward the known
  thermodynamic value **−4/π = −1.27324** as n grows:

  | n | 12 | 16 | 24 | 32 | 48 |
  |:---:|:---:|:---:|:---:|:---:|:---:|
  | E/n | −1.24383 | −1.25102 | −1.25831 | −1.26200 | −1.26571 |
  | Δ to −4/π | +0.0294 | +0.0222 | +0.0149 | +0.0112 | +0.0075 |

  Open boundaries miss the end bonds, so they approach the bulk value from above — the longer the
  chain, the closer it sits. Exactly the finite-size scaling the exact engine could never show.

**Figure:** `figures/phase13_tensor.png` — (a) MPS vs exact energies on the y = x line (max error
6e-5); (b) the effective-χ peak at Γ ≈ 0.77 with the entanglement-entropy profile, at n = 32;
(c) the critical energy density vs 1/n converging to −4/π.

**Honest scope.** This is the honest **reference** solver: exact while χ ≤ chi_max and degrading
*measurably* (χ pinned at the cap, rising entropy) when the entanglement outgrows the budget —
so criticality, where entanglement grows with n, is where it costs the most and where a finite χ
eventually caps the accuracy. It is 1-D and CPU; the higher-dimensional / large-χ / GPU story is
the Rust/CUDA port, still deferred on purpose. What is *no longer* deferred is the thesis itself:
DRIFT finally reads a ground state with a tensor network — and the cost of doing so, χ, is the
very quantity the microscope was built to measure.
