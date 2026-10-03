# Phase 18 — the free-fermion oracle and entanglement scaling

**Understood:** *the critical Ising chain's entanglement grows like (c/6)·log with c = ½,* and
**`drift.mps` under-converges at the critical point with its default stopping rule** — which
only an exact oracle at n = 64 could show.

**Built** (`drift/freefermion.py`, `tests/test_freefermion.py`) — the open TFIM
H = −Σ ZᵢZᵢ₊₁ − g Σ Xᵢ solved exactly by Jordan–Wigner to Majoranas:

- `majorana_matrix`, `ground_state` → (Γ, E₀) with Γ = −i·sign(iA), E₀ = −¼ Σ|eig(iA)|.
- `block_entropies` — S(ℓ) of the first ℓ sites from the 2ℓ × 2ℓ block of Γ, any length.
- `fit_central_charge` — open-boundary Calabrese–Cardy fit over ℓ ∈ [L/8, 7L/8].

It plays for `drift.quantum` and `drift.mps` the role `exact_ground_state` plays for the
classical faces: the oracle.

**Measured** in rse-hpc-lab exercise 14 (ADR-005), predictions committed before the code.
Digits in that repository's `exercises/14-entanglement-scaling/RESULTS.md`.

| | Result |
|---|---|
| oracle vs `drift.quantum` (Lanczos), n = 16 | max \|ΔS\| = 2.5e-13 nats, \|ΔE₀\| ~ 1e-14 |
| c, critical, n = 512 | **0.5022** (n = 64: 0.5169) |
| c, gapped g = 1.5, n = 512 | 0.0000 — area law |
| `drift.mps`, n = 64, χ_max 64, **default schedule** | c = **0.284**, max \|ΔS\| = 0.049, E − E_exact = 1.5e-3, χ kept 50 |
| `drift.mps`, n = 64, `max_sweeps=400, econv=1e-12` | c = **0.5152**, max \|ΔS\| = 3e-4, E − E_exact = 5e-8, χ kept 64 |

**The MPS finding.** The default stopping rule is a per-sweep *relative* energy change. At the
critical point the gap ~ 1/n makes each sweep move the energy very little, so it stops while
the long-wavelength modes are still unprojected, and the state is still too close to its
|+x⟩ seed. Energy barely notices (2.4e-5 per site); the half-chain entropy is 9 % low. χ
stayed below χ_max, so this is convergence, not truncation.

**Consequence for Phase 13.** Its n = 32 χ curve was computed with the default schedule. It
should be re-checked against this oracle before its χ-peak height is quoted again. The energy
table (E/n → −4/π) is not materially affected: the non-convergence is two orders of
magnitude below the open-boundary offset it reports.

**Reading.** c = ½ is the quantitative content MERA is built to reproduce. The "space emerges
from entanglement" reading is interpretation; nothing here measures it.
