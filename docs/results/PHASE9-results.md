# Phase 9 — the optimization face, run quantum (adiabatic annealing)

**Understood:** *the same Ising ground state Phase 2 found by thermal annealing can be reached
by quantum adiabatic evolution — and the cost of doing so is set by the spectral gap, not by
cleverness.* Quantum annealing is not magic: when the gap closes, it fails too.

**Built** (`drift/anneal.py`, `experiments/phase9_quantum_anneal.py`):

- `H(s) = (1 − s)·H_driver + s·H_problem`, swept `s : 0 → 1`.
  - `H_driver = −Σ_i X_i` (transverse field); its ground state is the uniform superposition
    |+…+⟩ (`driver_ground_state`), the quantum "start everywhere at once".
  - `H_problem` is **diagonal in the computational basis**, with `diag[b]` set straight from
    `IsingModel.energy` of configuration `b` — so the quantum problem *is* the classical one,
    with no coupling/sign convention able to drift between them.
- `spectral_gap_path` — the two lowest eigenvalues along the schedule and the minimum gap
  `Δ_min` (dense for small dim, Lanczos otherwise).
- `quantum_anneal` — real-time evolution of |+…+⟩ under `H(t/T)` via a piecewise-constant
  midpoint schedule and a Krylov matrix exponential (`expm_multiply`).
- `success_probability` / `expected_energy` — projection onto the (diagonal) ground subspace.

**Validated** (`tests/test_anneal.py`, 5/5):

- The driver's ground state is the uniform superposition, energy `−n` (exact).
- `H_problem`'s diagonal equals the classical Ising energy of every configuration, and its
  minimum equals the brute-force ground-state energy.
- A **slow anneal solves it**: on a ferromagnetic chain in a strong field (unique ground
  state), `T = 50` reaches the true ground state with **success ≈ 1.00**, and the most
  probable configuration is the classical ground state.
- A **sudden quench does not**: `T = 0.4` stays near the uniform start (**success < 0.01**) —
  adiabaticity is doing the work, not the final Hamiltonian alone.
- **A smaller gap is genuinely harder** (the honest limit): shrinking the symmetry-breaking
  field from `h = 0.40` to `h = 0.04` leaves two nearly-degenerate minima behind a wide
  tunnelling barrier — `Δ_min` falls **0.96 → 0.27**, and at the same anneal time `T = 20`
  the success drops **0.99 → 0.60**. Quantum annealing pays the gap.

**Figure:** `figures/phase9_quantum_anneal.png` — (a) success vs anneal time `T` for the
open-gap and closing-gap problems (both rise toward 1, but the small-gap one lags at every
`T`); (b) the gap `Δ(s)` along the schedule, with the smaller `Δ_min` marked.

**Honest scope:** this is the *mechanism*, simulated exactly on small `n` (state-vector
evolution) — **not** a speed claim over classical annealing. Phase 2's simulated annealing
solves these same small instances too; the point of the microscope is to *see* the quantum
route to the same ground state and to measure the gap that prices it. The `≈ 1/Δ_min²`
adiabatic scaling is shown qualitatively (smaller gap ⇒ more time for equal success), not fit
as a law. The closing-gap problem is the deliberate falsifier: a case where quantum annealing
does **not** win, because the gap is what is hard.
