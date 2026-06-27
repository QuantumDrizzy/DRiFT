# Phase 11 — integer factorization as an Ising ground state

**Understood:** *the boldest "matter computes" claim, made literal — encode `p · q = N` so that
the **energy minimum is the arithmetic**.* Nothing divides or searches the integers; DRIFT's own
ground-state engine relaxes a spin system and the factors fall out of the lowest-energy bits.
This is the Phase-2 optimization face pointed straight at number theory.

**Built** (`drift/factoring.py`, `experiments/phase11_factoring.py`):

- `factoring_qubo(N, p_bits, q_bits)` — both factors are odd, so `p = 1 + Σ 2ⁱ pᵢ`,
  `q = 1 + Σ 2ʲ qⱼ`; `p·q` is linearised by replacing each product `pᵢ qⱼ` with an auxiliary
  bit `t_{ij}` pinned to the AND of its inputs by the standard penalty
  `pᵢ qⱼ − 2t(pᵢ+qⱼ) + 3t`. The objective `(N − p·q)²` is added on top. The result is a genuine
  QUBO whose ground state sits exactly at `p·q = N`.
- `factor(N)` — builds the QUBO, converts it to an `IsingModel` (`qubo_to_ising`), finds the
  ground state with DRIFT's exact engine (`exact_ground_state`), and decodes the spins into
  `(p, q)`. The reported energy is the true `(N − p·q)² + penalties`, **exactly 0 at a valid
  factorization** and positive otherwise.

**Validated** (`tests/test_factoring.py`, 5/5):

- **The ground state recovers the factors** for `15, 21, 35, 77, 143` (and the experiment goes
  to `221 = 13 × 17`) — non-trivial factors, `p · q = N`, **energy = 0** in every case.
- The QUBO's own global minimum (brute force over `xᵀQx`) decodes to the same factorization,
  and DRIFT's Ising path agrees with it — the answer is in the QUBO, not an artefact of the
  solver.
- **The construction is honest about scaling:** a too-large instance *raises* rather than
  pretending — `factor(143)` with default widths needs 23 variables, past the exact engine's 22.

**Figure:** `figures/phase11_factoring.png` — (a) the QUBO variable count grows only ~ (log N)²;
(b) but the search space `2^(#vars)` explodes past the exact engine's 2²² ceiling. The wall is
the whole point.

**Honest scope — this is a demonstration, not a cryptographic attack.** Factoring as a ground
state is exact and elegant on small semiprimes, and it **scales exponentially**: the variable
count and the shrinking spectral gap (Phase 9) are precisely *why* factoring is hard and why RSA
is safe. DRIFT factors `221` because `221` is tiny; it cannot touch a 2048-bit modulus, and the
figure shows exactly where the wall is. What the phase shows is the *principle* — that arithmetic
can be the ground state of matter — measured and bounded, not oversold.
