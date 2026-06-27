"""
drift.factoring — Phase 11: integer factorization as an Ising ground state.
===========================================================================
The boldest "matter computes" demonstration in the lab: encode the constraint **p · q = N**
as a QUBO whose **ground state reveals the factors**. Nothing searches or divides — the energy
minimum *is* the arithmetic. We build the QUBO, hand it to DRIFT's own Ising ground-state
engine (`exact_ground_state`), and read the factors straight off the lowest-energy spins.

How it is built (a textbook multiplication encoding):

- Both factors are odd, so write ``p = 1 + Σ_{i≥1} 2ⁱ pᵢ`` and ``q = 1 + Σ_{j≥1} 2ʲ qⱼ`` with
  binary variables ``pᵢ, qⱼ``.
- ``p · q`` is then linear in ``{pᵢ, qⱼ}`` *and* their products ``pᵢ qⱼ``. Each product is
  replaced by an auxiliary bit ``t_{ij}`` pinned to the AND of its inputs by a penalty
  ``pᵢ qⱼ − 2 t (pᵢ + qⱼ) + 3 t`` (zero iff ``t = pᵢ ∧ qⱼ``). The whole thing is now quadratic.
- The objective ``(N − p·q)²`` is added on top. Its minimum (zero) sits exactly at ``p·q = N``;
  the penalty weight is large enough that the ground state always honours the AND constraints.

**Honest scope:** this *scales exponentially* — the variable count and the shrinking spectral
gap are precisely why factoring stays hard, and why RSA is safe. This is a demonstration of the
principle on small semiprimes, **not** a cryptographic attack. It is the optimization face of
DRIFT (Phase 2) pointed at arithmetic: matter computing a product by relaxing to its minimum.
"""

from __future__ import annotations

import numpy as np

from .builders.qubo import qubo_to_ising
from .solvers.exact import exact_ground_state


def _layout(p_bits: int, q_bits: int) -> tuple[dict, list]:
    """Assign QUBO variable indices: the free bits p₁…p_{p_bits-1}, q₁…q_{q_bits-1}, then one
    auxiliary t_{ij} per product. Returns (index map, list of (i, j) product pairs)."""
    idx: dict = {}
    for i in range(1, p_bits):
        idx[("p", i)] = len(idx)
    for j in range(1, q_bits):
        idx[("q", j)] = len(idx)
    pairs = [(i, j) for i in range(1, p_bits) for j in range(1, q_bits)]
    for (i, j) in pairs:
        idx[("t", i, j)] = len(idx)
    return idx, pairs


def factoring_qubo(N: int, p_bits: int, q_bits: int, penalty: float | None = None) -> dict:
    """Build the factorization QUBO for a semiprime ``N``. Returns a dict with the dense matrix
    ``Q`` (``min_{x∈{0,1}ⁿ} xᵀ Q x``), the variable index map, the product pairs, and the bit
    widths. The default penalty (> N²) guarantees the ground state honours the AND gadgets."""
    if penalty is None:
        penalty = float(N * N + 1)
    idx, pairs = _layout(p_bits, q_bits)
    n = len(idx)
    Q = np.zeros((n, n))

    def add_linear(v: int, c: float) -> None:
        Q[v, v] += c

    def add_quad(u: int, v: int, c: float) -> None:
        a, b = (u, v) if u < v else (v, u)
        Q[a, b] += c  # upper-triangular bilinear term

    # p·q = 1 + Σ_i 2^i p_i + Σ_j 2^j q_j + Σ_{ij} 2^{i+j} t_{ij}  ==  const0 + Σ_v coeff[v]·x_v
    const0 = 1.0
    coeff = np.zeros(n)
    for i in range(1, p_bits):
        coeff[idx[("p", i)]] += 2.0 ** i
    for j in range(1, q_bits):
        coeff[idx[("q", j)]] += 2.0 ** j
    for (i, j) in pairs:
        coeff[idx[("t", i, j)]] += 2.0 ** (i + j)

    # Objective (N - p·q)^2 = (K - Σ coeff·x)^2,  K = N - const0
    K = float(N) - const0
    # constant K^2 is dropped (does not affect argmin); linear: x_v·(coeff_v^2 - 2 K coeff_v)
    for v in range(n):
        add_linear(v, coeff[v] ** 2 - 2.0 * K * coeff[v])
    # quadratic: 2·coeff_u·coeff_v·x_u x_v  for u<v
    for u in range(n):
        for v in range(u + 1, n):
            add_quad(u, v, 2.0 * coeff[u] * coeff[v])

    # AND penalty per product:  penalty·( p_i q_j − 2 t (p_i + q_j) + 3 t )
    for (i, j) in pairs:
        pi, qj, t = idx[("p", i)], idx[("q", j)], idx[("t", i, j)]
        add_quad(pi, qj, penalty * 1.0)
        add_quad(pi, t, -2.0 * penalty)
        add_quad(qj, t, -2.0 * penalty)
        add_linear(t, 3.0 * penalty)

    # K² is the constant dropped from (N − p·q)² above; tracked so the reported energy is the
    # true objective, which is exactly 0 at a valid factorization and positive otherwise.
    return {"Q": Q, "idx": idx, "pairs": pairs, "p_bits": p_bits, "q_bits": q_bits,
            "N": N, "const": K * K}


def decode(spec: dict, x: np.ndarray) -> tuple[int, int]:
    """Decode a 0/1 assignment ``x`` (one entry per QUBO variable) into the integers (p, q)."""
    idx, p_bits, q_bits = spec["idx"], spec["p_bits"], spec["q_bits"]
    p = 1 + sum((1 << i) * int(x[idx[("p", i)]]) for i in range(1, p_bits))
    q = 1 + sum((1 << j) * int(x[idx[("q", j)]]) for j in range(1, q_bits))
    return p, q


def factor(N: int, p_bits: int | None = None, q_bits: int | None = None) -> dict:
    """Factor a semiprime ``N`` by finding the ground state of its factorization QUBO with
    DRIFT's exact Ising engine. Returns the recovered (p, q), whether p·q == N, and the QUBO
    ground-state energy (≈ 0 when every constraint is satisfied)."""
    import math

    if p_bits is None:  # the smaller factor is ≤ √N
        p_bits = max(2, math.isqrt(N).bit_length())
    if q_bits is None:  # the larger factor is ≤ N/3 (smallest odd prime factor ≥ 3)
        q_bits = max(2, (N // 3).bit_length())
    nvars = (p_bits - 1) + (q_bits - 1) + (p_bits - 1) * (q_bits - 1)
    if nvars > 22:
        raise ValueError(
            f"N={N} needs {nvars} QUBO variables (> 22 for the exact engine). That blow-up "
            "*is* the point — factoring scales exponentially. Pass tighter p_bits/q_bits, or "
            "use a bigger solver."
        )
    spec = factoring_qubo(N, p_bits, q_bits)
    model, offset = qubo_to_ising(spec["Q"])
    s_min, e_min, _ = exact_ground_state(model)
    x = ((s_min + 1) // 2).astype(int)  # spins ±1 → bits 0/1
    p, q = decode(spec, x)
    energy = float(e_min + offset + spec["const"])  # true (N−p·q)²+penalties; 0 iff valid
    return {"N": N, "p": p, "q": q, "ok": p * q == N, "energy": energy}
