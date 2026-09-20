"""Inverse computronium — target truth table -> QUBO penalty (Ising compilation).

`logic.py` goes *forward*: a hand-written QUBO penalty has ground states that spell out
a gate's truth table. This module goes **backward** — the inverse-design problem applied
to computation: given the truth table you *want*, synthesise the QUBO whose ground states
are exactly those rows. It is the computational sibling of inverse materials design
(target property -> structure); here the "structure" is the coupling matrix and the
"property" is the Boolean function.

Method (honest and exact for the representable case):

    A QUBO over n binary vars is the general quadratic  H(x) = Σ a_i x_i + Σ_{i<j} b_ij x_i x_j + c
    (x_i ∈ {0,1}, so x_i² = x_i — no separate diagonal-quadratic term). H is *linear in the
    coefficients* θ = (c, a, b) for any fixed row x, so demanding

        H(x) = 0      on every valid  row  (the target truth table)
        H(x) ≥ gap    on every invalid row

    is a linear feasibility program in θ. If it is feasible, the minimum of H is 0 and is
    attained on exactly the target rows — the forward contract of `logic.py`, recovered.

Two facts kept honest:

  * **The inverse is multi-modal.** Many coupling matrices share one ground-state set, so we
    validate by *ground states*, never by coefficient equality (ADR-0001: report the family,
    not a false-unique answer).
  * **Not every function is 2-local.** XOR has no QUBO over its own variables — the textbook
    non-quadratic case. We *detect* this (NotQuadratic) instead of faking a fit, and resolve
    it the way the substrate already does: by **composition** of synthesised AND/OR/NOT
    penalties over shared wires (universality). General ancilla synthesis (Boros–Hammer
    quadratization) is out of scope.

Closes the cross-repo item of AETHER's inverse-design ADR-0001: a synthesised QUBO is fed
straight to DRIFT's own anneal spine (`solvers.simulated_annealing`) — the annealer relaxes
to a ground state that *is* a valid truth-table row, i.e. matter computing the function.
"""

from __future__ import annotations

import itertools
from typing import Callable, Iterable

import numpy as np
from scipy.optimize import linprog

from .ising import IsingModel
from .solve import Solution, solve


class NotQuadratic(Exception):
    """Raised when no QUBO over the given variables reproduces the target truth table."""


def truth_from_fn(fn: Callable[..., int], n_inputs: int) -> frozenset:
    """Build a truth table {(x_0,…,x_{n-1}, y)} from a Boolean function of `n_inputs` bits."""
    rows = []
    for x in itertools.product((0, 1), repeat=n_inputs):
        rows.append((*x, int(fn(*x)) & 1))
    return frozenset(rows)


def _features(x: tuple[int, ...], n: int) -> np.ndarray:
    """[1, x_0…x_{n-1}, (x_i x_j)_{i<j}] — H(x) = features · θ."""
    pairs = [x[i] * x[j] for i, j in itertools.combinations(range(n), 2)]
    return np.array([1.0, *x, *pairs], dtype=np.float64)


def synthesize(truth_rows: Iterable[tuple[int, ...]], n_vars: int,
               gap: float = 1.0, bound: float = 8.0):
    """Synthesise a QUBO whose ground states are exactly `truth_rows`.

    Returns (Q, offset): Q is an (n×n) upper-triangular matrix (diagonal = linear terms
    a_i, super-diagonal = couplings b_ij), offset = c, so H(x) = x·Q·x + offset for binary x.
    Raises NotQuadratic if no quadratic over these n_vars fits.
    """
    valid = {tuple(r) for r in truth_rows}
    if not valid:
        raise ValueError("empty truth table")
    if any(len(r) != n_vars for r in valid):
        raise ValueError("every row must have length n_vars")

    p = 1 + n_vars + n_vars * (n_vars - 1) // 2          # params: c, a_i, b_ij
    A_eq, A_ub = [], []
    for x in itertools.product((0, 1), repeat=n_vars):
        phi = _features(x, n_vars)
        if x in valid:
            A_eq.append(phi)                              # H(x) = 0
        else:
            A_ub.append(-phi)                             # -H(x) ≤ -gap  ⇔  H(x) ≥ gap

    res = linprog(
        c=np.zeros(p),                                    # feasibility (bounds keep it bounded)
        A_ub=np.array(A_ub) if A_ub else None,
        b_ub=-gap * np.ones(len(A_ub)) if A_ub else None,
        A_eq=np.array(A_eq),
        b_eq=np.zeros(len(A_eq)),
        bounds=[(-bound, bound)] * p,
        method="highs",
    )
    if not res.success:
        raise NotQuadratic(
            f"no QUBO over {n_vars} vars reproduces this truth table "
            f"(linprog: {res.message.strip()}); needs an auxiliary spin or composition")

    theta = res.x
    Q = np.zeros((n_vars, n_vars))
    offset = float(theta[0])
    for i in range(n_vars):
        Q[i, i] = theta[1 + i]
    for k, (i, j) in enumerate(itertools.combinations(range(n_vars), 2)):
        Q[i, j] = theta[1 + n_vars + k]
    return Q, offset


def qubo_energy(Q: np.ndarray, offset: float, x) -> float:
    """H(x) = x·Q·x + offset for a binary vector x (Q upper-triangular)."""
    x = np.asarray(x, dtype=np.float64)
    return float(x @ Q @ x + offset)


def qubo_ground_states(Q: np.ndarray, offset: float, n: int) -> frozenset:
    """The realised truth table: all binary rows that minimise the synthesised penalty.

    This enumerates 2ⁿ — it is the **certified** check that a synthesised QUBO's ground-state
    *set* equals the target truth table. Use it only on small n. To find *one* low-energy
    assignment of a larger QUBO, use ``minimise_qubo`` (routes through ``drift.solve``;
    ``certified`` is True only when the exact engine ran).
    """
    configs = list(itertools.product((0, 1), repeat=n))
    energies = [qubo_energy(Q, offset, c) for c in configs]
    emin = min(energies)
    return frozenset(c for c, e in zip(configs, energies) if abs(e - emin) < 1e-6)


def minimise_qubo(Q: np.ndarray, offset: float = 0.0, **solve_kw) -> Solution:
    """Find a low-energy assignment of the QUBO via ``drift.solve``.

    Use this when *one* minimum is enough — evaluating a synthesised gate, reading a
    circuit output. A heuristic minimum is acceptable there; the result's ``certified``
    flag is True only for the exact engine. Pass ``require_certified=True`` when a proven
    ground state is required.

    For the *full* ground-state set (truth-table certification), keep
    ``qubo_ground_states``, which enumerates 2ⁿ and is only for small n.
    """
    return solve(qubo_to_ising(Q, offset), **solve_kw)


def qubo_to_ising(Q: np.ndarray, offset: float = 0.0) -> IsingModel:
    """Map the QUBO (x ∈ {0,1}) to DRIFT's Ising model (s ∈ {−1,+1}, E = −½sᵀJs − h·s).

    x_i = (1 + s_i)/2 gives J_ij = −b_ij/4 and h_i = −(a_i/2 + Σ_{j≠i} b_ij/4); the constant
    is dropped (it shifts every energy equally and does not move the ground state).
    """
    n = Q.shape[0]
    B = np.zeros((n, n))                                   # symmetric pair couplings b_ij
    for i, j in itertools.combinations(range(n), 2):
        B[i, j] = B[j, i] = Q[i, j]
    J = -B / 4.0
    h = -(np.diag(Q) / 2.0 + B.sum(axis=1) / 4.0)
    return IsingModel(h=h, J=J)


def spins_to_bits(s: np.ndarray) -> tuple[int, ...]:
    """Ising spin config (±1) back to QUBO bits (0/1)."""
    return tuple(((np.asarray(s) + 1) // 2).astype(int).tolist())


# ── composition: build a non-quadratic function from synthesised gates ────────────────
def _gate_qubo(name: str):
    """Synthesise a primitive gate's QUBO from its truth table (round-trip on demand)."""
    from .logic import TRUTH
    n = 3 if name in ("AND", "OR") else 2
    return synthesize(TRUTH[name], n)


def xor_via_composition() -> frozenset:
    """XOR = (x∨y) ∧ ¬(x∧y), assembled by **adding** synthesised penalties over wires.

    Variables (x, y, a=x∧y, o=x∨y, b=¬a, z). The combined penalty's ground states, with the
    internal wires marginalised out, are the XOR truth table — universality from parts that
    are each individually quadratic, the standard cure for XOR's non-quadraticity.
    """
    Qand, cand = synthesize(truth_from_fn(lambda x, y: x & y, 2), 3)
    Qor, cor = synthesize(truth_from_fn(lambda x, y: x | y, 2), 3)
    Qnot, cnot = synthesize(truth_from_fn(lambda a: 1 - a, 1), 2)

    # column order: x y a o b z
    X, Y, A, O, B, Z = range(6)

    def energy(cfg):
        x, y, a, o, b, z = cfg
        e = qubo_energy(Qand, cand, (x, y, a))            # a = x ∧ y
        e += qubo_energy(Qor, cor, (x, y, o))             # o = x ∨ y
        e += qubo_energy(Qnot, cnot, (a, b))              # b = ¬a
        e += qubo_energy(Qand, cand, (o, b, z))           # z = o ∧ b = (x∨y) ∧ ¬(x∧y)
        return e

    configs = list(itertools.product((0, 1), repeat=6))
    energies = [energy(c) for c in configs]
    emin = min(energies)
    gs = [c for c, e in zip(configs, energies) if abs(e - emin) < 1e-6]
    return frozenset((c[X], c[Y], c[Z]) for c in gs)


def _main() -> None:
    from .logic import TRUTH
    from .solvers import simulated_annealing

    print("=== DRIFT — inverse computronium: target truth table -> QUBO penalty ===\n")

    print("  synthesise primitive gates from their truth tables (round-trip):")
    for name in ("AND", "OR", "NOT"):
        n = 3 if name in ("AND", "OR") else 2
        Q, off = synthesize(TRUTH[name], n)
        ok = qubo_ground_states(Q, off, n) == TRUTH[name]
        print(f"    {name:<4} -> QUBO {n}×{n}, ground states == truth table: {ok}")

    print("\n  synthesise an arbitrary target (3-in majority z = maj(x,y,w)):")
    maj = truth_from_fn(lambda x, y, w: (x + y + w) >= 2, 3)
    Qm, offm = synthesize(maj, 4)
    print(f"    majority round-trip: {qubo_ground_states(Qm, offm, 4) == maj}")

    print("\n  honest limit — XOR is not 2-local over its own variables:")
    try:
        synthesize(truth_from_fn(lambda x, y: x ^ y, 2), 3)
        print("    (unexpected) XOR synthesised directly")
    except NotQuadratic as exc:
        print(f"    NotQuadratic raised  ✓   ({str(exc).split(';')[0]})")
    print(f"    XOR via composition of synthesised gates == XOR truth table: "
          f"{xor_via_composition() == truth_from_fn(lambda x, y: x ^ y, 2)}")

    print("\n  consume the solver path — let DRIFT's solve / SA compute the synthesised AND:")
    Qa, offa = synthesize(TRUTH['AND'], 3)
    sol = minimise_qubo(Qa, offa)
    model = qubo_to_ising(Qa, offa)
    s_sa, _, _, _ = simulated_annealing(model, n_sweeps=400, seed=0)
    print(f"    solve ({sol.method}, certified={sol.certified}) -> bits {spins_to_bits(sol.s)}  "
          f"(valid AND row: {spins_to_bits(sol.s) in TRUTH['AND']})")
    print(f"    annealed (SA, 400 sweeps) -> bits {spins_to_bits(s_sa)}  (valid AND row: "
          f"{spins_to_bits(s_sa) in TRUTH['AND']})")


if __name__ == "__main__":
    _main()
