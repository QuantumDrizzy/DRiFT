"""
drift.circuits — Phase 12: universal computation as an Ising ground state.
==========================================================================
Phase 11 made matter compute *one* function (multiply → factor). This phase makes it compute
**any** function. Logic gates are synthesised as QUBO penalties (`inverse_logic.synthesize`),
then **composed by sharing wires**: the total penalty of a wired-up netlist has energy 0 exactly
on the consistent evaluations of the whole circuit. Clamp the inputs and the ground state *is*
the output — a Boolean circuit computed by relaxing a spin system.

AND, OR and NOT are each 2-local (a direct QUBO). XOR is not (the textbook non-quadratic case),
so it is built by composition — `(x∨y) ∧ ¬(x∧y)` — exactly the cure `inverse_logic` already
demonstrates. AND/OR/NOT are functionally complete, so this is genuine universality: we wire a
**1-bit full adder** and DRIFT's own Ising engine computes its full truth table.

**Honest scope:** the same wall as Phase 11 — a `k`-bit ripple adder needs ~14·k variables, so
the exact engine tops out around a 1-bit adder; wider circuits are the *principle*, bounded by
the exponential state space, not solved here. What is shown is real: arithmetic and, by
universality, any Boolean function, computed as the ground state of matter.
"""

from __future__ import annotations

import itertools

import numpy as np

from .inverse_logic import qubo_to_ising, spins_to_bits, synthesize, truth_from_fn
from .solvers.exact import exact_ground_state

# Primitive gates as (Q, offset), synthesised once from their truth tables.
_GATES = {
    "AND": synthesize(truth_from_fn(lambda x, y: x & y, 2), 3),
    "OR": synthesize(truth_from_fn(lambda x, y: x | y, 2), 3),
    "NOT": synthesize(truth_from_fn(lambda a: 1 - a, 1), 2),
}


class Circuit:
    """A Boolean circuit compiled to one QUBO: gates are penalties summed over shared, named
    wires. Clamp the inputs, find the ground state with DRIFT's Ising engine, read the outputs."""

    def __init__(self) -> None:
        self._idx: dict[str, int] = {}
        self._gates: list[tuple[np.ndarray, float, list[int]]] = []
        self._auto = 0

    def var(self, name: str) -> int:
        if name not in self._idx:
            self._idx[name] = len(self._idx)
        return self._idx[name]

    @property
    def n(self) -> int:
        return len(self._idx)

    def add(self, gate: str, *wires: str) -> str:
        """Add a primitive gate over `wires` (inputs…, output). Returns the output wire name."""
        Q, off = _GATES[gate]
        gi = [self.var(w) for w in wires]
        self._gates.append((Q, off, gi))
        return wires[-1]

    def add_xor(self, x: str, y: str, out: str) -> str:
        """XOR by composition: out = (x∨y) ∧ ¬(x∧y), via three internal wires."""
        tag = f"#{self._auto}"
        self._auto += 1
        a, o, b = f"and{tag}", f"or{tag}", f"not{tag}"
        self.add("AND", x, y, a)
        self.add("OR", x, y, o)
        self.add("NOT", a, b)
        self.add("AND", o, b, out)
        return out

    def _total_qubo(self) -> tuple[np.ndarray, float]:
        n = self.n
        Q = np.zeros((n, n))
        offset = 0.0
        for Ql, ol, gi in self._gates:
            m = len(gi)
            for i in range(m):
                Q[gi[i], gi[i]] += Ql[i, i]
                for j in range(i + 1, m):
                    a, b = (gi[i], gi[j]) if gi[i] < gi[j] else (gi[j], gi[i])
                    Q[a, b] += Ql[i, j]
            offset += ol
        return Q, offset

    def evaluate(self, inputs: dict[str, int], outputs: list[str],
                 clamp: float = 1000.0) -> dict:
        """Clamp `inputs`, let DRIFT's exact Ising engine find the ground state, read `outputs`.
        Returns the output bits plus the ground-state energy (0 ⇔ a consistent evaluation)."""
        Q, offset = self._total_qubo()
        Qc = Q.copy()
        for name, v in inputs.items():
            k = self._idx[name]
            Qc[k, k] += clamp * (1.0 - 2.0 * v)  # field forcing x_k → v
        model = qubo_to_ising(Qc, offset)
        s_min, _, _ = exact_ground_state(model)
        bits = spins_to_bits(s_min)
        # Report the *unclamped* circuit penalty on the solution: 0 ⇔ every gate is satisfied
        # (the clamp's large constants make the raw Ising energy uninformative).
        x = np.asarray(bits, dtype=np.float64)
        penalty = float(x @ Q @ x + offset)
        return {
            "out": {o: bits[self._idx[o]] for o in outputs},
            "energy": penalty,
        }


def full_adder(circuit: Circuit, a: str, b: str, cin: str, s: str, cout: str) -> None:
    """Wire a 1-bit full adder into `circuit`: (a, b, cin) → (s = sum, cout = carry)."""
    tag = f"_{a}{b}{cin}"
    s1 = circuit.add_xor(a, b, f"s1{tag}")          # a ⊕ b
    circuit.add_xor(s1, cin, s)                      # sum = a ⊕ b ⊕ cin
    circuit.add("AND", a, b, f"ab{tag}")            # carry term a·b
    circuit.add("AND", s1, cin, f"sc{tag}")         # carry term (a⊕b)·cin
    circuit.add("OR", f"ab{tag}", f"sc{tag}", cout)  # cout = a·b + (a⊕b)·cin


def adder_truth_table(circuit: Circuit, a: str, b: str, cin: str,
                      s: str, cout: str) -> list[dict]:
    """Run DRIFT's engine on all 8 inputs of a 1-bit full adder; return the computed rows."""
    rows = []
    for va, vb, vc in itertools.product((0, 1), repeat=3):
        r = circuit.evaluate({a: va, b: vb, cin: vc}, [s, cout])
        out = r["out"]
        rows.append({
            "a": va, "b": vb, "cin": vc,
            "sum": out[s], "cout": out[cout],
            "expected": va + vb + vc,
            "ok": (out[cout] * 2 + out[s]) == (va + vb + vc),
            "energy": r["energy"],
        })
    return rows
