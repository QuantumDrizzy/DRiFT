# Phase 12 — universal computation as an Ising ground state

**Understood:** *Phase 11 made matter compute one function; this makes it compute **any**
function.* Logic gates are synthesised as QUBO penalties and composed by sharing wires — the
total penalty of a wired-up netlist has energy 0 exactly on the circuit's consistent
evaluations. Clamp the inputs and the ground state **is** the output. AND/OR/NOT are functionally
complete, so this is genuine universality.

**Built** (`drift/circuits.py`, `experiments/phase12_universal.py`), on top of Phase-11's
`inverse_logic`:

- Primitive gates `AND`, `OR`, `NOT` are synthesised from their truth tables (`synthesize`) —
  each a 2-local QUBO whose ground states are exactly the gate's rows.
- `Circuit` — variables by name; `add(gate, *wires)` sums a gate's penalty over shared global
  indices; `add_xor` builds the non-2-local XOR by composition `(x∨y) ∧ ¬(x∧y)`; `evaluate`
  clamps the inputs with a field, hands the QUBO to DRIFT's Ising engine
  (`qubo_to_ising → exact_ground_state`), and reads the outputs. The reported energy is the
  *unclamped* circuit penalty: **0 ⇔ every gate is satisfied**.
- `full_adder` wires (a, b, cin) → (sum, cout) from these gates (14 variables).

**Validated** (`tests/test_circuits.py`, 5/5):

- The primitive gates round-trip: their synthesised QUBO ground states equal their truth tables.
- **The full adder computes every input.** DRIFT's ground-state engine returns the correct
  (sum, cout) for all 8 combinations of (a, b, cin), penalty 0 each:

  | a b cin | sum cout | a+b+cin |
  |:---:|:---:|:---:|
  | 0 0 0 | 0 0 | 0 |
  | 0 0 1 | 1 0 | 1 |
  | 0 1 0 | 1 0 | 1 |
  | 0 1 1 | 0 1 | 2 |
  | 1 0 0 | 1 0 | 1 |
  | 1 0 1 | 0 1 | 2 |
  | 1 1 0 | 0 1 | 2 |
  | 1 1 1 | 1 1 | 3 |

- XOR, not 2-local on its own, is computed correctly by composition.
- **The gates are real constraints:** forcing a wrong output (clamp `sum=1` for `0+0+0`) leaves
  every assignment violating a gate — the penalty cannot reach 0. Matter will not compute a false
  result for free.
- **Honest scaling:** a 1-bit adder is 14 variables (solvable exactly); a 2-bit ripple adder is
  past 22 — the same wall as Phase 11, stated rather than hidden.

**Figure:** `figures/phase12_universal.png` — (a) the full-adder truth table the ground state
computes; (b) circuit variables vs ripple-adder width against the exact engine's 22-variable
ceiling.

**Honest scope:** the *principle* is universal — AND/OR/NOT compose to any Boolean function, so
in principle any circuit is a ground state. The *demonstration* is a 1-bit adder, because the
exact engine's state space is exponential in the variable count (a `k`-bit ripple adder needs
~14·k variables). Wider arithmetic is real in principle and bounded in practice — exactly the
computronium thesis, measured and not oversold: matter computing, with the cost out in the open.
