# Phase 15 — the drawing is the function (and only while it couples)

**Understood:** *Phase 12's coupling graph is a drawing that computes, so the old argument
"a PCB and a sigil are the same kind of object" can be measured instead of argued.* Wires are
nodes, couplings are traces, coupling strengths are line weights. Three questions:
does one line matter, does a look-alike compute, and does the drawing do anything when its
coupling is small against kT?

**Built** (`drift/drawing.py`, `tests/test_drawing.py`):

- `adder_drawing()` — Phase 12's full adder as (fields, couplings): 14 wires, 25 couplings.
- `AdderScorer` — exhaustive and **degeneracy-aware**: inputs are fixed by selecting
  configurations (no clamp constant), and a row is correct only if *every* minimum-energy
  configuration carries the right (sum, carry). `ground_set` gives the unclamped
  zero-energy set: the drawing run in every direction.
- `rewire` — Maslov–Sneppen double-edge swaps, weights travelling with their edge, so degree
  sequence and weight multiset are preserved exactly.
- `boltzmann_adder` — P(correct), bits removed, and the heat bill at inverse temperature β.

**Measured** in rse-hpc-lab exercise 11, against predictions committed before any code
(rse-hpc-lab `docs/adr/ADR-004`). Digits are in that repository's
`exercises/11-drawing-is-the-function/RESULTS.md`; all seven predictions held, and the two
results that looked off were re-derived before anything was written.

| Question | Result |
|---|---|
| One line deleted | 24 / 25 deletions break ≥ 1 row; **25 / 25** change the unclamped ground set |
| One line moved | 448 / 458 endpoint moves break ≥ 1 row |
| Same look, rewired | 1000 rewires with identical degree sequence and weights compute the adder **0** times (fully mixed); accuracy 0.27–0.28, a random map scores 0.25 |
| Coupling vs kT | β → 0: P = 0.25, ΔS = 0 — the drawing does nothing. P = ½ at β·gap ≈ 2. β → ∞: 11 bits removed, paid in heat ≥ 2× the Landauer minimum for a quench |

**Two corrections kept on the record.** The one "redundant" line couples two inputs; it is
invisible only because the inputs are clamped by hand (`test_input_input_line_is_redundant_
only_under_the_clamp`). And 0.375 is the best *constant* output, not "computes nothing" — a
random output map is 0.25.

**Reading.** The function lives in the exact lines (true), drawings that merely look alike
are different systems (so the PCB ≈ sigil claim is false), and a drawing computes only while
its coupling beats kT. A chalk line is coupled to no electron: it sits at β·J = 0.

**Honest scope.** One circuit, 14 spins, exhaustive. A coupling graph, not an image: no PCB
layout or sigil corpus is analysed.
