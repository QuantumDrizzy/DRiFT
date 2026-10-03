# Phase 16 — graphity: does a geometry emerge, or is it put in?

**Understood:** *a local graph Hamiltonian, cooled from the complete graph, does not weave a
lattice.* It gives an expander, or it would fragment into dense blocks if the anneal could
reach them. Geometry has to be put in.

**Built** (`drift/graphity.py`, `tests/test_graphity.py`) — the classical, diagonal part of
Quantum Graphity (Konopka, Markopoulou, Smolin). The spins are **edges**: N(N−1)/2 binary
variables, Metropolis edge flips, geometric cooling from the complete graph.

- Bitset graphs: exact triangle and 4-cycle counts (checked against brute force), exact
  energy deltas per flip (checked against recounts for every Hamiltonian shape), diameter,
  ball growth, components, and a random-regular **null model** (an expander).
- `GraphHamiltonian(v0, g_v, g3, g4, power, keep_connected)` and `anneal(...)`.

**Measured** in rse-hpc-lab exercise 12 (ADR-005 and its Amendment 1), N = 128, v₀ = 4,
3 seeds, 300 sweeps, predictions committed before the runs. Digits in that repository's
`exercises/12-graphity/RESULTS.md`.

| Hamiltonian | Result | Prediction |
|---|---|---|
| H_A valence only | 4-regular, connected, diameter 6 = the random-regular null | hit — an expander, no geometry |
| H_B + triangles | stays **connected**, E ∈ [−30, −17.5]; disjoint K₅ built by hand: **E = −103** | **miss** — predicted fragmentation; the anneal freezes far above it |
| H_C + 4-cycles, quadratic valence | stays the **complete graph** (E ≈ −1.4·10⁷) | **miss** — and a design error, below |
| H_D = H_C + connectivity | same as H_C | **miss** |
| H_C′ quartic valence, g₄ = ⅛ | 4-regular, connected, diameter 6–7, ball growth ≈ the null; disjoint K₄,₄ by hand: **E = −72** vs best anneal −6 | hit (Q3′a, Q3′b) |
| H_D′ = H_C′ + connectivity | identical to H_C′ seed for seed — the rule never binds | **miss** on the coin flip (diameter ≥ 2× null) |

**The design error, kept as a regression test.** With a quadratic valence penalty, the
4-cycles through a vertex grow like d³ and win: the complete graph is H_C's minimum
(`test_quadratic_valence_lets_four_cycles_win_density`). This is why the original model uses
an exponential valence penalty. Quartic valence with g₄ = ⅛ restores degree ≈ v₀.

**Reading.** Cooling a complete graph under local terms produces either a random regular
graph — an expander, diameter ~log N, no dimension — or, at lower energy than anything the
anneal reaches, disjoint dense blocks. Neither is a lattice. "The graph weaves 3-D space"
needs a term or a constraint that names the target geometry.

**Honest scope.** Classical only (no hopping terms); Metropolis minima, not certified; one
size; 300 sweeps — a much slower anneal might reach the fragmented states, which would make
the reading stronger, not weaker.
