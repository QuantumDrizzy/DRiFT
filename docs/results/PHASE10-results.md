# Phase 10 — quantum vs simulated annealing, honestly

**Understood:** *the question Phase 9 deferred — is quantum annealing better than thermal
annealing? The honest answer is "it depends on what is hard."* Quantum annealing has a real,
measured edge exactly where the bottleneck is a barrier thin enough to tunnel — and none at
all where the landscape is easy. A specific advantage, not a general one.

**Built** (`drift/tunneling.py`, `experiments/phase10_tunneling.py`): both annealers run on
the *same* energy landscape (the diagonal `energies[b]`), so the comparison is exact.

- `hamming_cost_energies(n, spike_at, spike_height)` — cost = the Hamming weight `w` (a smooth
  funnel down to the all-`+1` ground state at `w = 0`), with an optional **tall thin spike** at
  one weight. The textbook Farhi-spike barrier.
- `metropolis_sa(energies, n, …)` — single-spin-flip Metropolis annealing: local moves that
  change `w` by ±1, so a barrier in Hamming weight is a wall it must *climb*.
- Quantum side reuses `drift.anneal.quantum_anneal` (tunnels under the transverse field).

**Validated** (`tests/test_tunneling.py`, 4/4; deterministic — QA is pure linear algebra, the
SA seed is fixed). On `n = 10`, spike of height 10 at weight 2:

| | simulated annealing | quantum annealing (T=40) |
|---|---|---|
| **funnel** (no barrier) | **1.00** | **0.99** |
| **spike** (thin barrier) | **0.17** | **0.45** |

- **SA solves the funnel but is walled by the spike** (1.00 → 0.17): single-spin-flip moves
  cannot climb the barrier, and more sweeps barely help — it is a wall, not a slope.
- **Quantum annealing tunnels the spike** (0.45 vs SA's 0.17 — a ~2.6× edge), and rises as the
  anneal slows (0.21 at T=20 → 0.45 at T=40 → ~0.6 at T=60): a real, measured quantum edge
  exactly where local search is stuck.
- **The edge is specific, not general:** on the funnel quantum annealing has no advantage (both
  ≈ 1.0), and even on the spike it does not get the answer for free — a taller barrier shrinks
  its spectral gap too, so QA's spike success (0.45) stays well below its funnel success (0.99).

**Figure:** `figures/phase10_tunneling.png` — (a) QA success vs anneal time `T` (tunnels the
spike as `T` grows); (b) SA success vs sweeps (walled out by the spike, flat under added
effort). The two panels side by side *are* the result.

**Honest scope:** simulated exactly on small `n`, and deliberately not a sweeping "quantum
supremacy" claim. The advantage shown is the well-understood tunnelling-vs-barrier effect for a
*thin* barrier; for wide barriers, or where the spectral gap closes (Phase 9), the quantum
route has no edge or fails outright. The point of the microscope is to locate the advantage
precisely and honestly, not to oversell it.
