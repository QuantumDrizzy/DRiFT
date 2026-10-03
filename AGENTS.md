# AGENTS.md — orientation for AI assistants working on DRiFT

DRiFT is a Python Ising/tensor-network engine for studying computation as energy
minimisation. Read `README.md` first, then `docs/ROADMAP.md` for what each phase built.

## Layout

| Path | What lives there |
|---|---|
| `drift/` | the engine. `ising.py` (core model), `solve.py` (the dispatcher every face uses), `solvers/`, `builders/`, one module per phase (`circuits.py` P12, `mps.py` P13, `gpu.py` P14, `drawing.py` P15, `graphity.py` P16, `scrambling.py` P17, `freefermion.py` P18, …) |
| `tests/` | CPU pytest suite, one file per phase. GPU tests skip without the local binary |
| `experiments/` | one script per phase; produce the figures in `figures/` |
| `docs/results/PHASE{N}-results.md` | what each phase measured, with its honest scope |
| `docs/ADR-*.md` | architecture decisions |
| `cuda/` | the Phase-14 GPU engine; a local Windows/sm_120 build, not in CI |

## Conventions

- **A phase** = a module + `tests/test_<name>.py` + `docs/results/PHASE{N}-results.md` + a
  `docs/ROADMAP.md` entry. Add a figure via `experiments/` when one exists.
- **Never pass a heuristic off as exact.** `drift.solve` returns `certified=True` only from the
  exact engine; keep that distinction in anything new.
- **Oracles before scale.** Small cases are checked against an exact method: `exact_ground_state`
  for classical faces, `drift.quantum` (Lanczos) and `drift.freefermion` (any length) for the
  transverse-field Ising chain.
- Science vs speculation is tagged in `docs/CONCEPTS.md`. Keep claims in the established part.
- Code and comments in English.

## Running

```bash
pip install -e ".[dev]"
pytest tests/ -q          # ~1 min on CPU; CI runs exactly this
```

Run numpy-heavy jobs with `OMP_NUM_THREADS=1` when several run in parallel: concurrent
processes oversubscribing BLAS threads stalled runs by an order of magnitude.

## Where P15–P18 were measured

Phases 15–18 were measured against predictions committed *before* the code, in the private
`QuantumDrizzy/rse-hpc-lab` repository (ADR-004, ADR-005; exercises 11–14). The results
notes here carry the numbers and the misses; the lab carries the pre-registration and the
raw per-run tables.

## Known limits to respect

- `drift.mps.ground_state` with default arguments **under-converges at the critical point**
  (c = 0.28 instead of 0.5 at n = 64): its stopping rule is a per-sweep relative energy change
  that critical slowing down fools. Use `max_sweeps`/`econv` explicitly, or check against
  `drift.freefermion`, until the rule is fixed. See `docs/results/PHASE18-results.md`.
- `drift.graphity` minima are Metropolis results, not certified ground states.
- Exact reach: `drift.solve` certifies up to `exact_max` (default 18 spins); `drift.quantum`
  is practical to ~16 spins. Past that, results are heuristic and must say so.
