"""Validation of Phase 10 — quantum vs simulated annealing, honestly.

The numbers are deterministic (quantum annealing is pure linear algebra; the SA seed is
fixed), so these assert the honest, reproducible story:
  * single-spin-flip SA solves a plain funnel but is walled out by a thin Hamming-weight spike;
  * quantum annealing tunnels that spike — a real, measured edge over SA where SA is stuck;
  * but the edge is *specific*: on the funnel there is no quantum advantage, and even on the
    spike quantum annealing does not get the answer for free (a tall barrier costs it too).

Run standalone:  python tests/test_tunneling.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from drift.anneal import quantum_anneal, success_probability, transverse_driver  # noqa: E402
from drift.tunneling import hamming_cost_energies, metropolis_sa  # noqa: E402

N = 10
_DRIVER = transverse_driver(N)
_FUNNEL = hamming_cost_energies(N)
_SPIKE = hamming_cost_energies(N, spike_at=2, spike_height=10.0)


def _qa(energies, T, steps=160):
    psi = quantum_anneal(_DRIVER, sp.diags(energies, format="csr"), T, steps=steps)
    return success_probability(psi, energies)


def _sa(energies):
    return metropolis_sa(energies, N, num_reads=40, sweeps=250, seed=1)["success"]


def test_spike_landscape_has_a_barrier():
    """The spike adds a tall thin barrier; the global minimum (all +1, w=0) is unchanged."""
    assert _SPIKE.min() == _FUNNEL.min() == 0.0
    assert _SPIKE.argmin() == 0  # all-+1 config is still the ground state
    assert _SPIKE.max() > _FUNNEL.max()  # the barrier sticks up above the funnel


def test_sa_solves_funnel_but_is_walled_by_spike():
    """Local single-spin-flip SA descends the funnel easily, but the spike walls it out."""
    sa_funnel, sa_spike = _sa(_FUNNEL), _sa(_SPIKE)
    assert sa_funnel >= 0.8, sa_funnel
    assert sa_spike <= 0.5, sa_spike
    assert sa_funnel > sa_spike + 0.4  # the barrier is what hurts SA


def test_quantum_annealing_tunnels_the_spike():
    """Quantum annealing tunnels the spike — a real edge over SA where SA is stuck."""
    qa_spike = _qa(_SPIKE, 40.0)
    sa_spike = _sa(_SPIKE)
    assert qa_spike > sa_spike + 0.15, (qa_spike, sa_spike)


def test_the_edge_is_specific_not_general():
    """No quantum advantage on the easy funnel; and even on the spike QA does not solve it for
    free — a tall barrier shrinks its gap too. The honest bound on the advantage."""
    qa_funnel = _qa(_FUNNEL, 40.0)
    qa_spike = _qa(_SPIKE, 40.0)
    assert qa_funnel >= 0.9                 # funnel is easy for QA too — no special edge
    assert qa_spike < qa_funnel - 0.2       # the spike is genuinely hard for QA as well


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all Phase 10 tests passed")
