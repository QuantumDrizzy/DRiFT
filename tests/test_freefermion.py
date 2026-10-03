"""Validation of Phase 18 — the free-fermion oracle and entanglement scaling.

  * ground energy and every S(ℓ) agree with the exact Lanczos engine (`drift.quantum`);
  * the MPS solver (`drift.mps`) agrees with the oracle past the exact wall;
  * the critical chain's fitted central charge is ½; the gapped chain's is 0 (area law).

Run standalone:  python tests/test_freefermion.py
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from drift import freefermion as ff  # noqa: E402
from drift import mps as dmps  # noqa: E402
from drift import quantum as dq  # noqa: E402


def test_matches_exact_lanczos():
    L = 10
    for g in (0.7, 1.0, 1.5):
        h_zz, h_x = dq.tfim_terms(dq.ising_chain_1d(L))
        e0, psi = dq.ground_state(h_zz + g * h_x)
        s_ed = [dq.entanglement_entropy(psi, L, cut=l)[0] * math.log(2) for l in range(1, L)]
        assert abs(ff.ground_state(L, g)[1] - e0) < 1e-10
        assert np.abs(ff.block_entropies(L, g) - np.array(s_ed)).max() < 1e-10


def test_mps_matches_oracle():
    L = 24
    r = dmps.ground_state(L, j=1.0, gamma=1.0, chi_max=32)
    s_mps = np.array(dmps.bond_entropies(r.mps)) * math.log(2)
    assert np.abs(s_mps - ff.block_entropies(L, 1.0)).max() < 0.02
    assert r.energy >= ff.ground_state(L, 1.0)[1] - 1e-8


def test_central_charge():
    L = 256
    c_crit, _ = ff.fit_central_charge(ff.block_entropies(L, 1.0), L)
    c_gap, _ = ff.fit_central_charge(ff.block_entropies(L, 1.5), L)
    assert abs(c_crit - 0.5) < 0.02
    assert abs(c_gap) < 0.05


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
