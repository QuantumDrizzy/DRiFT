"""Validation of Phase 17 — scrambling.

  * the reflection-even basis is an isometry and H does not leak out of it;
  * the eigenbasis OTOC equals brute-force e^{iHt} evolution, and F(0) = 1;
  * level statistics separate the chaotic point from the free-fermion chain;
  * on the free-fermion chain the Z–Z OTOC decays while the X–X one does not — an OTOC
    that decays is not, on its own, chaos.

Run standalone:  python tests/test_scrambling.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import scipy.sparse as sp
from scipy.linalg import expm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from drift import scrambling as scr  # noqa: E402

CHAOTIC = (1.05, 0.5)
INTEGRABLE = (1.05, 0.0)


def test_reflection_sector_is_exact():
    B = scr.reflection_even_basis(8)
    assert abs((B.T @ B - sp.identity(B.shape[1])).toarray()).max() < 1e-12
    assert scr.spacing_ratio(8, *CHAOTIC)["leak"] < 1e-10


def test_otoc_matches_brute_force():
    L, t = 6, 1.3
    H = scr.mixed_field_ising(L, *CHAOTIC).toarray()
    U = expm(-1j * H * t)
    W, V = scr.site_op(L, 0, "Z"), scr.site_op(L, 3, "Z")
    Wt = U.conj().T @ W @ U
    brute = np.trace(Wt @ V @ Wt @ V).real / 2 ** L
    F = scr.otoc(L, *CHAOTIC, "Z", [3], [0.0, t])
    assert abs(F[0, 0] - 1) < 1e-12
    assert abs(F[0, 1] - brute) < 1e-12


def test_level_statistics_separate_chaos():
    chaotic = scr.spacing_ratio(12, *CHAOTIC)["r_mean"]
    integrable = scr.spacing_ratio(12, *INTEGRABLE)["r_mean"]
    assert chaotic > 0.50 > 0.45 > integrable


def test_integrable_decay_is_not_chaos():
    times = np.arange(6.0, 10.01, 0.5)
    fx = scr.otoc(8, *INTEGRABLE, "X", [3], times).mean()
    fz = scr.otoc(8, *INTEGRABLE, "Z", [3], times).mean()
    assert fx > fz


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
