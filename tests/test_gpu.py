"""GPU engine tests — skipped when the local CUDA binary is not built.

CI is CPU-only. The Windows/sm_120 binary (`cuda/ising_pt.exe`) is a local artifact
and is never committed. Prefer skip over fail when it is missing.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from drift.builders.qubo import maxcut_ising, random_graph  # noqa: E402
from drift.gpu import gpu_available, parallel_tempering_gpu  # noqa: E402
from drift.ising import IsingModel  # noqa: E402
from drift.solvers.exact import exact_ground_state  # noqa: E402


def _tiny_ferro() -> IsingModel:
    n = 4
    J = np.zeros((n, n))
    for i in range(n - 1):
        J[i, i + 1] = J[i + 1, i] = 1.0
    return IsingModel(J=J, h=np.zeros(n))


def test_gpu_available_is_a_bool():
    """The probe itself must run on CPU CI — True iff the local binary exists."""
    assert isinstance(gpu_available(), bool)


def test_gpu_raises_when_binary_missing():
    """Without the CUDA engine, the glue fails loudly with build instructions."""
    if gpu_available():
        pytest.skip("CUDA engine binary is present")
    with pytest.raises(RuntimeError, match="not built"):
        parallel_tempering_gpu(_tiny_ferro(), n_replicas=2, n_rounds=2)


@pytest.mark.gpu
@pytest.mark.skipif(not gpu_available(), reason="CUDA engine binary not built")
def test_gpu_reproduces_exact_on_small_maxcut():
    """On-device acceptance: GPU PT matches exact ground energy on a small MaxCut."""
    model = maxcut_ising(random_graph(12, p=0.5, seed=3))
    _, e_exact, _ = exact_ground_state(model)
    res = parallel_tempering_gpu(
        model, n_replicas=16, T_min=0.05, T_max=4.0, n_rounds=200, sweeps_per_round=4, seed=1
    )
    assert res.best_E == pytest.approx(e_exact)
