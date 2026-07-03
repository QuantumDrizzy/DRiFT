"""
drift.gpu — the GPU Ising engine (Python glue to the CUDA parallel-tempering binary).

The heavy solve lives in `cuda/ising_pt.cu` (compiled to `cuda/ising_pt.exe`); this module is the
thin bridge so the Python faces can offload a general Ising/QUBO to the GPU and get the ground
state back, with the *same* `PtResult` the CPU reference returns. Right language per domain: the
kernel is CUDA, the glue and the faces are Python.

Contract: `parallel_tempering_gpu(model, …)` is drop-in with
`drift.solvers.parallel_tempering.parallel_tempering`, so the falsifier is trivial — run both on a
small instance and they must agree with `exact_ground_state`. If the binary is not built,
`gpu_available()` is False and the call raises with the build instructions.
"""

from __future__ import annotations

import os
import re
import struct
import subprocess
import tempfile

import numpy as np

from .ising import IsingModel
from .solvers.parallel_tempering import PtResult, geometric_ladder

# repo_root/cuda/ising_pt.exe  (this file is repo_root/drift/gpu.py)
_BIN = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cuda",
                    "ising_pt.exe" if os.name == "nt" else "ising_pt")


def gpu_binary_path() -> str:
    """Absolute path where the compiled CUDA engine is expected."""
    return _BIN


def gpu_available() -> bool:
    """True iff the CUDA engine has been built (see cuda/build.bat)."""
    return os.path.exists(_BIN)


def parallel_tempering_gpu(
    model: IsingModel,
    *,
    n_replicas: int = 16,
    T_min: float = 0.1,
    T_max: float = 5.0,
    n_rounds: int = 300,
    sweeps_per_round: int = 5,
    seed: int = 0,
) -> PtResult:
    """Solve `model` by GPU parallel tempering. Same signature/return as the CPU reference.

    Serializes (J, h, params) to a binary file, runs the CUDA engine, and reads back the best
    configuration and energy. `throughput` (spin-flips/sec) and `seconds` are parsed from the
    engine's report.
    """
    if not gpu_available():
        raise RuntimeError(
            f"GPU engine not built at {_BIN}.\n"
            "Build it from the x64 Native Tools Command Prompt for VS 2022:\n"
            "    cd cuda && build.bat   (nvcc -O3 -arch=sm_120 ising_pt.cu -o ising_pt.exe)"
        )

    n = model.n
    h = np.ascontiguousarray(model.h, dtype=np.float32)

    # CSR of the symmetric, zero-diagonal coupling matrix (np.nonzero is row-major, so already
    # grouped by row). Each flip then touches only a spin's neighbours — the Phase-14b O(degree) win.
    Jm = np.ascontiguousarray(model.J, dtype=np.float64)
    rows, cols = np.nonzero(Jm)
    col_idx = cols.astype(np.int32)
    weight = Jm[rows, cols].astype(np.float32)
    row_ptr = np.zeros(n + 1, dtype=np.int32)
    row_ptr[1:] = np.cumsum(np.bincount(rows, minlength=n))
    nnz = int(col_idx.size)

    with tempfile.TemporaryDirectory() as d:
        prob = os.path.join(d, "problem.bin")
        res = os.path.join(d, "result.bin")
        with open(prob, "wb") as f:
            # matches the packed header read in ising_pt.cu: <Q 5i 2f> then CSR + h
            f.write(struct.pack("<Qiiiiiff", seed & 0xFFFFFFFFFFFFFFFF, n, n_replicas, n_rounds,
                                sweeps_per_round, nnz, float(T_min), float(T_max)))
            f.write(row_ptr.tobytes())
            f.write(col_idx.tobytes())
            f.write(weight.tobytes())
            f.write(h.tobytes())

        proc = subprocess.run([_BIN, prob, res], capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"GPU engine failed (exit {proc.returncode}):\n{proc.stderr}")

        with open(res, "rb") as f:
            best_E = struct.unpack("<d", f.read(8))[0]
            best_s = np.frombuffer(f.read(4 * n), dtype=np.float32).astype(np.float64).copy()

    thr = _grab(proc.stderr, r"throughput=([\d.eE+]+)")
    sec = _grab(proc.stderr, r"time=([\d.]+)s")
    return PtResult(
        best_s=best_s,
        best_E=float(best_E),
        temperatures=geometric_ladder(n_replicas, T_min, T_max),
        swap_rate=None,
        throughput=thr,
        seconds=sec,
    )


def _grab(text: str, pattern: str):
    m = re.search(pattern, text)
    return float(m.group(1)) if m else None
