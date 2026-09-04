"""
DRIFT — a microscope for physical computation.

One Ising/tensor engine, many loads (QUBO, Hopfield, tiles, crystals). The shared
primitives live here; builders (Hamiltonian generators) and solvers plug into them.

Public surface (Phase 1):
    IsingModel               the core object: H = -1/2 sᵀJs - hᵀs
    exact_ground_state       brute-force ground state (small n)
    simulated_annealing      Metropolis relaxation = the physics computing
    magnetization, landauer_energy_j   observables
    viz.*                    relaxation / spin figures
"""

import sys as _sys

# Eight of the experiment scripts print physics notation (arrows, rho, chi) to
# stdout. On a Windows console the default cp1252 codec raises
# UnicodeEncodeError on those, which killed the run before the figure was
# saved. Reconfiguring here fixes every entry point that imports drift.
for _stream in (_sys.stdout, _sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError, OSError):
        pass  # not a reconfigurable text stream (redirected, or already utf-8)

from .ising import IsingModel
from .solvers.exact import exact_ground_state, all_configs
from .solvers.annealing import simulated_annealing
from .metrics import magnetization, energy_per_spin, landauer_energy_j
from .mps import ground_state as tensor_ground_state, MpsResult
from .solve import solve, Solution

__all__ = [
    "IsingModel",
    "exact_ground_state",
    "all_configs",
    "simulated_annealing",
    "tensor_ground_state",
    "MpsResult",
    "solve",
    "Solution",
    "magnetization",
    "energy_per_spin",
    "landauer_energy_j",
]

__version__ = "0.1.0.dev0"
