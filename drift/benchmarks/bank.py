"""Versioned instance bank: stable IDs, seeded generators, a checked-in manifest.

The catalog is ``drift/benchmarks/instances/manifest.json``. IDs look like
``v1.maxcut-er.n008.s001`` and never change meaning inside a bank version: bump
``BANK_VERSION`` if a generator's output would change.

Small n≤8 classical instances are also checked in as JSON fixtures (J, h, known
energy) so a generator regression is caught without re-deriving the graph.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from ..ising import IsingModel
from ..quantum import ground_state as quantum_ground_state
from ..quantum import ising_chain_1d, tfim_terms
from ..solvers.exact import exact_ground_state
from .families import (
    FAMILIES,
    MPS_FAMILIES,
    MPS_N_MAX_SWEEP,
    analytic_energy,
    build_model,
    unknown_family,
)

BANK_VERSION = "v1"
MANIFEST_NAME = "manifest.json"

# Exact enumeration is cheap through n=20 (2^20 ≈ 1M); n=24 is 16M configs and is
# not precomputed. Analytic families keep a known energy at every n.
EXACT_ENERGY_N_MAX = 20
TFIM_LANCZOS_N_MAX = 14

# Even-n ladder. PR #4 shipped 8…32 with a gap after 20; v1 grows by *adding* IDs
# (22, 28, 36, 40) and extra seeds. Existing IDs keep their meaning — do not bump
# BANK_VERSION unless a generator's (J, h) for an existing ID would change.
CLASSICAL_N_LADDER = (8, 10, 12, 14, 16, 18, 20, 22, 24, 28, 32, 36, 40)

PRIMARY_SEEDS: dict[str, int] = {
    "maxcut-er": 1,
    "pmj-glass": 7,
    "bipartite-maxcut": 2,
    "ferro-chain": 0,
}

# Extra seeds only on stochastic families. ferro-chain ignores seed (the chain is
# determined by n and J), so duplicating it would be the same Hamiltonian twice.
EXTRA_SEEDS: dict[str, tuple[int, ...]] = {
    "maxcut-er": (2, 3),
    "pmj-glass": (8, 9),
    "bipartite-maxcut": (3, 4),
}
# Exact-cheap n (milliseconds). Extra seeds here keep CI and local exact rungs honest
# about instance-to-instance scatter without a PT-time blowup.
CHEAP_EXTRA_N = (8, 10, 12, 14, 16, 18)
# One extra seed at the exact → PT dispatcher wall (n=20 still has an exact oracle
# for MaxCut ER / ±J). Not repeated at n>20: those PT rungs are the expensive ones.
PT_BOUNDARY_EXTRA_N = (20,)


def _instances_dir() -> Path:
    return Path(__file__).resolve().parent / "instances"


def manifest_path() -> Path:
    return _instances_dir() / MANIFEST_NAME


def fixtures_dir() -> Path:
    return _instances_dir() / "fixtures"


@dataclass
class InstanceSpec:
    """One row of the catalog — enough to rebuild the instance bit-identically."""

    id: str
    family: str
    n: int
    seed: int
    params: dict = field(default_factory=dict)
    notes: str = ""
    known_energy: float | None = None
    known_energy_source: str | None = None
    fingerprint: str = ""

    def to_json(self) -> dict[str, Any]:
        row = asdict(self)
        return row


@dataclass
class BuiltInstance:
    spec: InstanceSpec
    model: IsingModel
    extra: dict = field(default_factory=dict)


def make_id(family: str, n: int, seed: int) -> str:
    if family not in FAMILIES:
        unknown_family(family)
    return f"{BANK_VERSION}.{family}.n{n:03d}.s{seed:03d}"


def fingerprint(model: IsingModel) -> str:
    """Short sha256 of (J, h) in C-contiguous float64 — pins generator output."""
    payload = np.ascontiguousarray(model.J, dtype=np.float64).tobytes()
    payload += np.ascontiguousarray(model.h, dtype=np.float64).tobytes()
    return hashlib.sha256(payload).hexdigest()[:16]


def _classical_row(family: str, n: int, seed: int, *, extra: bool = False) -> dict[str, Any]:
    """One catalog dict for a classical (non-crystal, non-TFIM) family."""
    tag = " Extra seed (same ensemble, independent draw)." if extra else ""
    if family == "maxcut-er":
        return {
            "family": family, "n": n, "seed": seed,
            "params": {"p": 0.5},
            "notes": (
                "Erdős–Rényi MaxCut G(n, 1/2); known energy from exact enum at n≤20."
                + tag
            ),
        }
    if family == "pmj-glass":
        return {
            "family": family, "n": n, "seed": seed,
            "params": {"p": 1.0, "field": 0.0},
            "notes": (
                "Complete ±J spin glass, h=0; frustrated; exact energy at n≤20."
                + tag
            ),
        }
    if family == "bipartite-maxcut":
        return {
            "family": family, "n": n, "seed": seed,
            "params": {"p": 0.5},
            "notes": (
                "Bipartite MaxCut; analytic optimum = every edge, E = −n_edges."
                + tag
            ),
        }
    if family == "ferro-chain":
        return {
            "family": family, "n": n, "seed": seed,
            "params": {"j": 1.0},
            "notes": "Open ferro chain; analytic E = −(n−1). 1-D geometry, not an MPS load.",
        }
    unknown_family(family)


def extra_seeds_for(family: str, n: int) -> tuple[int, ...]:
    """Extra seeds at this n, or empty. ferro-chain is seed-invariant — never duplicated."""
    extras = EXTRA_SEEDS.get(family, ())
    if not extras:
        return ()
    if n in CHEAP_EXTRA_N:
        return extras
    if n in PT_BOUNDARY_EXTRA_N:
        return extras[:1]
    return ()


def catalog_rows() -> list[dict[str, Any]]:
    """The v1 size ladder. Add IDs freely; bump BANK_VERSION only if a generator changes."""
    rows: list[dict[str, Any]] = []

    for n in CLASSICAL_N_LADDER:
        for family, seed in PRIMARY_SEEDS.items():
            rows.append(_classical_row(family, n, seed, extra=False))
        for family, extras in EXTRA_SEEDS.items():
            for seed in extra_seeds_for(family, n):
                rows.append(_classical_row(family, n, seed, extra=True))

    # Period-4 crystal needs cols % 4 == 0. rows is free. Denser size steps inside n≤40.
    for n, rows_c, cols_c, note in (
        (16, 4, 4, "Phase-6 stripe crystal 4×4; analytic E = −2n, period 4."),
        (24, 6, 4, "Phase-6 stripe crystal 6×4; analytic E = −2n. Size step between 4×4 and 4×8."),
        (32, 4, 8, "Phase-6 stripe crystal 4×8; analytic E = −2n. Cheap self-rep size step."),
        (40, 10, 4, "Phase-6 stripe crystal 10×4; analytic E = −2n. Size step at n=40."),
    ):
        rows.append({
            "family": "crystal", "n": n, "seed": 0,
            "params": {"rows": rows_c, "cols": cols_c},
            "notes": note,
        })

    # TFIM chain: χ lives here. Sweep default cuts at MPS_N_MAX_SWEEP; n=20,24 are
    # in the bank for local runs (documented cutoff MPS_N_MAX_LOCAL).
    for n in (8, 10, 12, 14, 16, 20, 24):
        rows.append({
            "family": "tfim-chain", "n": n, "seed": 0,
            "params": {"j": 1.0, "gamma": 1.0},
            "notes": (
                "Open TFIM at Γ/J = 1 (critical-ish). Solved by MPS/TEBD, not drift.solve. "
                f"Sweep χ cutoff n≤{MPS_N_MAX_SWEEP} (local feasible ≤24; Phase 13 showed 48)."
            ),
        })
    return rows


def _tfim_lanczos_energy(n: int, j: float, gamma: float) -> float:
    h_zz, h_x = tfim_terms(ising_chain_1d(n, j))
    e0, _ = quantum_ground_state(h_zz + gamma * h_x)
    return float(e0)


def _fill_known_energy(spec: InstanceSpec, built: BuiltInstance) -> None:
    e, src = analytic_energy(spec.family, spec.n, built.extra)
    if e is not None:
        spec.known_energy = float(e)
        spec.known_energy_source = src
        return
    if spec.family == "tfim-chain" and spec.n <= TFIM_LANCZOS_N_MAX:
        gamma = float(spec.params.get("gamma", 1.0))
        j = float(spec.params.get("j", 1.0))
        spec.known_energy = _tfim_lanczos_energy(spec.n, j, gamma)
        spec.known_energy_source = "lanczos-exact"
        return
    if spec.family in ("maxcut-er", "pmj-glass") and spec.n <= EXACT_ENERGY_N_MAX:
        _, e_min, _ = exact_ground_state(built.model, max_n=max(EXACT_ENERGY_N_MAX, spec.n))
        spec.known_energy = float(e_min)
        spec.known_energy_source = "exact-enumeration"
        return
    spec.known_energy = None
    spec.known_energy_source = None


def build_instance(spec: InstanceSpec) -> BuiltInstance:
    """Rebuild one catalog instance from its spec (seeded, bit-stable)."""
    model, extra = build_model(spec.family, spec.n, spec.seed, spec.params)
    fp = fingerprint(model)
    if spec.fingerprint and spec.fingerprint != fp:
        raise ValueError(
            f"fingerprint mismatch for {spec.id}: catalog {spec.fingerprint} vs generator {fp}. "
            "Bump BANK_VERSION if the generator changed on purpose."
        )
    return BuiltInstance(spec=spec, model=model, extra=extra)


def spec_from_row(row: dict[str, Any]) -> InstanceSpec:
    return InstanceSpec(
        id=row["id"],
        family=row["family"],
        n=int(row["n"]),
        seed=int(row["seed"]),
        params=dict(row.get("params") or {}),
        notes=str(row.get("notes") or ""),
        known_energy=row.get("known_energy"),
        known_energy_source=row.get("known_energy_source"),
        fingerprint=str(row.get("fingerprint") or ""),
    )


def load_manifest(path: Path | None = None) -> list[InstanceSpec]:
    """Load the checked-in catalog. Missing file → build the in-code ladder (no fingerprints)."""
    p = path or manifest_path()
    if not p.is_file():
        return [_spec_from_catalog_dict(d) for d in catalog_rows()]
    data = json.loads(p.read_text(encoding="utf-8"))
    if data.get("version") != BANK_VERSION:
        raise ValueError(
            f"manifest version {data.get('version')!r} != BANK_VERSION {BANK_VERSION!r}"
        )
    return [spec_from_row(row) for row in data["instances"]]


def _spec_from_catalog_dict(d: dict[str, Any]) -> InstanceSpec:
    family = d["family"]
    n = int(d["n"])
    seed = int(d["seed"])
    return InstanceSpec(
        id=make_id(family, n, seed),
        family=family,
        n=n,
        seed=seed,
        params=dict(d.get("params") or {}),
        notes=str(d.get("notes") or ""),
    )


def iter_instances(
    *,
    n_min: int = 1,
    n_max: int | None = None,
    families: tuple[str, ...] | None = None,
    specs: list[InstanceSpec] | None = None,
) -> Iterator[BuiltInstance]:
    """Yield built instances, optionally filtered by size and family."""
    wanted = set(families) if families is not None else None
    if wanted:
        for f in wanted:
            if f not in FAMILIES:
                unknown_family(f)
    for spec in (specs if specs is not None else load_manifest()):
        if spec.n < n_min:
            continue
        if n_max is not None and spec.n > n_max:
            continue
        if wanted is not None and spec.family not in wanted:
            continue
        yield build_instance(spec)


def scientific_question() -> str:
    return (
        "How do wall-clock time and solution quality (and χ where an MPS/tensor path "
        "applies) scale with system size n for fixed instance families, when solving "
        "via drift.solve (exact→GPU-PT→CPU-PT)?"
    )


def refresh_manifest(path: Path | None = None, *, write_fixtures: bool = True) -> Path:
    """Rebuild fingerprints, known energies, and (optionally) n≤8 fixtures. Writes JSON."""
    p = path or manifest_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    previous: dict[str, dict[str, Any]] = {}
    if p.is_file():
        prev_data = json.loads(p.read_text(encoding="utf-8"))
        previous = {row["id"]: row for row in prev_data.get("instances", [])}

    instances: list[dict[str, Any]] = []
    for d in catalog_rows():
        spec = _spec_from_catalog_dict(d)
        built = build_instance(spec)
        spec.fingerprint = fingerprint(built.model)
        _fill_known_energy(spec, built)
        old = previous.get(spec.id)
        # Keep published known_energy bit-stable when Lanczos/exact only moved ULP noise.
        if (
            old
            and old.get("fingerprint") == spec.fingerprint
            and old.get("known_energy") is not None
            and spec.known_energy is not None
            and abs(float(old["known_energy"]) - float(spec.known_energy)) < 1e-12
        ):
            spec.known_energy = old["known_energy"]
            spec.known_energy_source = old.get("known_energy_source") or spec.known_energy_source
        instances.append(spec.to_json())
        if write_fixtures and spec.n <= 8 and spec.family != "tfim-chain":
            _write_fixture(built)

    payload = {
        "version": BANK_VERSION,
        "question": scientific_question(),
        "mps_n_max_sweep": MPS_N_MAX_SWEEP,
        "mps_applies_to": sorted(MPS_FAMILIES),
        "notes": (
            "IDs are stable inside v1. Generators are seeded. "
            "known_energy is analytic, exact-enumerated (n≤20), or Lanczos (TFIM n≤14). "
            "fingerprint is sha256(J||h)[:16]. Heuristic solves are never certified."
        ),
        "instances": instances,
    }
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return p


def _write_fixture(built: BuiltInstance) -> None:
    dest = fixtures_dir()
    dest.mkdir(parents=True, exist_ok=True)
    spec = built.spec
    payload = {
        "id": spec.id,
        "family": spec.family,
        "n": spec.n,
        "seed": spec.seed,
        "params": spec.params,
        "known_energy": spec.known_energy,
        "known_energy_source": spec.known_energy_source,
        "fingerprint": spec.fingerprint or fingerprint(built.model),
        "J": np.ascontiguousarray(built.model.J, dtype=np.float64).tolist(),
        "h": np.ascontiguousarray(built.model.h, dtype=np.float64).tolist(),
    }
    (dest / f"{spec.id}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def load_fixture(instance_id: str) -> BuiltInstance:
    """Load a checked-in n≤8 fixture (J, h) and wrap it as a BuiltInstance."""
    path = fixtures_dir() / f"{instance_id}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    spec = spec_from_row(data)
    model = IsingModel(
        J=np.asarray(data["J"], dtype=np.float64),
        h=np.asarray(data["h"], dtype=np.float64),
    )
    return BuiltInstance(spec=spec, model=model, extra={})


def main(argv: list[str] | None = None) -> int:
    """Rebuild the checked-in manifest + n≤8 fixtures. Run after catalog_rows() changes."""
    del argv
    path = refresh_manifest()
    n = len(json.loads(path.read_text(encoding="utf-8"))["instances"])
    print(f"wrote {path}  ({n} instances, bank {BANK_VERSION})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
