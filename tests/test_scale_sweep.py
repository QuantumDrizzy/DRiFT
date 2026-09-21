"""Validation of the versioned instance bank and the scale sweep.

Falsifiers:
  * the catalog loads, IDs are unique, generators are seeded and match fingerprints;
  * a tiny n≤10 sweep finishes;
  * certified is True only for the exact engine — never for cpu-pt / mps.

Run:  pytest tests/test_scale_sweep.py -q
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from drift import gpu
from drift.benchmarks import (
    BANK_VERSION,
    CLASSICAL_N_LADDER,
    FAMILIES,
    MPS_FAMILIES,
    build_instance,
    extra_seeds_for,
    fingerprint,
    fixtures_dir,
    iter_instances,
    load_fixture,
    load_manifest,
    make_id,
    manifest_path,
    scientific_question,
    unknown_family,
)
from drift.benchmarks.bank import spec_from_row
from drift.solvers.exact import exact_ground_state
from drift.solve import solve
from experiments.scale_sweep import (
    SweepMeta,
    run_sweep,
    sweep_instance,
    write_markdown_report,
    write_results,
)


def test_bank_loads_with_stable_ids():
    """The checked-in manifest is v1, IDs are unique, and every family is present."""
    specs = load_manifest()
    assert specs, "empty instance bank"
    ids = [s.id for s in specs]
    assert len(ids) == len(set(ids))
    assert all(s.id.startswith(f"{BANK_VERSION}.") for s in specs)
    present = {s.family for s in specs}
    assert present == set(FAMILIES)
    data = json.loads(manifest_path().read_text(encoding="utf-8"))
    assert data["version"] == BANK_VERSION
    assert "wall-clock" in data["question"]
    assert data["question"] == scientific_question()


def test_generators_are_deterministic_and_match_fingerprints():
    """Same (family, n, seed) rebuilds the same (J, h); catalog fingerprints pin that."""
    specs = [s for s in load_manifest() if s.n <= 12]
    assert specs
    for spec in specs:
        a = build_instance(spec)
        b = build_instance(spec)
        assert np.allclose(a.model.J, b.model.J)
        assert np.allclose(a.model.h, b.model.h)
        assert fingerprint(a.model) == spec.fingerprint
        assert a.spec.id == make_id(spec.family, spec.n, spec.seed)


def test_small_fixtures_match_generators():
    """Checked-in n≤8 JSON fixtures are the same Hamiltonians the generators emit."""
    paths = sorted(fixtures_dir().glob("v1.*.json"))
    assert paths, "expected checked-in fixtures under drift/benchmarks/instances/fixtures/"
    for path in paths:
        fix = load_fixture(path.stem)
        gen = build_instance(fix.spec)
        assert np.allclose(fix.model.J, gen.model.J)
        assert np.allclose(fix.model.h, gen.model.h)
        if fix.spec.known_energy is not None:
            _, e, _ = exact_ground_state(fix.model)
            assert e == pytest.approx(fix.spec.known_energy)


def test_analytic_oracles_match_exact_on_small_n():
    """Ferro, bipartite, and crystal known energies agree with exact enumeration."""
    wanted = ("ferro-chain", "bipartite-maxcut", "crystal")
    for inst in iter_instances(n_max=16, families=wanted):
        if inst.spec.n > 16:
            continue
        _, e, _ = exact_ground_state(inst.model, max_n=16)
        assert inst.spec.known_energy is not None
        assert e == pytest.approx(inst.spec.known_energy), inst.spec.id


def test_tiny_sweep_n_le_10_finishes(tmp_path):
    """CI-sized sweep: every n≤10 classical instance solves, and we write results."""
    rows = run_sweep(n_max=10, skip_mps=True, use_gpu=False, n_rounds=20)
    assert rows
    assert all(r.n <= 10 for r in rows)
    assert all(r.method != "skipped" for r in rows)
    csv_path, json_path = write_results(rows, tmp_path)
    assert csv_path.is_file() and json_path.is_file()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["n_rows"] == len(rows)
    assert "drift.solve" in payload["question"]


def test_certified_true_only_for_exact():
    """Honesty: certified is True only when the exact engine ran."""
    rows = run_sweep(n_max=10, skip_mps=True, use_gpu=False, exact_max=18)
    assert any(r.method == "exact" and r.certified for r in rows)
    for r in rows:
        if r.method == "exact":
            assert r.certified is True, r.id
        else:
            assert r.certified is False, r.id


def test_heuristic_is_never_certified():
    """Forcing CPU-PT past exact_max never sets certified, even if the energy is right."""
    spec = next(s for s in load_manifest() if s.family == "ferro-chain" and s.n == 8)
    inst = build_instance(spec)
    row = sweep_instance(inst, exact_max=6, use_gpu=False, n_replicas=8, n_rounds=40)
    assert row.method == "cpu-pt"
    assert row.certified is False
    sol = solve(inst.model, exact_max=6, use_gpu=False, n_replicas=8, n_rounds=40, seed=0)
    assert sol.method == "cpu-pt" and sol.certified is False


def test_mps_path_is_not_certified():
    """χ is measured on the TFIM family; the MPS energy is variational, never certified."""
    spec = next(s for s in load_manifest() if s.family in MPS_FAMILIES and s.n == 8)
    inst = build_instance(spec)
    row = sweep_instance(inst, mps_n_max=8, mps_max_sweeps=6, mps_chi_max=8)
    assert row.method == "mps"
    assert row.certified is False
    assert row.chi is not None and row.chi >= 1


def test_unknown_family_is_exhaustive():
    """A new family name must be added to the switch, not silently ignored."""
    with pytest.raises(ValueError, match="unknown instance family"):
        unknown_family("not-a-family")


def test_fingerprint_mismatch_is_loud():
    """Tampering with a catalog fingerprint must fail closed."""
    spec = load_manifest()[0]
    bad = spec_from_row({**spec.to_json(), "fingerprint": "deadbeefdeadbeef"})
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        build_instance(bad)


PR4_CLASSICAL_N = (8, 10, 12, 14, 16, 18, 20, 24, 32)
PR4_PRIMARY_IDS = (
    "v1.maxcut-er.n008.s001",
    "v1.pmj-glass.n008.s007",
    "v1.bipartite-maxcut.n008.s002",
    "v1.ferro-chain.n008.s000",
    "v1.crystal.n016.s000",
    "v1.crystal.n032.s000",
    "v1.tfim-chain.n008.s000",
)


def test_bank_is_broader_than_pr4_ladder():
    """v1 grew by adding IDs: denser n (through 40) and extra seeds. PR #4 IDs remain."""
    specs = load_manifest()
    ids = {s.id for s in specs}
    for iid in PR4_PRIMARY_IDS:
        assert iid in ids, iid
    er_n = {s.n for s in specs if s.family == "maxcut-er"}
    assert set(PR4_CLASSICAL_N).issubset(er_n)
    assert 40 in er_n
    assert CLASSICAL_N_LADDER[-1] == 40
    assert len(CLASSICAL_N_LADDER) > len(PR4_CLASSICAL_N)
    er8 = [s for s in specs if s.family == "maxcut-er" and s.n == 8]
    assert len(er8) >= 2
    ferro_seeds = {s.seed for s in specs if s.family == "ferro-chain"}
    assert ferro_seeds == {0}


def test_ferro_chain_is_not_duplicated_by_seed():
    """The open ferro chain ignores seed; extra seeds would be the same Hamiltonian."""
    assert extra_seeds_for("ferro-chain", 8) == ()
    assert extra_seeds_for("maxcut-er", 8)
    assert extra_seeds_for("maxcut-er", 32) == ()


def test_ci_sized_sweep_stays_small_and_never_fakes_gpu_pt(tmp_path):
    """--ci envelope: n≤10, extra seeds allowed, no gpu-pt without a binary."""
    rows = run_sweep(n_max=10, skip_mps=True, use_gpu=True, n_rounds=20)
    assert rows
    assert all(r.n <= 10 for r in rows)
    assert all(r.method in ("exact", "cpu-pt") for r in rows)
    if not gpu.gpu_available():
        assert all(r.method != "gpu-pt" for r in rows)
    er = [r for r in rows if r.family == "maxcut-er"]
    assert len(er) >= 2
    n_zero = sum(
        1 for r in rows
        if r.energy_error is not None and r.method != "mps" and abs(float(r.energy_error)) < 1e-9
    )
    text = write_markdown_report(
        rows,
        tmp_path / "SCALE-sweep.md",
        meta=SweepMeta(
            profile="ci", n_min=1, n_max=10, exact_max=18, n_rounds=20,
            use_gpu_flag=True, gpu_available=gpu.gpu_available(),
            skip_mps=True, mps_n_max=16, command="pytest",
        ),
    ).read_text(encoding="utf-8")
    assert f"Classical rows with dE = 0: **{n_zero}**" in text
    assert n_zero == len(rows)  # n≤10 classical exact, every row has an oracle
    if not gpu.gpu_available():
        assert "gpu-pt rows in this file: none" in text
        assert all(r.method != "gpu-pt" for r in rows)


def test_heuristic_pt_past_exact_max_is_cpu_when_gpu_missing():
    """Forcing a solve past exact_max without a binary records cpu-pt, never gpu-pt."""
    spec = next(s for s in load_manifest() if s.family == "ferro-chain" and s.n == 20)
    inst = build_instance(spec)
    row = sweep_instance(inst, exact_max=18, use_gpu=True, n_replicas=8, n_rounds=20)
    if gpu.gpu_available():
        assert row.method in ("gpu-pt", "cpu-pt")
    else:
        assert row.method == "cpu-pt"
    assert row.certified is False
