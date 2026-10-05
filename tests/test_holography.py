"""Phase 19: random tensor networks as Ising models -- checked against oracles, not against itself."""

import numpy as np
import pytest

from drift import holography as H
from drift.solvers.exact import all_configs


def _one_vertex_two_legs():
    return H.BoundaryGraph(1, [], [0, 0], "one-vertex")


@pytest.mark.parametrize("d", [2, 3, 10])
def test_single_vertex_matches_wishart(d):
    # One Gaussian D x D matrix M: E[Tr (MM†)²] = 2D³, E[(Tr MM†)²] = D²(D²+1)  (complex Wishart).
    expected = -np.log(2 * d**3 / (d**2 * (d**2 + 1)))
    assert H.renyi2(_one_vertex_two_legs(), (0,), d) == pytest.approx(expected, abs=1e-12)


def test_ising_energies_are_log_d_times_the_wall_length():
    g = H.square_grid(2, 3)
    region = (0, 1)
    log_d = np.log(5.0)
    model = H.ising_model(g, region, log_d)
    spins = all_configs(g.n)                       # all_configs: bit 1 -> spin -1, bit k -> spin k
    e_drift = model.energy_batch(spins)
    cost = H.domain_wall_costs(g, region).astype(float)
    diff = e_drift - log_d * cost
    assert np.ptp(diff) < 1e-9                     # equal up to one constant


@pytest.mark.parametrize("g", [H.square_grid(3, 4), H.hyperbolic_rings(3), H.random_regular(12, 3, 8, seed=4)])
def test_max_flow_equals_enumerated_ground_state(g):
    for region in H.intervals(g.n_legs):
        wall, _ = H.ground_state_degeneracy(g, region)
        assert H.min_cut(g, region) == wall, region


def test_the_bulk_decides_when_it_is_cheaper():
    # On the small graphs above every minimal cut is A's own legs (one leg per boundary vertex costs
    # as much as a bond), so they cannot see a wrong bond capacity. Two vertices with three legs
    # each, joined by one bond: separating one vertex's legs costs the bond (1), not the legs (3).
    g = H.BoundaryGraph(2, [(0, 1)], [0, 0, 0, 1, 1, 1], "dumbbell")
    assert H.min_cut(g, (0, 1, 2)) == 1
    assert H.ground_state_degeneracy(g, (0, 1, 2)) == (1, 1)


def test_large_d_is_ryu_takayanagi_up_to_degeneracy():
    g = H.square_grid(3, 4)
    d = 1e6
    for region in H.intervals(g.n_legs):
        wall, deg = H.ground_state_degeneracy(g, region)
        s2 = H.renyi2(g, region, d)
        assert s2 - wall * np.log(d) == pytest.approx(-np.log(deg), abs=1e-4)


def test_monte_carlo_ratio_of_averages_on_one_vertex():
    rng = np.random.default_rng(0)
    d, n = 3, 6000
    g = _one_vertex_two_legs()
    num = den = 0.0
    for _ in range(n):
        psi = H.random_boundary_state(g, d, rng)
        sv2 = np.linalg.svd(psi, compute_uv=False) ** 2
        num += (sv2**2).sum()
        den += sv2.sum() ** 2
    assert -np.log(num / den) == pytest.approx(H.renyi2(g, (0,), d), rel=0.03)


def test_product_state_has_zero_entropy():
    state = np.einsum("i,j,k->ijk", *[np.random.default_rng(i).standard_normal(3) for i in range(3)])
    assert H.state_renyi2(state, (0,)) == pytest.approx(0.0, abs=1e-12)


def test_graph_shapes():
    g = H.hyperbolic_rings(3)
    assert g.n == 3 + 6 + 12 and g.n_legs == 12
    assert len(g.bonds) == (3 + 6 + 12) + 2 * (3 + 6)
    s = H.square_grid(4, 5)
    assert s.n == 20 and s.n_legs == 2 * (4 + 5) - 4 and len(set(s.legs)) == s.n_legs
