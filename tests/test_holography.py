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


# ── phase 20: the bulk from the boundary ─────────────────────────────────────────────────────────
import itertools


def _random_metric(n, seed):
    rng = np.random.default_rng(seed)
    d = rng.integers(1, 9, size=(n, n)).astype(float)
    d = d + d.T
    np.fill_diagonal(d, 0.0)
    return d


@pytest.mark.parametrize("seed", range(4))
def test_circular_splits_rebuild_any_symmetric_table(seed):
    # The n(n-1)/2 circular splits are a basis: the rebuild is exact whatever the signs of α.
    d = _random_metric(9, seed)
    assert np.allclose(H.rebuild_distances(H.split_weights(d)), d)


def test_split_weights_are_non_negative_on_a_min_cut_table():
    # Strong subadditivity of minimal cuts: every α is a conditional mutual information >= 0.
    g = H.square_grid(4, 5)
    a = H.split_weights(H.interval_table(g))
    assert a.min() >= 0.0


def test_cycle_metric_has_only_diametric_splits():
    n = 12
    d = np.array([[min(abs(i - j), n - abs(i - j)) for j in range(n)] for i in range(n)], float)
    a = H.split_weights(d)
    lengths = {(j - i) % n for i in range(n) for j in range(n) if a[i, j] > 1e-12}
    assert lengths == {n // 2}
    # α(i, i+n/2) = ½(n/2 + n/2 - 2(n/2 - 1)) = 1 for every i
    assert H.kinematic_density(a)[n // 2] == pytest.approx(float(n))


def test_tree_metric_is_zero_hyperbolic_and_fully_laminar():
    # A star with a long internal edge: leaves 0..5 in circular order, {1,2} and {4,5} hang off
    # internal nodes. Tree metrics have δ = 0 and every split weight on a non-crossing family.
    n = 6
    depth = {0: 1, 1: 2, 2: 2, 3: 1, 4: 3, 5: 3}
    group = {0: "r", 1: "p", 2: "p", 3: "r", 4: "q", 5: "q"}
    hub = {"r": 0, "p": 1, "q": 2}                         # distance from the root to each hub
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if group[i] == group[j] and group[i] != "r":
                d[i, j] = (depth[i] - hub[group[i]]) + (depth[j] - hub[group[j]])
            else:
                d[i, j] = depth[i] + depth[j]
    assert H.gromov_delta(d)[0] == pytest.approx(0.0)
    a = H.split_weights(d)
    assert a.min() >= -1e-12
    w = H.arc_weights(a)
    best, arcs = H.max_laminar(w)
    assert best == pytest.approx(w.sum())                  # treeness 1
    assert not any(H.crossing(p, q) for p, q in itertools.combinations(arcs, 2))


def _brute_laminar(w):
    arcs = [(a, b) for a in range(1, w.shape[0]) for b in range(a, w.shape[0]) if w[a, b] > 0]
    best = 0.0
    for k in range(len(arcs) + 1):
        for sub in itertools.combinations(arcs, k):
            if any(H.crossing(p, q) for p, q in itertools.combinations(sub, 2)):
                continue
            best = max(best, sum(w[p] for p in sub))
    return best


@pytest.mark.parametrize("seed", range(6))
def test_laminar_dp_equals_brute_force(seed):
    rng = np.random.default_rng(seed)
    n = 7
    w = np.zeros((n, n))
    for a in range(1, n):
        for b in range(a, n):
            if rng.random() < 0.6:
                w[a, b] = rng.integers(1, 10)
    best, arcs = H.max_laminar(w)
    assert best == pytest.approx(_brute_laminar(w))
    assert sum(w[p] for p in arcs) == pytest.approx(best)  # the traceback is the optimum
    assert not any(H.crossing(p, q) for p, q in itertools.combinations(arcs, 2))


def test_crossing_and_nesting():
    assert H.crossing((1, 4), (3, 6)) and H.crossing((3, 6), (1, 4))
    assert not H.crossing((1, 6), (2, 3)) and not H.crossing((1, 2), (3, 4))
    assert not H.crossing((1, 3), (4, 6)) and H.crossing((1, 3), (3, 6))   # they share gap 3
    assert H.nesting_depth([(1, 8), (2, 7), (3, 4), (5, 6), (3, 3)]) == 4


def test_four_point_delta_of_a_cycle():
    # C_n with n = 4k: the quadruple 0, k, 2k, 3k gives sums 2k, 2k, 4k -> δ = k, diameter 2k.
    n = 16
    d = np.array([[min(abs(i - j), n - abs(i - j)) for j in range(n)] for i in range(n)], float)
    assert H.gromov_delta(d) == (4.0, 8.0)


# ── phase 21: a bulk event seen from the boundary ────────────────────────────────────────────────
def _perturbed(g, edge, mode):
    """The graph with one bulk edge removed (mode 'drop') or doubled (mode 'double')."""
    bonds = [b for b in g.bonds if (min(b), max(b)) != edge]
    k = len(g.bonds) - len(bonds)                       # its multiplicity
    bonds += [edge] * (k - 1 if mode == "drop" else k + 1)
    return H.BoundaryGraph(g.n, bonds, g.legs, g.name)


@pytest.mark.parametrize("g", [H.square_grid(4, 5), H.hyperbolic_rings(4), H.random_regular(16, 3, 10, seed=3),
                               H.BoundaryGraph(4, [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)], [0, 0, 1, 2, 2, 3])])
def test_membership_equals_direct_perturbation(g):
    edges = H.bond_edges(g)
    for i in range(g.n_legs):
        for j in range(i + 1, g.n_legs + 1):
            region = tuple(range(i, j))
            gamma, some, every = H.cut_membership(g, region)
            assert gamma == H.min_cut(g, region)
            for k, e in enumerate(edges):
                drop = H.min_cut(_perturbed(g, e, "drop"), region)     # one unit of capacity less
                dbl = H.min_cut(_perturbed(g, e, "double"), region)    # one unit more
                assert some[k] == (drop == gamma - 1), (region, e)
                assert every[k] == (dbl == gamma + 1), (region, e)


def _random_multigraph(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(3, 8))
    bonds = []
    for a in range(n):
        for b in range(a + 1, n):
            if rng.random() < 0.55:
                bonds += [(a, b)] * int(rng.integers(1, 4))
    legs = [int(x) for x in rng.integers(0, n, size=int(rng.integers(4, 8)))]
    return H.BoundaryGraph(n, bonds, legs, f"multi-{seed}")


def test_membership_on_random_multigraphs():
    # Small multigraphs with parallel bonds and several legs per vertex reach the residual
    # configurations the regular graphs above never produce.
    for seed in range(150):
        g = _random_multigraph(seed)
        edges = H.bond_edges(g)
        for i in range(g.n_legs):
            for j in range(i + 1, g.n_legs):
                region = tuple(range(i, j))
                gamma, some, every = H.cut_membership(g, region)
                for k, e in enumerate(edges):
                    assert some[k] == (H.min_cut(_perturbed(g, e, "drop"), region) == gamma - 1), (seed, region, e)
                    assert every[k] == (H.min_cut(_perturbed(g, e, "double"), region) == gamma + 1), (seed, region, e)


def test_every_implies_some():
    g = H.hyperbolic_rings(5)
    for region in H.intervals(g.n_legs)[::3]:
        _, some, every = H.cut_membership(g, region)
        assert not (every & ~some).any()


def test_dumbbell_bond_is_on_every_cut():
    g = H.BoundaryGraph(2, [(0, 1)], [0, 0, 0, 1, 1, 1], "dumbbell")
    gamma, some, every = H.cut_membership(g, (0, 1, 2))
    assert gamma == 1 and some[0] and every[0]
    gamma, some, every = H.cut_membership(g, (0,))                  # one leg is cheaper than the bond
    assert gamma == 1 and not some[0] and not every[0]


def test_ring_of():
    g = H.hyperbolic_rings(3)
    assert list(H.ring_of(g)) == [0] * 3 + [1] * 6 + [2] * 12
