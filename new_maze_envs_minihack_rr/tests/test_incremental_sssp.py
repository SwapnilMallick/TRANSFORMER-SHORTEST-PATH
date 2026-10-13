"""Incrementally maintained cost-to-come (Ramalingam-Reps insertions) vs full Dijkstra."""
import math

import numpy as np
import pytest

import online_cost_transformer as O


def _grow(rng, n_ops, weight):
    """Random roadmap-like growth: new nodes hung off existing ones, extra edges between
    existing nodes, and repeated edges that may lower a weight. Checked after every op."""
    graph = O.Graph((0.0, 0.0), 0.03)
    stats = {}
    for _ in range(n_ops):
        n = len(graph.pos)
        r = rng.random()
        if r < 0.4 or n < 3:
            v = graph.add_node(rng.uniform(0, 1, 2))
            graph.add_edge(int(rng.integers(n)), v, weight(rng))
        else:
            graph.add_edge(int(rng.integers(n)), int(rng.integers(n)), weight(rng))
        graph.check_against_dijkstra(stats)
    return graph, stats


@pytest.mark.parametrize("seed", range(5))
def test_matches_dijkstra_float_weights(seed):
    rng = np.random.default_rng(seed)
    graph, stats = _grow(rng, 600, lambda r: float(r.uniform(0.01, 0.2)))
    assert stats["checks"] == 600
    g, prev = graph.costs()
    assert g == graph.dijkstra(0) and g[0] == 0.0 and prev[0] == -1


@pytest.mark.parametrize("seed", range(5))
def test_matches_dijkstra_with_ties(seed):
    # integer weights force many exactly tied paths: g must still match bit for bit and prev
    # must still be a valid shortest-path tree, even where it differs from Dijkstra's
    rng = np.random.default_rng(100 + seed)
    graph, stats = _grow(rng, 600, lambda r: float(r.integers(1, 4)))
    assert graph.costs()[0] == graph.dijkstra(0)
    assert stats["prev_diffs"] > 0          # ties really occurred, so the check above is not vacuous


def test_unreachable_then_connected():
    graph = O.Graph((0.0, 0.0), 0.03)
    a = graph.add_node((1.0, 0.0))
    b = graph.add_node((2.0, 0.0))
    graph.add_edge(a, b, 1.0)                       # a not reachable yet
    assert math.isinf(graph.g[a]) and math.isinf(graph.g[b])
    graph.add_edge(0, a, 2.0)                       # now both are, through a
    assert graph.g[a] == 2.0 and graph.g[b] == 3.0 and graph.prev[b] == a
    graph.add_edge(0, b, 0.5)                       # shortcut
    assert graph.g[b] == 0.5 and graph.prev[b] == 0
    graph.add_edge(0, a, 5.0)                       # more expensive duplicate is ignored
    assert graph.g[a] == 2.0
    graph.check_against_dijkstra()


def test_costs_returns_copies():
    graph = O.Graph((0.0, 0.0), 0.03)
    g, prev = graph.costs()
    v = graph.add_node((1.0, 0.0))
    graph.add_edge(0, v, 1.0)
    assert g == [0.0] and prev == [-1]
