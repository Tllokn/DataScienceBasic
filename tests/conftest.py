import pytest

from collections import Counter
import networkx as nx
import utils


@pytest.fixture(scope="module", params=range(4))
def test_number_warmup(request):
    return request.param


@pytest.fixture(scope="module")
def warmup_sets(test_number_warmup):
    return {
        0: [0],
        1: [1, 2, 3],
        2: ["A", "B", "C", "D", "E"],
        3: [1, 1, 1, 1, "A", "A", "A"],
    }[test_number_warmup]


@pytest.fixture(scope="module")
def warmup_subset_ks(test_number_warmup):
    return {0: 0, 1: 3, 2: 4, 3: 5}[test_number_warmup]


@pytest.fixture(scope="module", params=range(7))
def test_number_clique(request):
    return request.param


@pytest.fixture(scope="module")
def clique_inputs(test_number_clique):
    vertices = {
        0: [(1, {"weight": 0})],
        1: [(1, {"weight": 1}), (2, {"weight": 1}), (3, {"weight": 1})],
        2: [(1, {"weight": 2}), (2, {"weight": 1}), (3, {"weight": 1})],
        3: [(1, {"weight": 1}), (2, {"weight": 1}), (3, {"weight": 2})],
        4: [
            (1, {"weight": 1}),
            (2, {"weight": 1}),
            (3, {"weight": 1}),
            (4, {"weight": 1}),
            (5, {"weight": 1}),
            (6, {"weight": 1}),
        ],
        5: [
            (-2, {"weight": 100}),
            (-1, {"weight": 1}),
            (1, {"weight": 1}),
            (2, {"weight": 1}),
            (3, {"weight": 1}),
            (4, {"weight": 1}),
        ],
        6: [
            (1, {"weight": 1}),
            (2, {"weight": 2}),
            (3, {"weight": 2}),
            (4, {"weight": 1}),
        ],
    }[test_number_clique]

    edges = {
        0: [],
        1: [(1, 2), (2, 3)],
        2: [(1, 2), (2, 3)],
        3: [(1, 2), (2, 3)],
        4: [(1, 2), (2, 3), (1, 3), (4, 1), (5, 2), (6, 3)],
        5: [(-2, -1), (-1, 1), (1, 2), (2, 3), (1, 3)],
        6: [(1, 2), (2, 3), (3, 4), (4, 1)],
    }[test_number_clique]

    graph = nx.Graph()
    graph.add_nodes_from(vertices)
    graph.add_edges_from(edges)
    return graph


@pytest.fixture(scope="module")
def expected_cliques(test_number_clique):
    return {
        0: [Counter([1])],
        1: [Counter([1, 2]), Counter([2, 3])],
        2: [Counter([1, 2])],
        3: [Counter([2, 3])],
        4: [Counter([1, 2, 3])],
        5: [Counter([-1, -2])],
        6: [Counter([2, 3])],
    }[test_number_clique]


@pytest.fixture(scope="module", params=range(6))
def test_number_path(request):
    return request.param


@pytest.fixture(scope="module")
def path_inputs(test_number_path):
    vertices = {
        0: [1, 2],
        1: [1, 2, 3, 4],
        2: [1, 2, 3, 4],
        3: [1, 2, 3, 4],
        4: [1, 2, 3, 4],
        5: [1, 2, 3, 4, 5, 6],
    }[test_number_path]

    edges = {
        0: [(1, 2, {"weight": 1})],
        1: [(1, 2, {"weight": 1}), (4, 3, {"weight": 2})],
        2: [
            (1, 2, {"weight": 2}),
            (2, 3, {"weight": 1}),
            (3, 4, {"weight": 2}),
            (4, 1, {"weight": 2}),
        ],
        3: [
            (1, 2, {"weight": 1}),
            (2, 3, {"weight": 1}),
            (3, 4, {"weight": 1}),
            (4, 1, {"weight": 1}),
            (1, 3, {"weight": 1}),
            (4, 2, {"weight": 2}),
        ],
        4: [(2, 1, {"weight": 2}), (3, 1, {"weight": 4}), (4, 1, {"weight": 3})],
        5: [
            (1, 3, {"weight": 15}),
            (2, 1, {"weight": 5}),
            (2, 3, {"weight": 10}),
            (2, 5, {"weight": 50}),
            (3, 6, {"weight": 20}),
            (4, 1, {"weight": 10}),
        ],
    }[test_number_path]

    graph = nx.DiGraph()
    graph.add_nodes_from(vertices)
    graph.add_edges_from(edges)
    return graph


@pytest.fixture(scope="module")
def expected_paths(test_number_path):
    return {
        0: [(1, 2)],
        1: [(4, 3)],
        2: [(3, 4, 1, 2)],
        3: [(1, 3, 4, 2)],
        4: [(3, 1)],
        5: [(2, 5)],
    }[test_number_path]


@pytest.fixture(scope="module", params=range(6))
def test_number_knapsack(request):
    return request.param


@pytest.fixture(scope="module")
def knapsack_inputs(test_number_knapsack):
    capacities = {0: 10.1, 1: 0.1, 2: 3.14, 3: 25.0, 4: 3.2, 5: 10.0}[
        test_number_knapsack
    ]

    values = {
        0: [100.0],
        1: [100.0],
        2: [4.17, 2.7, 3.3],
        3: [4.3, 2.8, 1.2],
        4: [20, 30, 40, 50, 60, 70, 80, 90],
        5: [4.5, 2, 10],
    }[test_number_knapsack]

    weights = {
        0: [4],
        1: [2],
        2: [3.13, 1.5, 1.41],
        3: [4.3, 2.8, 1.2],
        4: [0.2, 0.4, 0.7, 0.9, 1.2, 1.6, 1.8, 2.1],
        5: [4, 2, 9],
    }[test_number_knapsack]

    return (capacities, values, weights)


@pytest.fixture(scope="module")
def expected_knapsacks(test_number_knapsack):
    return {
        0: [Counter([0,])],
        1: [Counter([])],
        2: [Counter([1, 2,])],
        3: [Counter([0, 1, 2])],
        4: [Counter([1, 2, 3, 4,])],
        5: [Counter([2,])],
    }[test_number_knapsack]
