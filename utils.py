from __future__ import annotations

import networkx as nx
import numpy as np


def gen_edge_weighted_graph(
    n: int, p=0.5, min_wt=0, max_wt=1, seed=None, directed=False
) -> nx.Graph:
    """Generates a graph with edge weights uniformly distributed between min_wt and max_wt.

    :param n: Number of vertices
    :param p: Edge probability
    :param min_wt: Minimum edge weight
    :param max_wt: Maximum edge weight
    :param seed: Seed for PRNG
    :param directed: Whether output graph should be directed or undirected
    :return graph: Graph with weighted edges on n vertices w/ edge prob. p
    """
    graph = nx.gnp_random_graph(n, p, seed=seed, directed=directed)
    rng = np.random.default_rng(seed)

    for u, v, data in graph.edges(data=True):
        data["weight"] = rng.uniform(low=min_wt, high=max_wt)

    return graph


def gen_vtx_weighted_graph(
    n: int, p=0.5, min_wt=0, max_wt=1, seed=None, directed=False
) -> nx.Graph:
    """Generates a graph with vertex weights uniformly distributed between min_wt and max_wt.

    :param n: Number of vertices
    :param p: Edge probability
    :param min_wt: Minimum edge weight
    :param max_wt: Maximum edge weight
    :param seed: Seed for PRNG
    :param directed: Whether output graph should be directed or undirected
    :return: SimpleGraph on n vertices w/ edge prob. p, and weighted vertices
    """
    graph = nx.gnp_random_graph(n, p, seed=seed, directed=directed)

    rng = np.random.default_rng(seed)
    for u, data in graph.nodes(data=True):
        data["weight"] = rng.uniform(low=min_wt, high=max_wt)

    return graph


def gen_knapsack_instance(
    n, capacity=10, min_wt=0, max_wt=1, min_val=0, max_val=1, seed=None
) -> tuple[float, np.ndarray, np.ndarray]:
    """Generates a knapsack instance with n items.

    Values are sampled uniformly at random from min_val to max_val.
    Weights are sampled uniformly at random from min_wt to max_wt.

    :param n: Number of items
    :param capacity: Knapsack capacity
    :param min_wt: Minimum item weight
    :param max_wt: Maximum item weight
    :param min_val: Minimum item value
    :param max_val: Maximum item value
    :param seed: Seed for PRNG
    :return capacity, values, weights:
    """
    rng = np.random.default_rng(seed)
    values = rng.uniform(min_val, max_val, n)
    weights = rng.uniform(min_wt, max_wt, n)
    return capacity, values, weights
