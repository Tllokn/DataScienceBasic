from __future__ import annotations

from typing import Hashable, Sequence

import networkx as nx
import numpy as np


def generate_metric_graph(n: int, p: int = 1, seed: int | None = None) -> nx.Graph:
    """Generates a complete undirected graph with metric weights using an Lp norm.

    Parameters
    ----------
    n : int
        Number of nodes.
    p : int
        Order of norm: 1 for L1 (Manhattan distance), 2 for L2 (Euclidean distance), etc.
    seed : int | None
        Seed for PRNG.

    Returns
    -------
    graph : nx.Graph
        Complete weighted graph with metric weight function.
    """
    rng = np.random.default_rng(seed)
    pairs: set[tuple[int, int]] = set()
    while len(pairs) < n:
        pairs.add(tuple(rng.integers(size=2, low=1, high=n, endpoint=True)))
    pairs: list[tuple[int, int]] = sorted(pairs)
    graph = nx.complete_graph(pairs)
    points = [np.array(pair) for pair in pairs]
    for i in range(n):
        u = pairs[i]
        for j in range(i + 1, n):
            v = pairs[j]
            graph[u][v]["weight"] = int(np.linalg.norm(points[i] - points[j], ord=p))
    return graph


def is_hamiltonian(graph: nx.Graph, cycle: Sequence[Hashable]) -> bool:
    """Checks that the cycle is Hamiltonian.

    Parameters
    ----------
    graph : nx.Graph
        NetworkX graph.
    cycle : Sequence[Hashable]
        Sequence of nodes forming a cycle in the graph.
        The list should begin and end with the same node, e.g. [u, v, w, u] instead of [u, v, w].

    Returns
    -------
    hamiltonian : bool
        True if the node sequence forms a Hamiltonian cycle, False otherwise.
    """
    n = graph.number_of_nodes()
    return all(
        [
            # Starts and ends at same node
            cycle[0] == cycle[-1],
            # Visits n vertices
            len(cycle) == n + 1,
            # No repeat vertices (simple cycle)
            len(set(cycle)) == n,
            # All edges exist
            all((u, v) in graph.edges for u, v in zip(cycle, cycle[1:])),
        ]
    )


def check_cycle(
    graph: nx.Graph,
    cycle: Sequence[Hashable],
    optimum: float,
    approximation_ratio: float,
) -> bool:
    """Checks that a Hamiltonian cycle meets the given approximation ratio where optimum is known.

    Parameters
    ----------
    graph : nx.Graph
        Weighted graph.
    cycle : Sequence[Hashable]
        Sequence of nodes forming an approximately min-weight Hamiltonian cycle.
        The list should begin and end with the same node, e.g. [u, v, w, u] instead of [u, v, w].
    optimum : float
        Minimum total weight of any Hamiltonian cycle in the graph.
    approximation_ratio : float
        Approximation ratio which must be satisfied, e.g. 3/2 for Christofides' algorithm.

    Returns
    -------
    valid : bool
        True if the cycle is Hamiltonian and satisfies the approximation ratio, False otherwise.
    """
    if not is_hamiltonian(graph, cycle):
        return False
    weight = sum(graph[u][v]["weight"] for u, v in zip(cycle, cycle[1:]))
    return weight <= optimum * approximation_ratio


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    seed = 42
    g = generate_metric_graph(6, seed=seed)
    pos = {u: u for u in g.nodes}
    nx.draw_networkx(g, pos, node_size=500, font_size=8, font_color="white")
    nx.draw_networkx_edge_labels(
        g, pos, edge_labels=nx.get_edge_attributes(g, "weight")
    )
    plt.savefig("example.png")
