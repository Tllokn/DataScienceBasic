from __future__ import annotations

import networkx as nx

import warmup


def max_weighted_clique(graph: nx.Graph) -> tuple:
    """Finds a maximum-weight clique in a graph with non-negative vertex weights.

    Note: We are interested in the clique with maximum total weight.
    This is not necessarily the same as the clique with the maximum number of vertices.

    :param graph: A NetworkX Graph where vertices have a "weight" attribute.
    :return clique: A list or tuple of vertices constituting a largest vertex weighted clique in the graph.
    """
    cliques=list(nx.enumerate_all_cliques(graph))
    max_clique=[]
    max_weight=-65535
    for clique in cliques:
        weight=0
        for vertex in clique:
            weight+=graph.nodes[vertex]['weight']
        if weight>=max_weight:
            max_weight=weight
            max_clique=clique
    return max_clique


def longest_path(graph: nx.Graph) -> tuple:
    """Finds a longest simple path in a graph with non-negative edge weights.

    :param graph: A NetworkX Graph where edges have a "weight" attribute.
    :return path: A list or tuple of vertices constituting a longest path in the graph.
    """
    max_length = 0
    max_path = []
    for source in graph.nodes:
        for target in graph.nodes:
            for path in nx.all_simple_paths(graph,source,target):
                path_weight = 0
                for i in range(len(path)-1):
                    path_weight+=graph.edges[path[i],path[i+1]]['weight']
                if path_weight>=max_length:
                        max_path=tuple(path)
                        max_length=path_weight

    return max_path


def knapsack_01(capacity: float, values: list[float], weights: list[float]) -> tuple[int]:
    """Maximize the value in the knapsack given the item values and capacity constraint.

    Note: We are interested in solving the 0-1 knapsack problem where the capacity and weights are not restricted to integer values.

    :param capacity: The capacity of the knapsack.
    :param values: The values for each item.
    :param weights: The weights for each item.
    :return items: A list or tuple of integers specifying the item numbers/indices making the best value knapsack given the capacity constraints.
    """
    assert len(weights) == len(values)

    item_num=len(weights)
    all_situation=list(warmup.generate_powerset(range(item_num)))
    ideal_situation=[]
    max_value=0
    for situation in all_situation:
        total_value=0
        total_weight=0
        for item in situation:
            total_weight+=weights[item]
            total_value+=values[item]
            if total_weight> capacity:
                break
        if total_weight<=capacity and total_value>max_value:
            max_value=total_value
            ideal_situation=situation

    return tuple(ideal_situation)



