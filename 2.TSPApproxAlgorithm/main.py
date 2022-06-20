from __future__ import annotations

from typing import Hashable

import networkx as nx

import time

import matplotlib.pyplot as plt

from utils import check_cycle, generate_metric_graph


def christofides(graph: nx.Graph) -> list[Hashable]:
    """Approximates a solution to the metric TSP using Christofides' algorithm.

    Parameters
    ----------
    graph : nx.Graph
        Weighted graph where weight function is metric.

    Returns
    -------
    cycle : list[Hashable]
        A Hamiltonian cycle with total weight within 3/2 of the minimum.
        The list should begin and end with the same node, e.g. [u, v, w, u] instead of [u, v, w].

    Examples
    --------
    >>> g = generate_metric_graph(6, seed=42)
    >>> cycle = christofides(g)
    >>> check_cycle(g, cycle, optimum=18, approximation_ratio=3/2)
    True
    """
    #1.Create a minimum spanning tree T of G
    MST=nx.minimum_spanning_tree(graph)

    #2.Let O be the set of vertices with odd degree in T. By the handshaking lemma, O has an even number of vertices.
    O=graph.copy()
    # O.remove_nodes_from([v for v, degree in MST.degree if not (degree % 2)])
    not_O_node=[]
    for v, deg in MST.degree:
        if deg%2 ==0:
            not_O_node.append(v)
    O.remove_nodes_from(not_O_node)

    #3.Find a minimum-weight perfect matching M in the induced subgraph given by the vertices from O.
    matching_edges=nx.min_weight_matching(O,maxcardinality=True)

    #4.Combine the edges of M and T to form a connected multigraph H in which each vertex has even degree.
    H=nx.MultiGraph()
    H.add_edges_from(MST.edges)
    H.add_edges_from(matching_edges)

    #5.Form an Eulerian circuit in H.
    EC=nx.eulerian_circuit(H)

    #6.Make the circuit found in previous step into a Hamiltonian circuit by skipping repeated vertices
    TSPC=[]
    for u,v in EC:
        if v in TSPC:
            continue
        if not TSPC:
            TSPC.append(u)
        TSPC.append(v)
    TSPC.append(TSPC[0])

    return TSPC




def two_opt(graph: nx.Graph) -> list[Hashable]:
    """Approximates a solution to the metric TSP using the 2-opt algorithm.

    Parameters
    ----------
    graph : nx.Graph
        Weighted graph where weight function is metric.

    Returns
    -------
    cycle : list[Hashable]
        A Hamiltonian cycle with total weight within sqrt(|V|/2) of the minimum.
        The list should begin and end with the same node, e.g. [u, v, w, u] instead of [u, v, w].

    Examples
    --------
    >>> g = generate_metric_graph(6, seed=42)
    >>> cycle = two_opt(g)
    >>> check_cycle(g, cycle, optimum=18, approximation_ratio=3**0.5)
    True
    """

    """
    This algorithm works, but total more than 6 minutes, plz wait a while.
    """


    best_route=[]
    for node in graph.nodes:
        best_route.append(node)
    best_route.append(best_route[0])

    route=best_route.copy()

    best_length=nx.path_weight(graph,route,"weight")

    optimized=True
    while optimized:
        optimized=False
        for i in range(1,len(best_route)-2):
            for j in range(i+1,len(best_route)):
                if j-i==1:
                    continue
                new_route=best_route[:]
                new_route[i:j]=best_route[j-1:i-1:-1]
                new_length=nx.path_weight(graph,new_route,"weight")
                if new_length < best_length:
                    optimized=True
                    best_route=new_route
                    best_length=new_length
    return best_route



if __name__ == "__main__":
    import doctest

    doctest.testmod()


    """
    test graph from node size 2-40, compare the running time and TSP distance for both algorithm
    """

    graph_size=[i for i in range(2,42,2)]

    cr_time=[]
    to_time=[]
    cr_length=[]
    to_length=[]

    for size in graph_size:
        g=generate_metric_graph(size)

        start_time=time.time()
        cr_route=christofides(g)
        run_time=time.time()-start_time
        cr_time.append(run_time)
        cr_length.append(nx.path_weight(g,cr_route,"weight"))

        start_time = time.time()
        to_route = two_opt(g)
        run_time = time.time() - start_time
        to_time.append(run_time)
        to_length.append(nx.path_weight(g, to_route, "weight"))


    print("grapg_size:",graph_size)
    print("cr time:",cr_time)
    print("cr distance:", cr_length)
    print("to time",to_time)
    print("to distance",to_length)

    plt.ylabel('Running Time')
    plt.xlabel('Graph Vertices Size')

    A1, = plt.plot(graph_size, to_time, "bo-", linewidth=2, markersize=12, label="2-Opt")
    B1, = plt.plot(graph_size, cr_time, "ro-", linewidth=2, markersize=12, label="Christofides")

    legend = plt.legend(handles=[A1, B1])

    plt.figure()

    plt.ylabel('Min Distance')
    plt.xlabel('Graph Vertices Size')

    A2, = plt.plot(graph_size, to_length, "bo-", linewidth=2, markersize=12, label="2-Opt")
    B2, = plt.plot(graph_size, cr_length, "ro-", linewidth=2, markersize=12, label="Christofides")

    legend = plt.legend(handles=[A2, B2])

    plt.show()
