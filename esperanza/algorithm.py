# Version: v0.2.1
# Modified on 04/04/2026
# Author: Frank Vega

import itertools
import networkx as nx
from hvala.algorithm import find_vertex_cover

def maximize_solution(G: nx.Graph, S: set):
    """
    Repair a candidate set into an independent set and greedily maximize it.
    
    By maintaining an explicit `blocked` hash set of neighbors, candidate node
    validity is checked in O(1) time and neighbor sets are updated only upon 
    adding a vertex. 

    Phase 2 uses degree-based sorting to dodge high-degree "trap" vertices 
    while operating in O(n log n + m) time per call instead of O(n^2).

    Args:
        G (nx.Graph): An undirected NetworkX graph.
        S (set): Candidate node set (may contain conflicts).

    Returns:
        set: A maximal independent set of G.
    """
    independent = set()
    blocked = set()
    
    # --- Phase 1: Fast repair ---
    # Respect perturbation starting point from candidate set S
    for u in S:
        if u not in blocked:
            independent.add(u)
            blocked.update(G[u])
            
    # --- Phase 2: Greedily maximize with degree-awareness ---
    remaining_nodes = [
        u for u in G.nodes() 
        if u not in independent and u not in blocked
    ]
    
    # Sort remaining candidates by degree in ascending order
    remaining_nodes.sort(key=lambda node: G.degree(node))
    
    for u in remaining_nodes:
        if u not in blocked:
            independent.add(u)
            blocked.update(G[u])
            
    return independent


def find_independent_set(graph):
    """
    Compute an approximate maximum independent set.

    Args:
        graph (nx.Graph): An undirected NetworkX graph.

    Returns:
        set: A maximal independent set of vertices (approximate maximum).
    """
    
    if not isinstance(graph, nx.Graph):
        raise ValueError("Input must be an undirected NetworkX Graph.")

    if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
        return set(graph.nodes())

    working_graph = graph.copy()
    working_graph.remove_edges_from(list(nx.selfloop_edges(working_graph)))

    isolates = set(nx.isolates(working_graph))
    working_graph.remove_nodes_from(isolates)

    if working_graph.number_of_nodes() == 0:
        return isolates

    cover = find_vertex_cover(working_graph)
    nodes = set(working_graph)
    approximate_independent_set = nodes - cover
    for u in cover:
        candidate = (cover - {u}) | set(working_graph.neighbors(u))
        iset = nodes - candidate
        solution = maximize_solution(working_graph, iset)
        if len(solution) >= len(approximate_independent_set):
            approximate_independent_set = solution

    approximate_independent_set.update(isolates)

    return approximate_independent_set


def find_independent_set_brute_force(graph):
    """
    Computes an exact independent set in exponential time.

    Args:
        graph: A NetworkX Graph.

    Returns:
        A set of vertex indices representing the exact Independent Set, or None if the graph is empty.
    """
    def is_independent_set(graph, independent_set):
        for u in independent_set:
            for v in independent_set:
                if u != v and graph.has_edge(u, v):
                    return False
        return True
    
    if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
        return None

    n_vertices = len(graph.nodes())

    n_max_vertices = 0
    best_solution = None

    for k in range(1, n_vertices + 1):
        for candidate in itertools.combinations(graph.nodes(), k):
            cover_candidate = set(candidate)
            if is_independent_set(graph, cover_candidate) and len(cover_candidate) > n_max_vertices:
                n_max_vertices = len(cover_candidate)
                best_solution = cover_candidate
                
    return best_solution


def find_independent_set_approximation(graph):
    """
    Computes an approximate Independent Set in polynomial time with an 
    approximation ratio of at most 2 for undirected graphs.

    Args:
        graph: A NetworkX Graph.

    Returns:
        A set of vertex indices representing the approximate Independent Set, or None if empty.
    """

    if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
        return None

    complement_graph = nx.complement(graph)
    independent_set = nx.approximation.max_clique(complement_graph)
    return independent_set