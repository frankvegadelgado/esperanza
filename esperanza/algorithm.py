# Created on 10/12/2025
# Author: Frank Vega

import itertools

import networkx as nx

from hvala.algorithm import find_vertex_cover

def maximize_solution(G: nx.Graph, S: set):
    """
    Repair a candidate set into an independent set and greedily maximize it.
    
    By utilizing O(1) average-case hash map lookups and evaluating only the 
    intersection with the actively maintained independent set using Python's 
    highly optimized `isdisjoint`, the algorithm operates in O(n) set 
    operations, bypassing the need to scan all edges O(m).

    Args:
        G (nx.Graph): An undirected NetworkX graph.
        S (set): Candidate node set (may contain conflicts).

    Returns:
        set: A maximal independent set of G.
    """
    independent = set()
    
    # --- Phase 1: Fast repair ---
    # Keep nodes from S that do not conflict with already kept nodes
    for u in S:
        # G[u] returns a dictionary-like adjacency view. 
        # isdisjoint evaluates lazily and breaks early.
        if independent.isdisjoint(G[u]):
            independent.add(u)
            
    # --- Phase 2: Greedily maximize ---
    for u in G.nodes():
        if u not in independent and independent.isdisjoint(G[u]):
            independent.add(u)
            
    return independent

def find_independent_set(graph):
    """
    Compute an approximate maximum independent set

    Args:
        graph (nx.Graph): An undirected NetworkX graph.

    Returns:
        set: A maximal independent set of vertices (approximate maximum).
    """
    
    # Validate that the input is an undirected simple graph from NetworkX
    if not isinstance(graph, nx.Graph):
        raise ValueError("Input must be an undirected NetworkX Graph.")

    # Trivial cases: empty graph or edgeless graph → all nodes are independent
    if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
        return set(graph.nodes())  # Return all nodes

    # Work on a copy to avoid modifying the original graph
    working_graph = graph.copy()

    # Remove self-loops (they would invalidate independence checks and are usually not allowed in simple graphs)
    working_graph.remove_edges_from(list(nx.selfloop_edges(working_graph)))

    # Collect all isolated nodes (degree 0) — they can always be added to any independent set
    isolates = set(nx.isolates(working_graph))
    working_graph.remove_nodes_from(isolates)

    # If only isolates remain after removal, return them
    if working_graph.number_of_nodes() == 0:
        return isolates

    cover = find_vertex_cover(working_graph)
    nodes = set(working_graph)
    approximate_independent_set = nodes - cover
    for u in cover:
        candidate = (cover - {u}) | set(working_graph.neighbors(u))
        iset = nodes - set(candidate)
        solution = maximize_solution(working_graph, iset)
        if len(solution) >= len(approximate_independent_set):
            approximate_independent_set = solution

    # Always add the original isolated nodes
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
        """
        Verifies if a given set of vertices is a valid Independent Set for the graph.

        Args:
            graph (nx.Graph): The input graph.
            independent_set (set): A set of vertices to check.

        Returns:
            bool: True if the set is a valid Independent Set, False otherwise.
        """
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

    for k in range(1, n_vertices + 1): # Iterate through all possible sizes of the cover
        for candidate in itertools.combinations(graph.nodes(), k):
            cover_candidate = set(candidate)
            if is_independent_set(graph, cover_candidate) and len(cover_candidate) > n_max_vertices:
                n_max_vertices = len(cover_candidate)
                best_solution = cover_candidate
                
    return best_solution



def find_independent_set_approximation(graph):
    """
    Computes an approximate Independent Set in polynomial time with an approximation ratio of at most 2 for undirected graphs.

    Args:
        graph: A NetworkX Graph.

    Returns:
        A set of vertex indices representing the approximate Independent Set, or None if the graph is empty.
    """

    if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
        return None

    #networkx doesn't have a guaranteed independent set function, so we use approximation
    complement_graph = nx.complement(graph)
    independent_set = nx.approximation.max_clique(complement_graph)
    return independent_set