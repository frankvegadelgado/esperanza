# Created on 10/12/2025
# Author: Frank Vega

import itertools

import networkx as nx
from hvala.algorithm import find_vertex_cover 

def find_independent_set(graph):
    """
    Compute an approximate maximum independent set using bipartite graphs.

    The algorithm works as follows:
    - Isolated nodes are always included (they form an independent set by themselves).
    - For each connected component:
        - If the component is bipartite, compute the exact maximum independent set using König's theorem
          (maximum independent set = total vertices - minimum vertex cover = total vertices - maximum matching).
        - If the component is not bipartite, use an approximation technique:
            1. Find a vertex cover C0 in the component.
            2. Take I0 = V - C0 (the complement, which is an independent set).
            3. Remove I0 from the graph.
            4. In the remaining graph, find isolated nodes and add them separately.
            5. Find another vertex cover C1 in the remaining non-isolated part.
            6. Take I1 = remaining vertices - C1 (another independent set).
            7. The union I0 ∪ I1 ∪ isolated nodes induces a bipartite subgraph (I0 and I1 are independent sets,
               and there are no edges within each).
            8. Compute the exact maximum independent set on this induced bipartite subgraph.
    - The result is a valid independent set (guaranteed by construction and verified at the end).
    - This is an approximation algorithm because the induced bipartite subgraph may be smaller than the full graph.

    Args:
        graph (nx.Graph): An undirected NetworkX graph.

    Returns:
        set: A maximal independent set of vertices (approximate maximum).
    """
    def iset_bipartite(bipartite_graph):
        """Compute a maximum independent set for a bipartite graph using maximum matching.

        Uses Hopcroft-Karp to find maximum matching, then converts it to a minimum vertex cover
        (via NetworkX's bipartite utility), then takes the complement as the maximum independent set.

        Args:
            bipartite_graph (nx.Graph): A bipartite NetworkX graph.

        Returns:
            set: A maximum independent set for the bipartite graph.
        """
        independent_set = set()
        # Process each connected component separately (matching is per component)
        for component in nx.connected_components(bipartite_graph):
            subgraph = bipartite_graph.subgraph(component)
            # Hopcroft-Karp is an efficient algorithm for maximum bipartite matching
            matching = nx.bipartite.hopcroft_karp_matching(subgraph)
            # Convert matching to minimum vertex cover using König's theorem implementation
            vertex_cover = nx.bipartite.to_vertex_cover(subgraph, matching)
            # Complement of vertex cover is the maximum independent set
            independent_set.update(set(subgraph) - vertex_cover)
        return independent_set

    def is_independent_set(graph, independent_set):
        """
        Verify if a set of vertices is an independent set in the graph.
        (Used for safety/debugging; raises an error if the result is invalid.)

        Args:
            graph (nx.Graph): The input graph.
            independent_set (set): Vertices to check.

        Returns:
            bool: True if the set is independent, False otherwise.
        """
        for u, v in graph.edges():
            if u in independent_set and v in independent_set:
                return False
        return True

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

    # Main loop: process each remaining connected component
    approximate_independent_set = set()
    for component in nx.connected_components(working_graph):
        component_subgraph = working_graph.subgraph(component)

        if nx.bipartite.is_bipartite(component_subgraph):
            # Exact solution for bipartite graphs
            solution = iset_bipartite(component_subgraph)
        else:
            # Approximation for non-bipartite graphs
            G = component_subgraph.copy()

            # Step 1: Find a vertex cover → complement is an independent set
            bipartite_set0 = set(G) - find_vertex_cover(G)

            # Step 2: Remove that independent set
            G.remove_nodes_from(bipartite_set0)

            # Step 3: Collect newly created isolated nodes
            isolated_nodes = set(nx.isolates(G))
            G.remove_nodes_from(isolated_nodes)

            # Step 4: Find another vertex cover in the remaining graph → complement is another independent set
            bipartite_set1 = set(G) - find_vertex_cover(G)

            # Construct a bipartite subgraph induced by the two independent sets plus isolates
            # (no edges within each set by construction)
            bipartite_graph = component_subgraph.subgraph(bipartite_set0 | bipartite_set1 | isolated_nodes)

            # Compute exact maximum independent set on this induced bipartite subgraph
            solution = iset_bipartite(bipartite_graph)

        # Accumulate solutions from all components
        approximate_independent_set.update(solution)

    # Always add the original isolated nodes
    approximate_independent_set.update(isolates)

    # Safety check: ensure the returned set is truly independent (should never fail if implementation is correct)
    if not is_independent_set(graph, approximate_independent_set):
        raise RuntimeError(f"Polynomial-time reduction failed: the set {approximate_independent_set} is not independent")

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