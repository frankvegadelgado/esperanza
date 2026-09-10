# Version: v0.1.4
# Modified on 10/09/2026
# Author: Frank Vega

import itertools
import networkx as nx
import math
from hvala.algorithm import find_vertex_cover
from .disjoint import FastCliqueUF
from collections import deque

def maximize_solution(G: nx.Graph, S: set):
    """
    Repair a candidate set into an independent set and greedily maximize it,
    in linear time O(n + m).

    Phase 1 (repair): Nodes of S in conflict (adjacent to other members of S)
    are removed. Each conflicting edge inside S is resolved by discarding the
    endpoint with the higher current conflict count, so a single removal fixes
    as many conflicts as possible. Counts are maintained incrementally, giving
    O(n + m) total work.

    Phase 2 (grow): All nodes outside the repaired set (including those removed
    in Phase 1) are scanned in ascending-degree order (counting sort, O(n)) and
    added whenever they have no neighbor already in the set. This yields a
    maximal independent set containing the repaired kernel, at O(n + m) cost.

    Args:
        G (nx.Graph): An undirected NetworkX graph.
        S (set): Candidate node set (may contain conflicts).

    Returns:
        set: A maximal independent set of G.
    """
    independent = set(S)

    # --- Phase 1: remove conflict nodes ---
    # conflicts[u] = number of neighbors of u inside the current set.
    # Total adjacency scans are bounded by 2m -> O(n + m).
    conflicts = {u: sum(1 for v in G.adj[u] if v in independent) for u in independent}

    for u, v in G.edges():
        if u != v and u in independent and v in independent:
            # Discard the endpoint involved in more remaining conflicts.
            loser = u if conflicts[u] >= conflicts[v] else v
            independent.discard(loser)
            # Keep counts accurate; each node is removed at most once,
            # so these updates cost O(m) overall.
            for w in G.adj[loser]:
                if w in conflicts:
                    conflicts[w] -= 1
            del conflicts[loser]

    # --- Phase 2: add non-conflicting nodes to enlarge the set ---
    # Counting sort by degree (O(n)); low-degree nodes first tends to
    # block fewer future additions.
    buckets = {}
    max_degree = 0
    for u in G.nodes():
        if u not in independent:
            d = G.degree(u)
            buckets.setdefault(d, []).append(u)
            if d > max_degree:
                max_degree = d

    for d in range(max_degree + 1):
        for u in buckets.get(d, ()):
            if all(w not in independent for w in G.adj[u]):
                independent.add(u)

    return independent

def pure_caro_wei_baseline(G: nx.Graph):
    """
    Computes a dynamic min-degree-greedy independent set, guaranteeing the
    Caro-Wei bound |I| >= sum(1/(d(v)+1)) >= n/(Delta+1), in O(n + m) time.

    This repeatedly removes the CLOSED neighborhood of the current minimum-
    degree vertex, which is a different (and slightly trickier) update
    pattern than a simple degree-decrement greedy: removing one vertex can
    remove many others in one step, and their neighbors' degrees must drop
    accordingly, potentially requiring the bucket scan to "retreat" to a
    lower bucket it had already passed. The retreat is what makes this
    still O(n + m) overall (each degree decrease is O(1) amortized), rather
    than needing a full rescan.
    """
    adj = {v: set(G[v]) for v in G.nodes()}
    n = len(adj)
    if n == 0:
        return set()

    deg = {v: len(adj[v]) for v in adj}
    maxd = max(deg.values(), default=0)
    buckets = [deque() for _ in range(maxd + 1)]
    for v, d in deg.items():
        buckets[d].append(v)

    removed = set()
    independent_set = set()
    remaining = n
    ptr = 0

    while remaining > 0:
        while ptr <= maxd:
            while buckets[ptr] and buckets[ptr][0] in removed:
                buckets[ptr].popleft()
            if buckets[ptr]:
                break
            ptr += 1
        if ptr > maxd:
            break  # safety net; should not trigger while remaining > 0

        v_min = buckets[ptr].popleft()
        independent_set.add(v_min)
        removed.add(v_min)
        remaining -= 1

        to_remove = [u for u in adj[v_min] if u not in removed]
        for u in to_remove:
            removed.add(u)
            remaining -= 1
            for w in adj[u]:
                if w not in removed and w != v_min:
                    deg[w] -= 1
                    buckets[deg[w]].append(w)
                    if deg[w] < ptr:
                        ptr = deg[w]

    return independent_set

def find_independent_set(graph: nx.Graph):
    """
    Compute an approximate maximum independent set with strict Caro-Wei 
    and Delta bounds.
    """
    if graph.number_of_nodes() == 0:
        return set()

    working_graph = graph.copy()
    working_graph.remove_edges_from(nx.selfloop_edges(working_graph))

    isolates = set(nx.isolates(working_graph))
    working_graph.remove_nodes_from(isolates)

    if working_graph.number_of_nodes() == 0:
        return isolates

    # 1. Guarantee the Caro-Wei bound unconditionally
    caro_wei_iset = pure_caro_wei_baseline(working_graph)

    # 2. Get the Hvala cover
    cover = find_vertex_cover(working_graph)
    nodes = set(working_graph.nodes())
    hvala_iset = nodes - cover

    # 3. Get best solution
    best_solution = caro_wei_iset if len(caro_wei_iset) > len(hvala_iset) else hvala_iset

    # 4. Explicitly inject the maximum degree vertex to guarantee branch evaluation
    v_max = max(working_graph.nodes, key=working_graph.degree)
    evaluation_pool = cover.union({v_max})
    
    # 5. Iterate over the guaranteed pool
    for u in evaluation_pool:
        candidate = (cover - {u}) | set(working_graph.neighbors(u))
        iset = nodes - candidate
        
        solution = maximize_solution(working_graph, iset) # Keep Phase 1/2 repair here
        
        if len(solution) > len(best_solution):
            best_solution = solution

    # 6. Use disjoint set on complement graph 
    complement_graph = nx.complement(working_graph)
    disjoint_set = FastCliqueUF(complement_graph)
    visited = set()
    for u in complement_graph:
        neighbors = list(complement_graph.neighbors(u))
        found = None
        for v in neighbors:
            pair = frozenset({u, v})
            if pair in visited:
                continue
            visited.add(pair)
            if disjoint_set.add(v):
                found = u
        if found is not None:
            # Extract all components of size >= 2 (potential cliques)
            cliques = [s for s in disjoint_set.to_sets() if len(s) >= 2]
            sorted_cliques = sorted(cliques, key=len)
            sqrt = math.floor(math.sqrt(len(sorted_cliques)))
            # Iterate from the maximal clique-like components;                        
            while len(sorted_cliques) >= sqrt:
                clique = sorted_cliques.pop()
                iset = clique | {found}
                solution = maximize_solution(working_graph, iset) # Keep Phase 1/2 repair here

                if len(solution) > len(best_solution):
                    best_solution = solution
        while neighbors:
            w = neighbors.pop()
            disjoint_set.remove(w)

    best_solution.update(isolates)
    return best_solution

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