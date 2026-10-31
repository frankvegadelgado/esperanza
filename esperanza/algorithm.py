# Version: v0.1.6
# Modified on 31/10/2026
# Author: Frank Vega

import itertools
import networkx as nx
# Salvador (>= 0.1.4) supplies the vertex cover. Salvador guarantees a
# 2-approximation for Minimum Vertex Cover: |C| <= 2 * tau(G).
from salvador.algorithm import find_vertex_cover as salvador_vertex_cover
from salvador.algorithm import refine_cover
from salvador.local_search import improve_vertex_cover
from .localsearch import local_restart_search
from collections import deque

# Work budgets, in units of (n + m), of the three budgeted phases. With fixed
# budgets every phase runs in O(n + m) time.
RESTART_BUDGET = 100
LOCAL_RESTART_BUDGET = 100
ILS_BUDGET = 100
ILS_SEED = 20261031

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

def find_independent_set(graph: nx.Graph,
                         restart_budget: float = RESTART_BUDGET,
                         local_restart_budget: float = LOCAL_RESTART_BUDGET,
                         ils_budget: float = ILS_BUDGET,
                         ils_seed: int = ILS_SEED):
    """
    Compute an approximate maximum independent set in near-linear time.

    The vertex cover C used by this function is computed by Salvador
    (salvador >= 0.1.4), which guarantees a 2-approximation for Minimum
    Vertex Cover, |C| <= 2 * tau(G). The incumbent starts at V - C (or a
    larger Caro-Wei set) and is only ever replaced by a larger independent
    set, so the returned set S satisfies |S| >= n - |C|, and the
    complementary vertex cover V - S satisfies |V - S| <= |C| <= 2 * tau(G).

    Phases (G is the working graph without self-loops and isolated vertices):

    1. Caro-Wei baseline: dynamic minimum-degree greedy, O(n + m).
    2. Salvador cover C and its complement V - C, O((n + m) alpha(n)).
    3. Restarts: for every u in C and for a maximum-degree vertex, move u
       into V - C, evict its neighbours and grow to a maximal independent
       set (the multi-restart loop of version 0.1.5), evaluated locally by
       Salvador's refine_cover with work budget restart_budget * (n + m).
    4. Local restarts (esperanza.localsearch): chained (k, k+1)-improvements
       on the incumbent, budget local_restart_budget * (n + m).
    5. Iterated local search with (1,2)-swaps (Andrade, Resende and Werneck),
       Salvador's improve_vertex_cover, budget ils_budget * (n + m), fixed
       seed ils_seed.
    6. A final MaximizeSolution pass, O(n + m), so the output is a maximal
       independent set.

    Version 0.1.6 removes the complement-clique phase of version 0.1.5,
    which built the complement graph and cost O(n^3 log n + n^4 / w).

    Running time: O((n + m) alpha(n)) for fixed budgets, where alpha is the
    inverse Ackermann function (from Salvador); every other phase is
    O(n + m) or O(budget * (n + m)).

    Args:
        graph (nx.Graph): An undirected NetworkX graph.
        restart_budget, local_restart_budget, ils_budget: Work budgets in
            units of (n + m); None in restart_budget removes that cap.
        ils_seed: Seed of the iterated local search.

    Returns:
        set: A maximal independent set of graph (self-loop vertices excluded
        unless isolated after removing their loops).
    """
    if graph.number_of_nodes() == 0:
        return set()

    working_graph = graph.copy()
    working_graph.remove_edges_from(nx.selfloop_edges(working_graph))

    isolates = set(nx.isolates(working_graph))
    working_graph.remove_nodes_from(isolates)

    if working_graph.number_of_nodes() == 0:
        return isolates

    nodes = set(working_graph.nodes())

    # 1. Guarantee the Caro-Wei bound unconditionally
    caro_wei_iset = pure_caro_wei_baseline(working_graph)

    # 2. Get the Salvador cover (2-approximation for vertex cover)
    cover = salvador_vertex_cover(working_graph)
    salvador_iset = nodes - cover

    best_solution = caro_wei_iset if len(caro_wei_iset) > len(salvador_iset) else salvador_iset

    # 3. Budgeted restarts from every cover vertex and a maximum-degree vertex
    v_max = max(working_graph.nodes, key=working_graph.degree)
    restart_iset = nodes - refine_cover(working_graph, cover, v_max, restart_budget)
    if len(restart_iset) > len(best_solution):
        best_solution = restart_iset

    # 4. Budgeted local restarts on the incumbent
    adj_map = {u: list(working_graph.adj[u]) for u in working_graph}
    solution = local_restart_search(adj_map, best_solution, local_restart_budget)
    if len(solution) > len(best_solution):
        best_solution = solution

    # 5. Budgeted iterated local search with (1,2)-swaps
    solution = nodes - improve_vertex_cover(adj_map, nodes - best_solution, ils_budget, ils_seed)
    if len(solution) > len(best_solution):
        best_solution = solution

    # 6. Make sure the output is maximal
    best_solution = maximize_solution(working_graph, best_solution)

    best_solution.update(isolates)
    return best_solution

def find_vertex_cover(graph: nx.Graph):
    """
    Vertex cover returned by Esperanza: the complement V - S of the
    independent set S computed by find_independent_set.

    On a graph without self-loops, |S| >= n - |C| for the Salvador cover C
    computed inside find_independent_set, so this cover is never larger than
    Salvador's cover, and Salvador guarantees a 2-approximation, hence
    |V - S| <= 2 * tau(G). A vertex with a self-loop must belong to every
    cover, so such vertices are added to the returned cover.

    Args:
        graph (nx.Graph): An undirected NetworkX graph.

    Returns:
        set: A vertex cover of graph.
    """
    if graph.number_of_nodes() == 0:
        return set()
    loops = {u for u, _ in nx.selfloop_edges(graph)}
    independent = find_independent_set(graph) - loops
    return (set(graph.nodes()) - independent) | loops

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