# Version: v0.1.6
# Created on 31/10/2026
# Author: Frank Vega

"""
Budgeted local-restart search for the maximum independent set problem.

A *local restart* at a vertex u outside the current independent set S forces u
into S, evicts the k solution neighbours X = N(u) & S of u, and refills the
region freed by the eviction: every vertex w adjacent to some x in X that has
no neighbour left in S is inserted greedily in ascending-degree order. The
move is committed when the set grows, that is when it inserts at least k
vertices besides u (a (k, k+1)-improvement), and undone otherwise.

This is the restart of Esperanza 0.1.5 (insert u, drop its neighbours, grow to
a maximal independent set) evaluated locally: when S is maximal, only the
neighbours of the evicted vertices can become free, so the refill visits the
two-hop neighbourhood of u instead of the whole graph. Committed moves are
applied to the incumbent, and the vertices whose tightness drops are queued
again, so improvements are chained.

Linear-time guarantee
---------------------
For every vertex v, tight[v] = |N(v) & S| is maintained incrementally. Every
adjacency scan, insertion, removal, undo step and the sorting of each refill
list (k log2 k for k candidates) is charged to a work counter. A move is only
started when its cost bound fits in the remaining budget, and the search stops
as soon as the counter exceeds budget * (n + m). With the O(n + m) set-up,
the routine runs in O(budget * (n + m)) time for any fixed budget.

The routine never returns a smaller set than its input, and it returns a
maximal independent set when its input is maximal.
"""

from collections import deque
from math import log2

DEFAULT_BUDGET = 100


class _State:
    """Independent set on an index graph with tightness counters and an undo log."""

    __slots__ = ("adj", "in_s", "tight", "work", "log")

    def __init__(self, adj, members):
        n = len(adj)
        self.adj = adj
        self.in_s = [False] * n
        self.tight = [0] * n
        self.work = 0
        self.log = []
        for v in members:
            self.insert(v)
        self.log.clear()

    def insert(self, v):
        self.in_s[v] = True
        nb = self.adj[v]
        self.work += len(nb) + 1
        for y in nb:
            self.tight[y] += 1
        self.log.append((True, v))

    def remove(self, v):
        self.in_s[v] = False
        nb = self.adj[v]
        self.work += len(nb) + 1
        for y in nb:
            self.tight[y] -= 1
        self.log.append((False, v))

    def undo(self):
        log = self.log
        self.log = []
        while log:
            added, v = log.pop()
            if added:
                self.remove(v)
            else:
                self.insert(v)
        self.log.clear()


def local_restart_search(adj_map, independent, budget=DEFAULT_BUDGET):
    """
    Improve an independent set by budgeted local restarts.

    Args:
        adj_map: Mapping from every vertex to an iterable of its neighbours
            (no self-loops).
        independent: An independent set of that graph.
        budget: Work budget in units of (n + m).

    Returns:
        set: An independent set at least as large as ``independent``.
    """
    nodes = list(adj_map)
    n = len(nodes)
    if n == 0:
        return set()
    index = {v: i for i, v in enumerate(nodes)}
    adj = [[index[u] for u in adj_map[v] if u != v] for v in nodes]
    m = sum(len(a) for a in adj) // 2
    deg = [len(a) for a in adj]

    s = _State(adj, (index[v] for v in independent))
    in_s, tight = s.in_s, s.tight
    limit = s.work + int(budget * (n + m)) + 1

    # Initial work list: vertices outside S by ascending tightness, ties by
    # ascending degree (two stable counting sorts, O(n + Delta)).
    max_deg = max(deg, default=0)
    by_degree = [[] for _ in range(max_deg + 1)]
    for v in range(n):
        if not in_s[v]:
            by_degree[deg[v]].append(v)
    by_tight = [[] for _ in range(max_deg + 1)]
    for bucket in by_degree:
        for v in bucket:
            by_tight[tight[v]].append(v)
    s.work += n + max_deg + 1
    queue = deque(v for bucket in by_tight for v in bucket)
    queued = [False] * n
    for v in queue:
        queued[v] = True

    stamp = [0] * n
    clock = 0

    while queue and s.work <= limit:
        u = queue.popleft()
        queued[u] = False
        s.work += 1
        if in_s[u]:
            continue
        if tight[u] == 0:
            # Free vertex: insert it directly (keeps S maximal).
            s.insert(u)
            s.log.clear()
            continue

        evicted = [x for x in adj[u] if in_s[x]]
        s.work += deg[u]
        # Cost bound of the move: evictions, refill scan, refill sort,
        # insertions and a possible undo, all bounded by the degrees below.
        two_hop = sum(deg[x] for x in evicted)
        bound = 3 * (two_hop + deg[u] + len(evicted) + 1)
        bound += int(two_hop * log2(two_hop + 1))
        if s.work + bound > limit:
            continue

        s.log.clear()
        for x in evicted:
            s.remove(x)
        s.insert(u)

        clock += 1
        candidates = []
        for x in evicted:
            s.work += deg[x]
            for w in adj[x]:
                if stamp[w] != clock and not in_s[w] and tight[w] == 0:
                    stamp[w] = clock
                    candidates.append(w)
        k = len(candidates)
        if k > 1:
            candidates.sort(key=lambda w: (deg[w], w))
            s.work += int(k * log2(k))

        added = 0
        for w in candidates:
            s.work += 1
            if not in_s[w] and tight[w] == 0:
                s.insert(w)
                added += 1

        if 1 + added > len(evicted):
            # Committed (k, k+1)-improvement: requeue the vertices whose
            # tightness dropped (neighbours of the evicted vertices).
            s.log.clear()
            for x in evicted:
                s.work += deg[x]
                for w in adj[x]:
                    if not in_s[w] and not queued[w]:
                        queued[w] = True
                        queue.append(w)
        else:
            s.undo()

    return {nodes[i] for i in range(n) if in_s[i]}
