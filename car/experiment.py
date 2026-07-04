#!/usr/bin/env python3
"""
car: Certified Approximation Ratio experiments for the Esperanza algorithm.

Generates 30,000 instances across structured, random, and amplified families,
computes ALG = |find_independent_set(G)| and the exact optimum OPT via a
mixed-integer linear program (MILP) solved with SciPy's HiGHS backend, and
reports the approximation ratio  rho = OPT / ALG  per instance and per family.

MILP formulation (exact Maximum Independent Set):

    maximize   sum_v x_v
    subject to x_u + x_v <= 1   for every edge (u, v)
               x_v in {0, 1}

Usage (from the repository root):

    python car/experiment.py                     # full 30,000-instance run
    python car/experiment.py --total 3000        # scaled-down smoke run
    python car/experiment.py --seed 7            # different RNG seed
    python car/experiment.py --self-check        # cross-check MILP vs brute force first

Outputs (written to --out, default car/results):

    results.csv          one row per instance: family, n, m, alg, opt, ratio
    summary.txt          worst / mean ratio per family and overall
    summary_table.tex    LaTeX rows (booktabs) for the paper
    worst_instance.txt   edge list of the worst-ratio instance found
"""

import argparse
import csv
import itertools
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix

# Import the algorithm from the repository (one level above car/).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from esperanza.algorithm import find_independent_set  # noqa: E402


# ----------------------------------------------------------------------------
# Exact optimum via MILP
# ----------------------------------------------------------------------------

def milp_independence_number(G: nx.Graph) -> int:
    """Exact independence number alpha(G), certified by MILP (HiGHS)."""
    n = G.number_of_nodes()
    if n == 0:
        return 0
    m = G.number_of_edges()
    if m == 0:
        return n

    nodes = list(G.nodes())
    idx = {v: i for i, v in enumerate(nodes)}

    rows, cols, data = [], [], []
    for k, (u, v) in enumerate(G.edges()):
        rows.extend((k, k))
        cols.extend((idx[u], idx[v]))
        data.extend((1.0, 1.0))
    A = csr_matrix((data, (rows, cols)), shape=(m, n))

    res = milp(
        c=-np.ones(n),                                # maximize sum x_v
        constraints=LinearConstraint(A, -np.inf, 1.0),
        integrality=np.ones(n),
        bounds=Bounds(0.0, 1.0),
    )
    if not res.success:
        raise RuntimeError(f"MILP failed on n={n}, m={m}: {res.message}")
    return round(-res.fun)


def brute_force_alpha(G: nx.Graph) -> int:
    """Exponential-time reference, used only by --self-check on tiny graphs."""
    nodes = list(G.nodes())
    best = 0
    for k in range(len(nodes), 0, -1):
        if k <= best:
            break
        for cand in itertools.combinations(nodes, k):
            s = set(cand)
            if not any(v in s for u in s for v in G.adj[u]):
                best = max(best, k)
                break
    return best


def is_independent(G: nx.Graph, S: set) -> bool:
    return not any(v in S for u in S for v in G.adj[u])


# ----------------------------------------------------------------------------
# Instance families
# ----------------------------------------------------------------------------

def blow_up(G: nx.Graph, t: int) -> nx.Graph:
    """Independent blow-up: each vertex becomes an independent cluster of t
    copies; each edge becomes a complete join between the two clusters."""
    H = nx.Graph()
    H.add_nodes_from((v, i) for v in G.nodes() for i in range(t))
    H.add_edges_from(
        ((u, i), (v, j))
        for u, v in G.edges()
        for i in range(t)
        for j in range(t)
    )
    return H


def disjoint_copies(G: nx.Graph, k: int) -> nx.Graph:
    """Disjoint union of k copies of G."""
    H = nx.Graph()
    for c in range(k):
        H.add_nodes_from((v, c) for v in G.nodes())
        H.add_edges_from(((u, c), (v, c)) for u, v in G.edges())
    return H


def gen_gnp(rng: random.Random) -> nx.Graph:
    n = rng.randint(6, 16)
    p = rng.uniform(0.15, 0.85)
    return nx.gnp_random_graph(n, p, seed=rng.randrange(2**31))


def gen_bipartite(rng: random.Random) -> nx.Graph:
    a = rng.randint(3, 14)
    b = rng.randint(3, 14)
    p = rng.uniform(0.10, 0.50)
    return nx.bipartite.random_graph(a, b, p, seed=rng.randrange(2**31))


def gen_regular(rng: random.Random) -> nx.Graph:
    d = rng.randint(3, 5)
    n = rng.choice([k for k in range(max(d + 1, 8), 17) if (k * d) % 2 == 0])
    return nx.random_regular_graph(d, n, seed=rng.randrange(2**31))


def gen_star(rng: random.Random) -> nx.Graph:
    return nx.star_graph(rng.randint(3, 80))


def gen_complete(rng: random.Random) -> nx.Graph:
    return nx.complete_graph(rng.randint(3, 30))


def gen_complete_bipartite(rng: random.Random) -> nx.Graph:
    a = rng.randint(2, 12)
    b = rng.randint(a, 20)
    return nx.complete_bipartite_graph(a, b)


# (family name, generator, share of the total instance budget)
SCHEDULE = [
    ("gnp",                gen_gnp,                0.50),
    ("bipartite_gnp",      gen_bipartite,          0.20),
    ("regular",            gen_regular,            0.10),
    ("star",               gen_star,               2000 / 30000),
    ("complete",           gen_complete,           1000 / 30000),
    ("complete_bipartite", gen_complete_bipartite, 1000 / 30000),
    # remaining budget: amplifications of the worst random instances found
]
AMPLIFIED_FAMILY = "amplified"
AMPLIFIED_SHARE = 2000 / 30000
TOP_BASES = 25          # worst base instances kept for amplification
MAX_BASE_N = 12         # keep amplified MILPs tractable


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------

def run_instance(G: nx.Graph):
    """Return (n, m, alg, opt, ratio) for one instance; certify everything."""
    S = find_independent_set(G)
    if not is_independent(G, S):
        raise AssertionError("Algorithm returned a non-independent set!")
    alg = len(S)
    opt = milp_independence_number(G)
    if alg == 0:
        raise AssertionError("Algorithm returned an empty set on a non-empty graph")
    return G.number_of_nodes(), G.number_of_edges(), alg, opt, opt / alg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--total", type=int, default=30000,
                    help="total number of instances (default 30000)")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed")
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "results",
                    help="output directory")
    ap.add_argument("--self-check", action="store_true",
                    help="cross-check the MILP against brute force on 50 tiny graphs first")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    if args.self_check:
        print("Self-check: MILP vs brute force on 50 tiny random graphs ...")
        for trial in range(50):
            G = nx.gnp_random_graph(rng.randint(1, 8), rng.random(),
                                    seed=rng.randrange(2**31))
            assert milp_independence_number(G) == brute_force_alpha(G), trial
        print("Self-check passed.\n")

    counts = {name: int(round(share * args.total)) for name, _, share in SCHEDULE}
    counts[AMPLIFIED_FAMILY] = int(round(AMPLIFIED_SHARE * args.total))
    # largest-remainder fix so counts sum exactly to --total
    counts["gnp"] += args.total - sum(counts.values())

    rows = []                     # (family, n, m, alg, opt, ratio)
    worst = (0.0, None, None)     # (ratio, family, edge list)
    bases = []                    # (ratio, edge list) candidates for amplification
    done = 0
    t0 = time.time()

    def record(family: str, G: nx.Graph) -> None:
        nonlocal worst, done
        n, m, alg, opt, ratio = run_instance(G)
        rows.append((family, n, m, alg, opt, f"{ratio:.6f}"))
        if ratio > worst[0]:
            worst = (ratio, family, sorted(G.edges()))
        if (family in ("gnp", "regular") and n <= MAX_BASE_N
                and nx.number_connected_components(G) == 1):
            bases.append((ratio, sorted(G.edges())))
        done += 1
        if done % 1000 == 0:
            rate = done / (time.time() - t0)
            print(f"  {done}/{args.total} instances "
                  f"({rate:.0f}/s, worst ratio so far {worst[0]:.4f})")

    print(f"Running {args.total} MILP-certified instances (seed={args.seed}) ...")
    for name, gen, _ in SCHEDULE:
        print(f"family: {name} ({counts[name]} instances)")
        for _ in range(counts[name]):
            record(name, gen(rng))

    # Amplification phase: disjoint copies and independent blow-ups of the
    # worst connected small bases discovered above.
    print(f"family: {AMPLIFIED_FAMILY} ({counts[AMPLIFIED_FAMILY]} instances)")
    bases.sort(key=lambda br: -br[0])
    top = [nx.Graph(edges) for _, edges in bases[:TOP_BASES]] or \
          [nx.gnp_random_graph(9, 0.5, seed=args.seed)]
    for _ in range(counts[AMPLIFIED_FAMILY]):
        base = rng.choice(top)
        if rng.random() < 0.5:
            record(AMPLIFIED_FAMILY, disjoint_copies(base, rng.randint(2, 8)))
        else:
            record(AMPLIFIED_FAMILY, blow_up(base, rng.randint(2, 6)))

    # ------------------------------------------------------------------ output
    csv_path = args.out / "results.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["family", "n", "m", "alg", "opt", "ratio"])
        w.writerows(rows)

    per_family = defaultdict(list)
    for family, *_rest, ratio in rows:
        per_family[family].append(float(ratio))

    summary_lines = [
        f"Esperanza MILP-certified stress run: {len(rows)} instances, "
        f"seed={args.seed}, {time.time() - t0:.1f}s",
        "",
        f"{'family':22s} {'count':>7s} {'worst':>8s} {'mean':>8s}",
    ]
    tex_lines = []
    for family in sorted(per_family, key=lambda k: -max(per_family[k])):
        r = per_family[family]
        summary_lines.append(
            f"{family:22s} {len(r):7d} {max(r):8.4f} {sum(r)/len(r):8.4f}")
        tex_lines.append(
            f"{family.replace('_', ' ')} & {len(r)} & "
            f"{max(r):.3f} & {sum(r)/len(r):.3f} \\\\")
    all_r = [float(r[-1]) for r in rows]
    summary_lines += ["",
                      f"GLOBAL worst ratio: {max(all_r):.6f}  "
                      f"(family: {worst[1]})",
                      f"GLOBAL mean  ratio: {sum(all_r)/len(all_r):.6f}"]

    (args.out / "summary.txt").write_text("\n".join(summary_lines) + "\n")
    (args.out / "summary_table.tex").write_text("\n".join(tex_lines) + "\n")
    (args.out / "worst_instance.txt").write_text(
        f"ratio = {worst[0]:.6f}\nfamily = {worst[1]}\nedges = {worst[2]}\n")

    print()
    print("\n".join(summary_lines))
    print(f"\nWrote {csv_path}, summary.txt, summary_table.tex, worst_instance.txt")


if __name__ == "__main__":
    main()
