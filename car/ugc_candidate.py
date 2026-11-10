#!/usr/bin/env python3
"""
car/ugc_candidate.py: Esperanza as a Vertex Cover candidate on YES-type
instances of the Khot-Regev reduction.

Esperanza's vertex cover is C_E = V - S, where S = find_independent_set(G).
Esperanza starts from the complement of the Salvador cover C_S and only
replaces its incumbent by a strictly larger independent set, so

    |C_E| <= |C_S| <= 2 * tau(G)        (Salvador is a 2-approximation).

This script tests, instance by instance, the inequality

    (E)   |C_E| <= 2 * tau(G) - |M|,

where M = nx.approximation.min_maximal_matching(G) is the NetworkX minimum
maximal matching of G. Since |M| <= tau, (E) gives |C_E| <= tau + (tau - |M|).
Every inequality is decided with certified bounds L <= tau <= U only:

    certified   |C_E| <= 2 * L - |M|   (then (E) holds on the instance)
    violated    |C_E| >  2 * U - |M|   (then (E) fails: a counterexample)
    open        otherwise (the bounds are too wide to decide)

The same test is applied to the Salvador cover C_S, and the per-instance
3/2 certificate |C| <= 3/2 * L is also reported.

Independent-set form. Khot and Regev state their UGC-hardness result as a
gap for independent sets: it is UG-hard to distinguish graphs with an
independent set of (1/2 - eps) n vertices (YES) from graphs with no
independent set of more than delta n vertices (NO). The script therefore
also reports |S| / n for Esperanza's independent set S and the certified
range n - U <= alpha <= n - L.

Instances
---------
YES-type graphs of the Khot-Regev reduction from a Unique Games instance with
a planted labelling, generated exactly as in Part D of the Salvador
experiments (car/car_yes_instances.py of the salvador repository, the same
seed and the same per-instance generator): a random d-regular constraint
graph on N vertices, labels [k], permutation or linear (Max-2-Lin(Z_k))
constraints, noise eta, and s subsets per block drawn from the p-biased
measure; (v, x) and (w, y) are joined when x and pi_vw(y) are disjoint. The
dictator set {(v, x) : L(v) in x}, made independent greedily on violated
constraints, gives the planted cover, an upper bound on tau.

Bounds
------
    L = max(ceil(LP relaxation), integer-program dual bound)
    U = min(planted cover, integer-program primal value)
The LP relaxation is half the maximum matching of the bipartite double cover;
the integer program is scipy.optimize.milp (HiGHS) with a time limit. When
the integer program proves optimality, L = U = tau.

Run from the repository root (the package does not need to be installed;
the repository's own esperanza/ is used; salvador >= 0.1.5 must be importable):

    python car/ugc_candidate.py                   # N = 40, 80, 160, 320 (640 to 10,240 vertices)
    python car/ugc_candidate.py --quick           # N = 40, 80 (the sizes of Salvador's Part D)
    python car/ugc_candidate.py --blocks 640 1280 # larger instances
    python car/ugc_candidate.py --timeout 0       # no per-instance time limit

Each instance is built and evaluated in a worker process with a time limit
(--timeout, default 900 s); an instance that exceeds it is listed as
skipped. Esperanza 0.1.7 runs in near-linear time; the integer program
(--ip-time, --ip-max-n) is usually the slowest step.

Outputs (in car/results/):

    ugc_candidate.csv          one row per instance
    ugc_candidate_summary.txt  per-family counts of certified / open / violated
    ugc_candidate.json         all rows, parameters and environment
    ugc_counterexamples/       edge list of every instance on which (E) is
                               violated or open, for further study
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing
import platform
import random
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix

# Import the algorithm from the repository (one level above car/), so the
# script works whether or not the esperanza package is installed.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from esperanza.algorithm import find_independent_set  # noqa: E402

try:
    from salvador.algorithm import find_vertex_cover as salvador_vertex_cover  # noqa: E402
except ImportError as exc:  # pragma: no cover
    raise SystemExit("salvador >= 0.1.5 is required: pip install \"salvador>=0.1.5\"") from exc

SEED = 20261009  # same seed as Part D of the Salvador experiments
OUT_DIR = Path(__file__).resolve().parent / "results"


# ----------------------------------------------------------------------------
# Khot-Regev YES-type instances (planted Unique Games labelling)
# ----------------------------------------------------------------------------

def unique_games(N: int, d: int, k: int, eta: float, linear: bool, rng: random.Random):
    """Return (constraint graph H, constraints [(v, w, pi)], labelling L, violated edges)."""
    H = nx.random_regular_graph(d, N - (N * d) % 2, seed=rng.randrange(2**31))
    L = {v: rng.randrange(k) for v in H}
    cons = []
    violated = set()
    for v, w in H.edges():
        if rng.random() < eta:
            if linear:
                c = rng.randrange(k)
                pi = tuple((i + c) % k for i in range(k))
            else:
                pi = list(range(k))
                rng.shuffle(pi)
                pi = tuple(pi)
        elif linear:
            c = (L[v] - L[w]) % k
            pi = tuple((i + c) % k for i in range(k))
        else:
            pi = list(range(k))
            rng.shuffle(pi)
            j = pi.index(L[v])
            pi[j], pi[L[w]] = pi[L[w]], pi[j]  # now pi[L[w]] = L[v]
            pi = tuple(pi)
        if pi[L[w]] != L[v]:
            violated.add((v, w))
        cons.append((v, w, pi))
    return H, cons, L, violated


def sample_subset(k: int, p: float, rng: random.Random) -> int:
    x = 0
    for i in range(k):
        if rng.random() < p:
            x |= 1 << i
    return x


def apply_perm(y: int, pi: tuple[int, ...]) -> int:
    out = 0
    i = 0
    while y:
        if y & 1:
            out |= 1 << pi[i]
        y >>= 1
        i += 1
    return out


def khot_regev_yes(N: int, d: int, k: int, p: float, eta: float, s: int, linear: bool,
                   rng: random.Random):
    """Sampled Khot-Regev graph of a planted UG instance: (G, planted independent set)."""
    H, cons, L, _violated = unique_games(N, d, k, eta, linear, rng)
    blocks = {v: [sample_subset(k, p, rng) for _ in range(s)] for v in H}
    vid = {}
    for v in H:
        for t in range(s):
            vid[(v, t)] = len(vid)
    G = nx.Graph()
    G.add_nodes_from(range(len(vid)))
    for v, w, pi in cons:
        bx = blocks[v]
        by = [apply_perm(y, pi) for y in blocks[w]]
        for a, x in enumerate(bx):
            for b, y in enumerate(by):
                if x & y == 0:
                    G.add_edge(vid[(v, a)], vid[(w, b)])
    I = {vid[(v, t)] for v in H for t in range(s) if blocks[v][t] >> L[v] & 1}
    for u in sorted(I):
        if u in I and any(w in I for w in G[u]):
            I.discard(u)
    return G, I


def shuffle_labels(G: nx.Graph, I: set, rng: random.Random):
    perm = list(range(G.number_of_nodes()))
    rng.shuffle(perm)
    mapping = {v: perm[i] for i, v in enumerate(G.nodes())}
    return nx.relabel_nodes(G, mapping), {mapping[v] for v in I}


def instances(blocks, reps: int):
    """(name, group, params) for every instance; same names as Salvador's Part D."""
    for N in blocks:
        for r in range(reps):
            base = dict(N=N, d=4, k=4, p=0.45, eta=0.0, s=16, linear=False)
            yield f"kr_base_N{N}_r{r}", "KR permutation UG (k=4, p=0.45, eta=0)", base
            for k in (3, 6):
                yield f"kr_k{k}_N{N}_r{r}", f"KR label size k={k}", {**base, "k": k}
            for p in (0.35, 0.49):
                yield f"kr_p{p}_N{N}_r{r}", f"KR bias p={p}", {**base, "p": p}
            for eta in (0.02, 0.1):
                yield f"kr_eta{eta}_N{N}_r{r}", f"KR noise eta={eta}", {**base, "eta": eta}
            yield f"kr_d8_N{N}_r{r}", "KR constraint degree d=8", {**base, "d": 8}
            yield f"kr_s32_N{N}_r{r}", "KR samples per block s=32", {**base, "s": 32}
            yield f"kr_linear_N{N}_r{r}", "KR linear UG Max-2-Lin(Z_k)", {**base, "linear": True}


def instance_rng(seed, name: str) -> random.Random:
    return random.Random(f"{seed}:{name}")


# ----------------------------------------------------------------------------
# Certified bounds on tau
# ----------------------------------------------------------------------------

def lp_lower_bound(G: nx.Graph) -> int:
    """ceil of the LP relaxation of vertex cover (half a maximum matching of G x K2)."""
    H = nx.Graph()
    top = [(v, 0) for v in G]
    H.add_nodes_from(top)
    H.add_nodes_from((v, 1) for v in G)
    for u, v in G.edges():
        H.add_edge((u, 0), (v, 1))
        H.add_edge((v, 0), (u, 1))
    if H.number_of_edges() == 0:
        return 0
    M = nx.bipartite.hopcroft_karp_matching(H, top_nodes=top)
    return math.ceil((len(M) // 2) / 2 - 1e-9)


def integer_program(G: nx.Graph, time_limit: float):
    """Vertex cover integer program: (primal value or None, proved optimal, dual bound or None)."""
    nodes = list(G.nodes())
    idx = {v: i for i, v in enumerate(nodes)}
    E = list(G.edges())
    if not E:
        return 0, True, 0.0
    rows, cols, data = [], [], []
    for r, (u, v) in enumerate(E):
        rows.extend((r, r))
        cols.extend((idx[u], idx[v]))
        data.extend((1.0, 1.0))
    A = csr_matrix((data, (rows, cols)), shape=(len(E), len(nodes)))
    res = milp(c=np.ones(len(nodes)), constraints=LinearConstraint(A, lb=1, ub=np.inf),
               integrality=np.ones(len(nodes)), bounds=Bounds(0, 1),
               options={"time_limit": time_limit})
    dual = getattr(res, "mip_dual_bound", None)
    if res.x is None:
        return None, False, dual
    return int(round(res.fun)), res.status == 0, dual


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------

def is_cover(G: nx.Graph, C: set) -> bool:
    return all(u in C or v in C for u, v in G.edges())


def verdict(size: int, matching: int, lower: int, upper: int) -> str:
    """Decide |C| <= 2 tau - |M| from L <= tau <= U."""
    if size <= 2 * lower - matching:
        return "certified"
    if size > 2 * upper - matching:
        return "violated"
    return "open"


def run_instance(name: str, group: str, params: dict, ip_time: float, ip_max_n: int) -> dict:
    rng = instance_rng(SEED, name)
    G, I = khot_regev_yes(rng=rng, **params)
    G, I = shuffle_labels(G, I, rng)
    assert not any(u in I and v in I for u, v in G.edges()), name
    n, m = G.number_of_nodes(), G.number_of_edges()
    planted = sum(1 for v in G if v not in I and G.degree(v) > 0)

    # Esperanza
    t0 = time.perf_counter()
    S = find_independent_set(G)
    t_esp = time.perf_counter() - t0
    assert not any(u in S and v in S for u, v in G.edges()), f"{name}: Esperanza set is not independent"
    C_E = set(G.nodes()) - S
    assert is_cover(G, C_E), f"{name}: Esperanza cover is not a vertex cover"

    # Salvador, on the same graph without isolated vertices (as inside Esperanza)
    W = G.copy()
    W.remove_nodes_from(list(nx.isolates(W)))
    t0 = time.perf_counter()
    C_S = salvador_vertex_cover(W) if W.number_of_nodes() else set()
    t_sal = time.perf_counter() - t0
    assert is_cover(G, C_S), f"{name}: Salvador cover is not a vertex cover"

    # Minimum maximal matching of NetworkX
    M = nx.approximation.min_maximal_matching(G)

    # Certified bounds L <= tau <= U
    lower = lp_lower_bound(G)
    upper = planted
    tau = None
    ip_status = "skipped"
    t0 = time.perf_counter()
    if ip_max_n and n <= ip_max_n:
        value, optimal, dual = integer_program(G, ip_time)
        if dual is not None:
            lower = max(lower, math.ceil(dual - 1e-6))
        if value is not None:
            upper = min(upper, value)
        if optimal and value is not None:
            tau = value
            lower = upper = value
            ip_status = "optimal"
        else:
            ip_status = "time limit"
    t_ip = time.perf_counter() - t0
    upper = min(upper, len(C_E), len(C_S))  # any vertex cover is an upper bound on tau
    lower = min(lower, upper)

    e, s_, mm = len(C_E), len(C_S), len(M)
    row = {
        "group": group, "name": name, **params, "n": n, "m": m,
        "planted_cover": planted, "matching": mm,
        "lower": lower, "upper": upper, "tau": tau, "ip_status": ip_status,
        "esperanza": e, "salvador": s_,
        # Independent-set form of the Khot-Regev gap: alpha = n - tau
        "independent_set": n - e, "independent_fraction": (n - e) / n if n else 0.0,
        "alpha_lower": n - upper, "alpha_upper": n - lower,
        "esperanza_le_salvador": e <= s_, "esperanza_lt_salvador": e < s_,
        "bound_2L_minus_M": 2 * lower - mm, "bound_2U_minus_M": 2 * upper - mm,
        "esperanza_verdict": verdict(e, mm, lower, upper),
        "salvador_verdict": verdict(s_, mm, lower, upper),
        "esperanza_cert_3_2": 2 * e <= 3 * lower,
        "esperanza_ratio_upper": e / lower if lower else 1.0,
        "esperanza_ratio_lower": e / upper if upper else 1.0,
        "esperanza_optimal": tau is not None and e == tau,
        "seconds_esperanza": round(t_esp, 3), "seconds_salvador": round(t_sal, 3),
        "seconds_ip": round(t_ip, 3),
    }
    if row["esperanza_verdict"] != "certified":
        d = OUT_DIR / "ugc_counterexamples"
        d.mkdir(parents=True, exist_ok=True)
        with (d / f"{name}.txt").open("w", encoding="utf-8") as fh:
            fh.write(f"# {name} {row['esperanza_verdict']} n={n} m={m} |C_E|={e} |M|={mm} L={lower} U={upper}\n")
            for u, v in G.edges():
                fh.write(f"{u} {v}\n")
    return row


def _job(args):
    return run_instance(*args)


def run_with_timeout(job_args, timeout: float, state: dict):
    if timeout <= 0:
        return _job(job_args)
    if state.get("pool") is None:
        state["pool"] = multiprocessing.get_context("spawn").Pool(processes=1)
    res = state["pool"].apply_async(_job, (job_args,))
    try:
        return res.get(timeout)
    except multiprocessing.TimeoutError:
        state["pool"].terminate()
        state["pool"].join()
        state["pool"] = None
        return None


FIELDS = ["group", "name", "N", "d", "k", "p", "eta", "s", "linear", "n", "m",
          "planted_cover", "matching", "lower", "upper", "tau", "ip_status",
          "esperanza", "salvador", "independent_set", "independent_fraction", "alpha_lower", "alpha_upper",
          "esperanza_le_salvador", "esperanza_lt_salvador",
          "bound_2L_minus_M", "bound_2U_minus_M", "esperanza_verdict", "salvador_verdict",
          "esperanza_cert_3_2", "esperanza_ratio_upper", "esperanza_ratio_lower",
          "esperanza_optimal", "seconds_esperanza", "seconds_salvador", "seconds_ip"]


def summarise(rows: list[dict]) -> str:
    lines = []
    hdr = (f"{'family':42} {'inst':>4} {'E cert':>6} {'E open':>6} {'E viol':>6} "
           f"{'S cert':>6} {'S open':>6} {'S viol':>6} {'E<S':>4} {'E opt':>5} {'3/2':>4} {'max |C_E|/L':>11} {'min |S|/n':>9}")
    lines.append(hdr)
    lines.append("-" * len(hdr))
    groups = sorted({r["group"] for r in rows})
    for g in groups + ["ALL"]:
        rs = rows if g == "ALL" else [r for r in rows if r["group"] == g]
        if not rs:
            continue

        def cnt(key, val):
            return sum(1 for r in rs if r[key] == val)

        lines.append(
            f"{g[:42]:42} {len(rs):>4} {cnt('esperanza_verdict', 'certified'):>6} "
            f"{cnt('esperanza_verdict', 'open'):>6} {cnt('esperanza_verdict', 'violated'):>6} "
            f"{cnt('salvador_verdict', 'certified'):>6} {cnt('salvador_verdict', 'open'):>6} "
            f"{cnt('salvador_verdict', 'violated'):>6} {sum(r['esperanza_lt_salvador'] for r in rs):>4} "
            f"{sum(r['esperanza_optimal'] for r in rs):>5} {sum(r['esperanza_cert_3_2'] for r in rs):>4} "
            f"{max(r['esperanza_ratio_upper'] for r in rs):>11.4f} "
            f"{min(r['independent_fraction'] for r in rs):>9.4f}")
    lines.append("")
    lines.append("E = Esperanza cover V - S, S = Salvador cover; inequality tested: |C| <= 2 tau - |M|,")
    lines.append("M = nx.approximation.min_maximal_matching(G); cert: |C| <= 2L - |M|; viol: |C| > 2U - |M|;")
    lines.append("E<S: Esperanza strictly smaller than Salvador; 3/2: |C_E| <= 3/2 L; max |C_E|/L: certified ratio bound;")
    lines.append("min |S|/n: smallest fraction of vertices in Esperanza's independent set S = V - C_E.")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="UG sizes N = 40, 80")
    ap.add_argument("--blocks", type=int, nargs="+", default=None,
                    help="UG sizes N (default 40 80 160 320; overrides --quick)")
    ap.add_argument("--reps", type=int, default=2, help="instances per family and size (default 2)")
    ap.add_argument("--ip-time", type=float, default=90.0, help="integer program time limit in seconds (default 90)")
    ap.add_argument("--ip-max-n", type=int, default=3000,
                    help="run the integer program up to this many vertices (0 = never; default 3000)")
    ap.add_argument("--timeout", type=float, default=900.0,
                    help="per-instance time limit in seconds (default 900; 0 = no limit)")
    args = ap.parse_args()

    blocks = args.blocks or ((40, 80) if args.quick else (40, 80, 160, 320))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()
    rows, skipped = [], []
    state: dict = {}
    csv_path = OUT_DIR / "ugc_candidate.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        try:
            for name, group, params in instances(blocks, args.reps):
                row = run_with_timeout((name, group, params, args.ip_time, args.ip_max_n), args.timeout, state)
                if row is None:
                    print(f"  {name:24} skipped (more than {args.timeout:g} s)", flush=True)
                    skipped.append(name)
                    continue
                rows.append(row)
                writer.writerow(row)
                fh.flush()
                print(f"  {name:24} n={row['n']:>6,} |M|={row['matching']:>5} L={row['lower']:>5} U={row['upper']:>5}"
                      f" E={row['esperanza']:>5} S={row['salvador']:>5}"
                      f" E:{row['esperanza_verdict']:9} S:{row['salvador_verdict']:9}"
                      f" {row['seconds_esperanza']:8.1f}s", flush=True)
        finally:
            if state.get("pool") is not None:
                state["pool"].terminate()
                state["pool"].join()

    summary = summarise(rows) if rows else "no instance finished"
    header = (f"Esperanza UGC-candidate run on Khot-Regev YES instances: seed={SEED}, blocks={list(blocks)}, "
              f"reps={args.reps}, {len(rows)} finished, {len(skipped)} skipped, {time.time() - started:.1f}s\n\n")
    (OUT_DIR / "ugc_candidate_summary.txt").write_text(header + summary + "\n", encoding="utf-8")
    (OUT_DIR / "ugc_candidate.json").write_text(json.dumps({
        "seed": SEED, "blocks": list(blocks), "reps": args.reps, "ip_time": args.ip_time,
        "ip_max_n": args.ip_max_n, "timeout": args.timeout, "skipped_instances": skipped,
        "python": sys.version, "platform": platform.platform(), "networkx": nx.__version__,
        "rows": rows, "elapsed_seconds": time.time() - started,
    }, indent=2, default=str), encoding="utf-8")
    print()
    print(header + summary)


if __name__ == "__main__":
    main()
