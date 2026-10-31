#!/usr/bin/env python3
"""
car/bounded_degree.py: Esperanza on bounded-degree graphs, measured against
the Unique-Games barrier of Austrin, Khot and Safra.

Austrin, Khot and Safra ("Inapproximability of Vertex Cover and Independent
Set in Bounded Degree Graphs", Theory of Computing 7(3), 2011) prove that,
under the Unique Games Conjecture, Maximum Independent Set on graphs of
maximum degree Delta is hard to approximate within O(Delta / log^2 Delta).
The best known polynomial-time ratio is O(Delta log log Delta / log Delta)
(Halperin). The barrier is asymptotic in Delta and its constant is not
explicit, so finite graphs can only show how the measured ratio grows with
Delta; they cannot confirm or refute the barrier.

For every instance the script computes S = find_independent_set(G) and the
certified range alpha_lower <= alpha(G) <= alpha_upper, where

    alpha_lower = max(|S|, planted independent set, integer-program primal value)
    alpha_upper = min(n - ceil(LP relaxation of vertex cover), integer-program dual bound)

(the LP relaxation is half the maximum matching of the bipartite double
cover; the integer program is scipy.optimize.milp / HiGHS with a time
limit). It reports

    ratio_upper = alpha_upper / |S|   (certified upper bound on OPT / ALG)
    ratio_lower = alpha_lower / |S|   (lower bound; exact when alpha is proved)
    aks_scale   = Delta / log2(Delta)^2
    normalized  = ratio_upper / aks_scale

with Delta the maximum degree of the instance, and the same ratios for two
baselines run on the same graph: the Caro-Wei minimum-degree greedy of
Esperanza (|I| >= sum 1/(d(v)+1)) and the complement of the Salvador cover.

Families
--------
    regular   random Delta-regular graphs
    planted   bounded-degree graphs with a planted independent set I of
              beta * n vertices: every vertex of I has Delta random
              neighbours outside I, and the remaining stubs outside I are
              paired at random (loops and repeated edges dropped), so every
              degree is at most Delta and alpha >= |I|
    kr        YES-type Khot-Regev graphs with few samples per block (the
              generator of car/ugc_candidate.py), whose degree is bounded by
              d * s; Delta is measured on the instance

Run from the repository root (the package does not need to be installed;
the repository's own esperanza/ is used; salvador >= 0.1.4 must be importable):

    python car/bounded_degree.py               # n = 256, 1024
    python car/bounded_degree.py --quick       # n = 256, one instance per setting
    python car/bounded_degree.py --sizes 4096  # other sizes
    python car/bounded_degree.py --ip-max-n 0  # skip the integer program

Each instance is built and evaluated in a worker process with a time limit
(--timeout, default 900 s); instances over the limit are listed as skipped.

Outputs (in car/results/):

    bounded_degree.csv          one row per instance
    bounded_degree_summary.txt  per family and Delta
    bounded_degree.json         all rows and run parameters
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
CAR_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CAR_DIR.parent))
sys.path.insert(0, str(CAR_DIR))
from esperanza.algorithm import find_independent_set, pure_caro_wei_baseline  # noqa: E402
from ugc_candidate import khot_regev_yes, lp_lower_bound, shuffle_labels  # noqa: E402

try:
    from salvador.algorithm import find_vertex_cover as salvador_vertex_cover  # noqa: E402
except ImportError as exc:  # pragma: no cover
    raise SystemExit("salvador >= 0.1.4 is required: pip install \"salvador>=0.1.4\"") from exc

SEED = 20261031
OUT_DIR = CAR_DIR / "results"

REGULAR_DEGREES = (3, 4, 5, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128)
PLANTED_DEGREES = (4, 8, 16, 32, 64)
PLANTED_BETAS = (0.35, 0.45)
KR_SETTINGS = (  # (N, d, s): Khot-Regev blocks, constraint degree, samples per block
    (16, 3, 4), (16, 3, 8), (16, 4, 8), (16, 4, 16),
    (64, 3, 4), (64, 3, 8), (64, 4, 8), (64, 4, 16),
)


def instance_rng(name: str) -> random.Random:
    return random.Random(f"{SEED}:{name}")


# ----------------------------------------------------------------------------
# Instance families
# ----------------------------------------------------------------------------

def gen_regular(n: int, delta: int, rng: random.Random):
    if (n * delta) % 2:
        n += 1
    return nx.random_regular_graph(delta, n, seed=rng.randrange(2**31)), set()


def gen_planted(n: int, delta: int, beta: float, rng: random.Random):
    """Graph of maximum degree <= delta with a planted independent set of beta * n vertices."""
    size_i = int(round(beta * n))
    I = list(range(size_i))
    R = list(range(size_i, n))
    G = nx.Graph()
    G.add_nodes_from(range(n))
    stubs = [v for v in R for _ in range(delta)]
    rng.shuffle(stubs)
    pos = 0
    for u in I:
        for _ in range(delta):
            if pos >= len(stubs):
                break
            v = stubs[pos]
            pos += 1
            if not G.has_edge(u, v):
                G.add_edge(u, v)
    rest = stubs[pos:]
    rng.shuffle(rest)
    for a in range(0, len(rest) - 1, 2):
        u, v = rest[a], rest[a + 1]
        if u != v and not G.has_edge(u, v):
            G.add_edge(u, v)
    G, planted = shuffle_labels(G, set(I), rng)
    return G, planted


def gen_kr(N: int, d: int, s: int, rng: random.Random):
    G, I = khot_regev_yes(N=N, d=d, k=4, p=0.45, eta=0.0, s=s, linear=False, rng=rng)
    return shuffle_labels(G, I, rng)


def instances(sizes, reps: int):
    """(name, family, params) for every instance."""
    for n in sizes:
        for delta in REGULAR_DEGREES:
            if delta >= n:
                continue
            for r in range(reps):
                yield f"regular_n{n}_D{delta}_r{r}", "regular", {"n": n, "delta": delta}
        for delta in PLANTED_DEGREES:
            for beta in PLANTED_BETAS:
                if delta >= (1 - beta) * n:
                    continue
                for r in range(reps):
                    yield (f"planted_n{n}_D{delta}_b{beta}_r{r}", "planted",
                           {"n": n, "delta": delta, "beta": beta})
    for N, d, s in KR_SETTINGS:
        for r in range(reps):
            yield f"kr_N{N}_d{d}_s{s}_r{r}", "kr", {"N": N, "d": d, "s": s}


def build(family: str, params: dict, rng: random.Random):
    if family == "regular":
        return gen_regular(params["n"], params["delta"], rng)
    if family == "planted":
        return gen_planted(params["n"], params["delta"], params["beta"], rng)
    return gen_kr(params["N"], params["d"], params["s"], rng)


# ----------------------------------------------------------------------------
# Certified bounds on alpha
# ----------------------------------------------------------------------------

def mis_integer_program(G: nx.Graph, time_limit: float):
    """Independent set integer program: (primal value or None, proved optimal, upper bound or None)."""
    nodes = list(G.nodes())
    idx = {v: i for i, v in enumerate(nodes)}
    E = list(G.edges())
    n = len(nodes)
    if not E:
        return n, True, n
    rows, cols, data = [], [], []
    for r, (u, v) in enumerate(E):
        rows.extend((r, r))
        cols.extend((idx[u], idx[v]))
        data.extend((1.0, 1.0))
    A = csr_matrix((data, (rows, cols)), shape=(len(E), n))
    res = milp(c=-np.ones(n), constraints=LinearConstraint(A, -np.inf, 1.0),
               integrality=np.ones(n), bounds=Bounds(0, 1),
               options={"time_limit": time_limit})
    dual = getattr(res, "mip_dual_bound", None)
    upper = math.floor(-dual + 1e-6) if dual is not None else None
    if res.x is None:
        return None, False, upper
    value = int(round(-res.fun))
    optimal = res.status == 0
    return value, optimal, (value if optimal else upper)


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------

def is_independent(G: nx.Graph, S: set) -> bool:
    return not any(u in S and v in S for u, v in G.edges())


def run_instance(name: str, family: str, params: dict, ip_time: float, ip_max_n: int) -> dict:
    rng = instance_rng(name)
    G, planted = build(family, params, rng)
    assert is_independent(G, planted), f"{name}: planted set is not independent"
    n, m = G.number_of_nodes(), G.number_of_edges()
    delta = max((d for _, d in G.degree()), default=0)

    t0 = time.perf_counter()
    S = find_independent_set(G)
    t_esp = time.perf_counter() - t0
    assert is_independent(G, S), f"{name}: Esperanza set is not independent"

    W = G.copy()
    isolates = set(nx.isolates(W))
    W.remove_nodes_from(isolates)
    cw = (pure_caro_wei_baseline(W) | isolates) if W.number_of_nodes() else set(isolates)
    sal = ((set(W.nodes()) - salvador_vertex_cover(W)) | isolates) if W.number_of_nodes() else set(isolates)
    assert is_independent(G, cw) and is_independent(G, sal), name
    caro_wei_sum = sum(1.0 / (d + 1) for _, d in G.degree())

    alpha_lower = max(len(S), len(planted), len(cw), len(sal))
    alpha_upper = n - lp_lower_bound(G)
    ip_status = "skipped"
    t0 = time.perf_counter()
    if ip_max_n and n <= ip_max_n:
        value, optimal, upper = mis_integer_program(G, ip_time)
        if value is not None:
            alpha_lower = max(alpha_lower, value)
        if upper is not None:
            alpha_upper = min(alpha_upper, upper)
        ip_status = "optimal" if optimal else "time limit"
    t_ip = time.perf_counter() - t0
    alpha_upper = max(alpha_upper, alpha_lower)
    alpha = alpha_lower if alpha_lower == alpha_upper else None

    aks = delta / math.log2(delta) ** 2 if delta >= 2 else float("nan")
    e = len(S)
    row = {
        "family": family, "name": name, **{k: params.get(k, "") for k in ("n_param", "beta", "N", "d", "s")},
        "delta_param": params["delta"] if "delta" in params else params["d"] * params["s"],
        "n": n, "m": m, "delta": delta, "planted": len(planted),
        "alpha_lower": alpha_lower, "alpha_upper": alpha_upper, "alpha": alpha, "ip_status": ip_status,
        "esperanza": e, "caro_wei": len(cw), "salvador_complement": len(sal),
        "caro_wei_sum": round(caro_wei_sum, 3),
        "ratio_upper": alpha_upper / e, "ratio_lower": alpha_lower / e,
        "caro_wei_ratio_upper": alpha_upper / len(cw), "salvador_ratio_upper": alpha_upper / len(sal),
        "aks_scale": aks, "normalized": (alpha_upper / e) / aks if aks == aks and aks > 0 else float("nan"),
        "esperanza_optimal": alpha is not None and e == alpha,
        "seconds_esperanza": round(t_esp, 3), "seconds_ip": round(t_ip, 3),
    }
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


FIELDS = ["family", "name", "n_param", "beta", "N", "d", "s", "delta_param", "n", "m", "delta", "planted",
          "alpha_lower", "alpha_upper", "alpha", "ip_status", "esperanza", "caro_wei",
          "salvador_complement", "caro_wei_sum", "ratio_upper", "ratio_lower",
          "caro_wei_ratio_upper", "salvador_ratio_upper", "aks_scale", "normalized",
          "esperanza_optimal", "seconds_esperanza", "seconds_ip"]


def summarise(rows: list[dict]) -> str:
    hdr = (f"{'family':8} {'D set':>5} {'Delta':>5} {'inst':>4} {'exact':>5} {'opt':>4} {'max up':>8} {'mean up':>8} "
           f"{'max low':>8} {'CW up':>8} {'Salv up':>8} {'D/log2D^2':>9} {'max norm':>9}")
    lines = [hdr, "-" * len(hdr)]
    keys = sorted({(r["family"], r["delta_param"]) for r in rows})
    for fam, dp in keys:
        rs = [r for r in rows if r["family"] == fam and r["delta_param"] == dp]
        delta = max(r["delta"] for r in rs)
        norms = [r["normalized"] for r in rs if r["normalized"] == r["normalized"]]
        aks = delta / math.log2(delta) ** 2 if delta >= 2 else float("nan")
        lines.append(
            f"{fam:8} {dp:>5} {delta:>5} {len(rs):>4} {sum(r['alpha'] is not None for r in rs):>5} "
            f"{sum(r['esperanza_optimal'] for r in rs):>4} "
            f"{max(r['ratio_upper'] for r in rs):>8.4f} {sum(r['ratio_upper'] for r in rs) / len(rs):>8.4f} "
            f"{max(r['ratio_lower'] for r in rs):>8.4f} {max(r['caro_wei_ratio_upper'] for r in rs):>8.4f} "
            f"{max(r['salvador_ratio_upper'] for r in rs):>8.4f} {aks:>9.4f} "
            f"{max(norms) if norms else float('nan'):>9.4f}")
    lines += [
        "",
        "up = alpha_upper / |S| (certified upper bound on OPT/ALG); low = alpha_lower / |S|;",
        "D set = Delta of the setting (d * s for kr); Delta = largest measured maximum degree;",
        "exact = instances with alpha proved; opt = Esperanza optimal among them;",
        "CW up, Salv up = the same upper bound for the Caro-Wei baseline and the Salvador complement;",
        "D/log2D^2 = Delta / log2(Delta)^2; max norm = max of (alpha_upper / |S|) / (Delta / log2(Delta)^2).",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="n = 256, one instance per setting")
    ap.add_argument("--sizes", type=int, nargs="+", default=None,
                    help="graph sizes n for the regular and planted families (default 256 1024)")
    ap.add_argument("--reps", type=int, default=None, help="instances per setting (default 3; 1 with --quick)")
    ap.add_argument("--ip-time", type=float, default=60.0, help="integer program time limit in seconds (default 60)")
    ap.add_argument("--ip-max-n", type=int, default=1100,
                    help="run the integer program up to this many vertices (0 = never; default 1100)")
    ap.add_argument("--timeout", type=float, default=900.0,
                    help="per-instance time limit in seconds (default 900; 0 = no limit)")
    args = ap.parse_args()

    sizes = args.sizes or ((256,) if args.quick else (256, 1024))
    reps = args.reps or (1 if args.quick else 3)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()
    rows, skipped = [], []
    state: dict = {}
    with (OUT_DIR / "bounded_degree.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        try:
            for name, family, params in instances(sizes, reps):
                params = {**params, "n_param": params.get("n", "")}
                row = run_with_timeout((name, family, params, args.ip_time, args.ip_max_n), args.timeout, state)
                if row is None:
                    print(f"  {name:34} skipped (more than {args.timeout:g} s)", flush=True)
                    skipped.append(name)
                    continue
                rows.append(row)
                writer.writerow(row)
                fh.flush()
                print(f"  {name:34} n={row['n']:>6} Delta={row['delta']:>4} |S|={row['esperanza']:>6}"
                      f" alpha in [{row['alpha_lower']}, {row['alpha_upper']}]"
                      f" up={row['ratio_upper']:.4f} norm={row['normalized']:.4f}"
                      f" {row['seconds_esperanza']:7.1f}s", flush=True)
        finally:
            if state.get("pool") is not None:
                state["pool"].terminate()
                state["pool"].join()

    summary = summarise(rows) if rows else "no instance finished"
    header = (f"Esperanza on bounded-degree graphs: seed={SEED}, sizes={list(sizes)}, reps={reps}, "
              f"{len(rows)} finished, {len(skipped)} skipped, {time.time() - started:.1f}s\n\n")
    (OUT_DIR / "bounded_degree_summary.txt").write_text(header + summary + "\n", encoding="utf-8")
    (OUT_DIR / "bounded_degree.json").write_text(json.dumps({
        "seed": SEED, "sizes": list(sizes), "reps": reps, "ip_time": args.ip_time,
        "ip_max_n": args.ip_max_n, "timeout": args.timeout, "skipped_instances": skipped,
        "python": sys.version, "platform": platform.platform(), "networkx": nx.__version__,
        "rows": rows, "elapsed_seconds": time.time() - started,
    }, indent=2, default=str), encoding="utf-8")
    print()
    print(header + summary)


if __name__ == "__main__":
    main()
