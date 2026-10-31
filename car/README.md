# car — Certified Approximation Ratio experiments

Reproducible stress suite for the Esperanza algorithm. It generates **30,000
instances**, certifies every optimum with an exact **MILP** (SciPy / HiGHS),
and reports the approximation ratio `OPT / ALG` per instance and per family.

## Run

From the repository root:

```bash
python car/experiment.py                 # full 30,000-instance run
python car/experiment.py --self-check    # verify MILP vs brute force first
python car/experiment.py --total 3000    # quick scaled-down pass
python car/experiment.py --seed 7        # different RNG seed
```

Requirements are the package's own: `salvador` (>= 0.1.4, which supplies the
vertex cover and brings `networkx`, `numpy`, `scipy` >= 1.15 for
`scipy.optimize.milp`). The scripts import the repository's own `esperanza/`,
so the package itself does not need to be installed.

## Instance schedule (default `--total 30000`)

| family              | count  | description                                        |
| ------------------- | -----: | -------------------------------------------------- |
| gnp                 | 15,000 | `G(n, p)`, n in 6–16, p in 0.15–0.85               |
| bipartite_gnp       |  6,000 | random bipartite, sides 3–14, p in 0.10–0.50       |
| regular             |  3,000 | random d-regular, d in 3–5, n in 8–16              |
| star                |  2,000 | `K_{1,k}`, k in 3–80                               |
| complete            |  1,000 | `K_n`, n in 3–30                                   |
| complete_bipartite  |  1,000 | `K_{a,b}`, a in 2–12, b in a–20                    |
| amplified           |  2,000 | disjoint copies (2–8) and independent blow-ups (t = 2–6) of the worst connected small bases found in the random phases |

The amplification family probes exactly the phenomenon reported in the paper:
whether disjoint copies and independent blow-ups preserve or amplify the worst
observed ratio.

## MILP certification

For each instance the exact independence number is obtained by solving

```
maximize   sum_v x_v
subject to x_u + x_v <= 1   for every edge (u, v)
           x_v in {0, 1}
```

to provable optimality with `scipy.optimize.milp`. Every returned set is also
re-verified to be independent before its size is recorded.

## Outputs (in `car/results/`)

- `results.csv` — one row per instance: `family, n, m, alg, opt, ratio`
- `summary.txt` — worst / mean ratio per family and overall
- `summary_table.tex` — booktabs rows ready to paste into the paper
- `worst_instance.txt` — edge list of the worst-ratio instance found

## UGC candidate: Khot–Regev YES instances (`ugc_candidate.py`)

Esperanza's vertex cover is `C_E = V - S` for `S = find_independent_set(G)`.
Esperanza starts from the complement of the Salvador cover `C_S` and only
replaces it by strictly larger independent sets, so `|C_E| <= |C_S|`, and
Salvador guarantees a 2-approximation, `|C_S| <= 2 tau`.

The script tests, on every instance, the inequality

```
|C_E| <= 2 tau - |M|,   M = nx.approximation.min_maximal_matching(G)
```

with certified bounds `L <= tau <= U` only (LP relaxation and a HiGHS integer
program with a time limit for `L`; the planted cover and the integer program
for `U`): **certified** when `|C_E| <= 2L - |M|`, **violated** when
`|C_E| > 2U - |M|`, **open** otherwise. The Salvador cover is tested the same
way, and the 3/2 certificate `|C_E| <= 3/2 L` is reported. In the
independent-set form of the Khot–Regev gap, the script also reports
`|S| / n` and the certified range `n - U <= alpha <= n - L`.

The instances are the YES-type graphs of the Khot–Regev reduction of Part D
of the Salvador experiments: same generator, same seed (`20261009`) and same
instance names, ten families (base `k=4, p=0.45, eta=0, d=4, s=16`; `k=3`,
`k=6`; `p=0.35`, `p=0.49`; `eta=0.02`, `eta=0.1`; `d=8`; `s=32`; linear UG).

```bash
python car/ugc_candidate.py                    # N = 40, 80, 160, 320 (640 to 10,240 vertices)
python car/ugc_candidate.py --quick            # N = 40, 80 (the sizes of Salvador's Part D)
python car/ugc_candidate.py --blocks 640 1280  # larger instances
python car/ugc_candidate.py --timeout 0        # no per-instance time limit
```

Each instance runs in a worker process with a time limit (`--timeout`, default
900 s); instances over the limit are listed as skipped. Esperanza 0.1.6 runs in
near-linear time; the integer program (`--ip-time`, `--ip-max-n`) is usually
the slowest step.

Outputs (in `car/results/`):

- `ugc_candidate.csv` — one row per instance (written as the run progresses)
- `ugc_candidate_summary.txt` — per family: certified / open / violated for
  Esperanza and Salvador, Esperanza strictly below Salvador, optimal, 3/2
  certificates, largest `|C_E| / L`, smallest `|S| / n`
- `ugc_candidate.json` — all rows and run parameters
- `ugc_counterexamples/` — edge list of every instance on which the
  inequality is violated or open

## Bounded-degree graphs: the Austrin–Khot–Safra barrier (`bounded_degree.py`)

Under the Unique Games Conjecture, Maximum Independent Set on graphs of
maximum degree `Delta` is hard to approximate within `O(Delta / log^2 Delta)`
(Austrin, Khot and Safra, Theory of Computing 7(3), 2011). The barrier is
asymptotic in `Delta` with an unspecified constant, so the script reports how
the measured ratio grows with `Delta` next to the scale `Delta / log2(Delta)^2`.

For every instance it computes `S = find_independent_set(G)` and certified
bounds `alpha_lower <= alpha <= alpha_upper` (lower: `|S|`, the planted set,
the baselines and the integer-program primal value; upper: `n` minus the LP
relaxation of vertex cover and the integer-program dual bound, HiGHS with a
time limit), and reports `alpha_upper / |S|`, `alpha_lower / |S|` and
`(alpha_upper / |S|) / (Delta / log2(Delta)^2)`. The Caro–Wei baseline of
Esperanza and the complement of the Salvador cover are measured on the same
graphs.

| family    | description |
| --------- | ----------- |
| regular   | random `Delta`-regular, `Delta` in 3–128 |
| planted   | maximum degree `<= Delta`, planted independent set of `beta n` vertices, `Delta` in 4–64, `beta` in {0.35, 0.45} |
| kr        | Khot–Regev YES graphs (generator of `ugc_candidate.py`) with `s` in {4, 8, 16} samples per block, degree `<= d s` |

```bash
python car/bounded_degree.py               # n = 256, 1024, 3 instances per setting
python car/bounded_degree.py --quick       # n = 256, 1 instance per setting
python car/bounded_degree.py --sizes 4096  # other sizes
python car/bounded_degree.py --ip-max-n 0  # skip the integer program
```

Outputs (in `car/results/`): `bounded_degree.csv` (one row per instance),
`bounded_degree_summary.txt` (per family and `Delta`) and `bounded_degree.json`.

