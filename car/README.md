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

Requirements are the package's own: `networkx`, `numpy`, `scipy` (>= 1.15 for
`scipy.optimize.milp`).

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
