"""
Reproduces Table 5 of the manuscript (MRR across N = 1,000 / 2,000 / 5,000).

Strategies evaluated for each network size:
    Greedy Influence Maximisation
    ACR (Adaptive Centrality Recalculation)
    MARIN

Usage
-----
    python experiments/run_scalability.py --omega 0.1
    python experiments/run_scalability.py --quick
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np                                       # noqa: E402

from experiments.run_baseline import run_heuristic       # noqa: E402
from src.simulation import run_simulation                # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--omega", type=float, default=0.1)
    parser.add_argument("--n_steps", type=int, default=200)
    parser.add_argument("--seeds_start", type=int, default=1)
    parser.add_argument("--seeds_end", type=int, default=100)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=str, default="results/tables/table5.json")
    args = parser.parse_args()

    if args.quick:
        sizes = [80, 160]
        args.n_steps = 20
        args.seeds_start = 1
        args.seeds_end = 5
    else:
        sizes = [1000, 2000, 5000]

    seeds = list(range(args.seeds_start, args.seeds_end + 1))
    out = {"config": vars(args), "results": {}}

    for N in sizes:
        print(f"\n=== N = {N} ===")
        for strat in ["greedy", "acr", "marin"]:
            per_run = []
            for s in seeds:
                if strat == "marin":
                    r = run_simulation(seed=s, n_nodes=N, omega=args.omega,
                                       n_steps=args.n_steps, train_agent=False,
                                       record_traces=False)
                    per_run.append(r.mrr)
                else:
                    per_run.append(
                        run_heuristic(strat, s, N, args.omega, args.n_steps)["mrr"]
                    )
            arr = np.array(per_run)
            n = len(arr)
            mean = float(arr.mean())
            sd = float(arr.std(ddof=1)) if n > 1 else 0.0
            ci = 1.96 * sd / math.sqrt(max(n, 1))
            key = f"{strat}_N{N}"
            out["results"][key] = {
                "strategy": strat, "N": N,
                "mrr_mean": mean, "mrr_std": sd,
                "ci_lo": mean - ci, "ci_hi": mean + ci, "n": n,
            }
            print(f"  {strat:8s}  MRR = {mean:6.2f} +/- {sd:5.2f}  "
                  f"95% CI [{mean - ci:6.2f}, {mean + ci:6.2f}]")

    # Cohen's d, MARIN vs Greedy, for each N
    out["cohens_d_marin_vs_greedy"] = {}
    for N in sizes:
        m1 = out["results"][f"marin_N{N}"]["mrr_mean"]
        s1 = out["results"][f"marin_N{N}"]["mrr_std"]
        m2 = out["results"][f"greedy_N{N}"]["mrr_mean"]
        s2 = out["results"][f"greedy_N{N}"]["mrr_std"]
        sp = math.sqrt(0.5 * (s1**2 + s2**2)) or 1e-6
        d = (m1 - m2) / sp
        out["cohens_d_marin_vs_greedy"][f"N{N}"] = round(d, 3)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
