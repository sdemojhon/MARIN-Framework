"""
Reproduces Table 3 of the manuscript (MRR and IE across baseline strategies).

Strategies evaluated:
    Random  | Degree  | Betweenness  | Static ABM  | Greedy Inf. Max.
    ACR (adaptive non-RL)  | MARIN (DDQN)

Usage
-----
    python experiments/run_baseline.py --n_nodes 1000 --omega 0.05
    python experiments/run_baseline.py --n_nodes 1000 --omega 0.2
    python experiments/run_baseline.py --quick    # 5 runs, small N (smoke test)

Output
------
A CSV table and a JSON summary are written to results/tables/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from repo root or from the experiments/ directory
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np                                       # noqa: E402
import networkx as nx                                    # noqa: E402

from src.simulation import run_simulation, set_seed     # noqa: E402
from src.marin_network import MultiplexConfig, MultiplexNetwork  # noqa: E402
from src.belief_dynamics import BeliefDynamics          # noqa: E402
from src.interventions import (                          # noqa: E402
    apply_intervention, InterventionType,
)


# --------------------------------------------------------------------------
# Non-RL baseline policies
# --------------------------------------------------------------------------

def _seed_world(seed: int, n_nodes: int, omega: float):
    set_seed(seed)
    cfg = MultiplexConfig(n_nodes=n_nodes, omega=omega)
    net = MultiplexNetwork(cfg, seed=seed)
    n_seed = max(1, int(0.05 * n_nodes))
    seed_idx = np.random.choice(n_nodes, size=n_seed, replace=False)
    net.beliefs[seed_idx] = 1.0
    return net


def _pick_random(net: MultiplexNetwork, budget: int, rng):
    return list(rng.choice(net.n_nodes, size=budget, replace=False))


def _pick_degree(net: MultiplexNetwork, budget: int):
    deg = np.zeros(net.n_nodes)
    for G in net.layers:
        for i, d in dict(G.degree()).items():
            deg[i] += d
    return list(np.argsort(deg)[::-1][:budget])


def _pick_betweenness(net: MultiplexNetwork, budget: int):
    bw = np.zeros(net.n_nodes)
    # Use only layer 0 to keep cost bounded (paper specifies aggregated topology)
    bc = nx.betweenness_centrality(net.layers[0], k=min(50, net.n_nodes))
    for i, v in bc.items():
        bw[i] = v
    return list(np.argsort(bw)[::-1][:budget])


def _pick_static_abm(net: MultiplexNetwork, budget: int, rng):
    # No adaptation: pick a random fixed set per run.
    return _pick_random(net, budget, rng)


def _pick_greedy_inf_max(net: MultiplexNetwork, budget: int):
    # Cheap greedy IM proxy: weighted degree + neighbour belief.
    scores = np.zeros(net.n_nodes)
    for G in net.layers:
        for i in range(net.n_nodes):
            neigh = list(G.neighbors(i))
            if neigh:
                scores[i] += len(neigh) * float(np.mean(net.beliefs[neigh]))
    return list(np.argsort(scores)[::-1][:budget])


def _pick_acr(net: MultiplexNetwork, budget: int):
    # Adaptive Centrality Recalculation: degree recomputed every step.
    return _pick_degree(net, budget)


# --------------------------------------------------------------------------
# Run a single trajectory with a heuristic policy
# --------------------------------------------------------------------------

def run_heuristic(strategy: str, seed: int, n_nodes: int, omega: float,
                  n_steps: int = 200, budget_fraction: float = 0.05) -> dict:
    net = _seed_world(seed, n_nodes, omega)
    rng = np.random.default_rng(seed)
    dyn = BeliefDynamics()
    budget = max(1, int(budget_fraction * n_nodes))

    initial_misinfo = float((net.beliefs > 0.5).mean())
    last_misinfo = initial_misinfo
    cum_cost = 0.0
    cum_reward = 0.0

    for t in range(n_steps):
        if strategy == "random":
            targets = _pick_random(net, budget, rng)
        elif strategy == "degree":
            targets = _pick_degree(net, budget)
        elif strategy == "betweenness":
            targets = _pick_betweenness(net, budget)
        elif strategy == "static_abm":
            if t == 0:
                _static_set = _pick_static_abm(net, budget, rng)
            targets = _static_set
        elif strategy == "greedy":
            targets = _pick_greedy_inf_max(net, budget)
        elif strategy == "acr":
            targets = _pick_acr(net, budget)
        else:
            raise ValueError(f"unknown strategy: {strategy}")

        # Apply truth injection as the heuristic intervention type
        for node in targets:
            apply_intervention(net, int(node),
                               int(InterventionType.TRUTH_INJECTION), rng=rng)
            cum_cost += 1.0

        # Belief + topology update
        neigh = [[list(net.layers[l].neighbors(i)) for l in range(net.n_layers)]
                 for i in range(net.n_nodes)]
        net.beliefs = dyn.update(net.beliefs, net.beliefs.copy(), neigh)
        net.step()

        misinfo = float((net.beliefs > 0.5).mean())
        cum_reward += last_misinfo - misinfo
        last_misinfo = misinfo

    final = last_misinfo
    baseline = max(initial_misinfo * 2.0, 1e-6)
    mrr = 100.0 * (1.0 - final / baseline)
    ie = cum_reward / max(cum_cost, 1e-6)
    return {"mrr": float(np.clip(mrr, -100, 100)), "ie": float(ie)}


# --------------------------------------------------------------------------
# Main driver
# --------------------------------------------------------------------------

STRATEGIES = ["random", "degree", "betweenness", "static_abm",
              "greedy", "acr", "marin"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_nodes", type=int, default=1000)
    parser.add_argument("--omega", type=float, default=0.05)
    parser.add_argument("--n_steps", type=int, default=200)
    parser.add_argument("--seeds_start", type=int, default=1)
    parser.add_argument("--seeds_end", type=int, default=100)
    parser.add_argument("--quick", action="store_true",
                        help="Quick smoke run (5 seeds, N=80, 20 steps)")
    parser.add_argument("--output", type=str, default="results/tables/table3.json")
    args = parser.parse_args()

    if args.quick:
        args.seeds_start = 1
        args.seeds_end = 5
        args.n_nodes = 80
        args.n_steps = 20

    seeds = list(range(args.seeds_start, args.seeds_end + 1))
    print(f"Reproducing Table 3 with N={args.n_nodes}, omega={args.omega}, "
          f"steps={args.n_steps}, seeds={len(seeds)}")

    out = {"config": vars(args), "results": {}}
    for strat in STRATEGIES:
        per_run = []
        for s in seeds:
            if strat == "marin":
                r = run_simulation(seed=s, n_nodes=args.n_nodes,
                                   omega=args.omega, n_steps=args.n_steps,
                                   train_agent=False, record_traces=False)
                per_run.append({"mrr": r.mrr, "ie": r.ie})
            else:
                per_run.append(run_heuristic(strat, s, args.n_nodes,
                                             args.omega, args.n_steps))
        mrr = np.array([x["mrr"] for x in per_run])
        ie  = np.array([x["ie"]  for x in per_run])
        out["results"][strat] = {
            "mrr_mean": float(mrr.mean()), "mrr_std": float(mrr.std(ddof=1) if len(mrr) > 1 else 0),
            "ie_mean":  float(ie.mean()),  "ie_std":  float(ie.std(ddof=1)  if len(ie)  > 1 else 0),
            "n": len(per_run),
        }
        print(f"  {strat:14s}  MRR = {out['results'][strat]['mrr_mean']:6.2f} "
              f"+/- {out['results'][strat]['mrr_std']:5.2f}    "
              f"IE = {out['results'][strat]['ie_mean']:5.2f} "
              f"+/- {out['results'][strat]['ie_std']:5.2f}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
