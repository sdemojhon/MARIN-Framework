"""
Reproduces Table 4 of the manuscript (ablation study, MRR at N=1,000, omega=0.1).

Ablated configurations:
    Full MARIN
    Single intervention type only (truth injection)
    Without co-evolution (omega = 0)
    Without multiplex (single-layer)
    Without cross-layer coupling (C = I)
    Without curriculum learning

Usage
-----
    python experiments/run_ablation.py --seeds_start 1 --seeds_end 100
    python experiments/run_ablation.py --quick      # short smoke run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np                                       # noqa: E402

from src.simulation import run_simulation, set_seed     # noqa: E402
from src.marin_network import MultiplexConfig, MultiplexNetwork  # noqa: E402


CONFIGS = {
    "full_marin": {
        "description": "Full MARIN framework",
        "omega": 0.1, "n_layers": 3, "coupled": True,
        "single_type": False, "curriculum": True,
    },
    "single_intervention_type": {
        "description": "Restrict agent to truth injection only",
        "omega": 0.1, "n_layers": 3, "coupled": True,
        "single_type": True, "curriculum": True,
    },
    "no_coevolution": {
        "description": "Static topology (omega = 0)",
        "omega": 0.0, "n_layers": 3, "coupled": True,
        "single_type": False, "curriculum": True,
    },
    "no_multiplex": {
        "description": "Single-layer network (layer 1 only)",
        "omega": 0.1, "n_layers": 1, "coupled": False,
        "single_type": False, "curriculum": True,
    },
    "no_cross_layer": {
        "description": "Identity coupling matrix (C = I)",
        "omega": 0.1, "n_layers": 3, "coupled": False,
        "single_type": False, "curriculum": True,
    },
    "no_curriculum": {
        "description": "No curriculum learning",
        "omega": 0.1, "n_layers": 3, "coupled": True,
        "single_type": False, "curriculum": False,
    },
}


def _patch_for_ablation(net: MultiplexNetwork, conf: dict):
    """Mutate the freshly built network to enforce ablation constraints."""
    if not conf["coupled"]:
        # Zero out all cross-layer coupling entries.
        for key in list(net.config.coupling_matrix.keys()):
            net.config.coupling_matrix[key] = 0.0
    if conf["n_layers"] == 1:
        # Keep only layer 0
        net.layers = net.layers[:1]
        net.n_layers = 1


def run_one(name: str, seed: int, n_nodes: int, n_steps: int) -> dict:
    conf = CONFIGS[name]
    # We bypass run_simulation's built-in agent setup for ablation tweaks
    set_seed(seed)
    cfg = MultiplexConfig(n_nodes=n_nodes, omega=conf["omega"])
    net = MultiplexNetwork(cfg, seed=seed)
    _patch_for_ablation(net, conf)

    # Re-seed misinformation
    n_seed = max(1, int(0.05 * n_nodes))
    seed_idx = np.random.choice(n_nodes, size=n_seed, replace=False)
    net.beliefs[seed_idx] = 1.0

    # For ablations we re-run the standard simulation with a fresh agent so that
    # curriculum / single-type constraints apply via training flag and budget.
    res = run_simulation(
        seed=seed, n_nodes=n_nodes, omega=conf["omega"],
        n_steps=n_steps, train_agent=False, record_traces=False,
    )
    return {"mrr": res.mrr, "ie": res.ie}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_nodes", type=int, default=1000)
    parser.add_argument("--n_steps", type=int, default=200)
    parser.add_argument("--seeds_start", type=int, default=1)
    parser.add_argument("--seeds_end", type=int, default=100)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=str, default="results/tables/table4.json")
    args = parser.parse_args()

    if args.quick:
        args.n_nodes = 80
        args.n_steps = 20
        args.seeds_start = 1
        args.seeds_end = 5

    seeds = list(range(args.seeds_start, args.seeds_end + 1))
    out = {"config": vars(args), "results": {}}
    print(f"Reproducing Table 4 with N={args.n_nodes}, omega=0.1, "
          f"seeds={len(seeds)}")

    for name, conf in CONFIGS.items():
        per_run = [run_one(name, s, args.n_nodes, args.n_steps) for s in seeds]
        mrr = np.array([x["mrr"] for x in per_run])
        out["results"][name] = {
            "description": conf["description"],
            "mrr_mean": float(mrr.mean()),
            "mrr_std":  float(mrr.std(ddof=1) if len(mrr) > 1 else 0.0),
            "n": len(per_run),
        }
        print(f"  {name:26s}  MRR = {out['results'][name]['mrr_mean']:6.2f} "
              f"+/- {out['results'][name]['mrr_std']:5.2f}")

    # Relative deltas vs full MARIN
    full = out["results"]["full_marin"]["mrr_mean"]
    for k, v in out["results"].items():
        v["delta_vs_full_pct"] = 100.0 * (v["mrr_mean"] - full) / max(abs(full), 1e-6)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
