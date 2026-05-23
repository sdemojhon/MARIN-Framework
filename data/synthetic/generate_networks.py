"""
Generate the synthetic multiplex networks used in the paper experiments.

For each network size N in {1000, 2000, 5000} and each seed in 1..100, this
script writes the three layer adjacency matrices and the initial belief vector
to disk so that downstream experiments can replay them deterministically.

Usage
-----
    python data/synthetic/generate_networks.py --output data/synthetic/
    python data/synthetic/generate_networks.py --quick    # 3 seeds, N=80
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np                                       # noqa: E402
import networkx as nx                                    # noqa: E402

from src.marin_network import MultiplexConfig, MultiplexNetwork  # noqa: E402


def generate(N: int, seed: int) -> dict:
    cfg = MultiplexConfig(n_nodes=N, omega=0.0)
    net = MultiplexNetwork(cfg, seed=seed)
    return {
        "N": N,
        "seed": seed,
        "beliefs_initial": net.beliefs.astype(np.float32),
        "adjacency": [nx.to_numpy_array(G, dtype=np.uint8) for G in net.layers],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="data/synthetic/")
    parser.add_argument("--seeds_start", type=int, default=1)
    parser.add_argument("--seeds_end", type=int, default=100)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        sizes = [80]
        args.seeds_start = 1
        args.seeds_end = 3
    else:
        sizes = [1000, 2000, 5000]

    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    for N in sizes:
        for s in range(args.seeds_start, args.seeds_end + 1):
            data = generate(N, s)
            out = out_root / f"N{N}_seed{s:03d}.npz"
            np.savez_compressed(
                out,
                beliefs_initial=data["beliefs_initial"],
                layer1=data["adjacency"][0],
                layer2=data["adjacency"][1],
                layer3=data["adjacency"][2],
            )
            print(f"wrote {out}")

    print(f"\nDone. {len(sizes) * (args.seeds_end - args.seeds_start + 1)} networks.")


if __name__ == "__main__":
    main()
