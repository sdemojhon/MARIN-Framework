"""
Train a MARIN DDQN agent end-to-end and save the resulting checkpoint.

Implements Algorithm 1 of the manuscript:
    * Curriculum on co-evolution rate omega (10% increase every 500 episodes)
    * Target-network update every tau = 1,000 steps
    * Prioritised experience replay buffer of size 1e6
    * Linear epsilon schedule 1.0 -> 0.01 over 500 episodes

Usage
-----
    # Full training (paper setting, takes hours on a GPU)
    python experiments/train.py --n_nodes 1000 --episodes 2000 --output models/marin_n1000.pt

    # Quick demo checkpoint (~minutes on CPU) for repo packaging
    python experiments/train.py --quick --output models/marin_demo.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np                                       # noqa: E402

from src.simulation import run_simulation, set_seed     # noqa: E402
from src.marin_agent import MARINAgent                  # noqa: E402


def train(
    n_nodes: int = 1000,
    episodes: int = 2000,
    n_steps_per_episode: int = 200,
    initial_omega: float = 0.05,
    curriculum_every: int = 500,
    curriculum_growth: float = 1.10,
    output_path: str = "models/marin_n1000.pt",
    log_every: int = 25,
) -> str:
    set_seed(42)

    budget = max(1, int(0.05 * n_nodes))
    agent = MARINAgent(
        node_feature_dim=10,
        embedding_dim=128,
        n_intervention_types=4,
        budget=budget,
        epsilon_start=1.0,
        epsilon_end=0.01,
        # paper uses linear decay over 500 episodes
        epsilon_decay=(0.01 / 1.0) ** (1.0 / max(500 * n_steps_per_episode, 1)),
    )

    omega = initial_omega
    log = []
    for ep in range(1, episodes + 1):
        if ep > 1 and ep % curriculum_every == 0:
            omega = min(0.5, omega * curriculum_growth)
            print(f"[curriculum] episode {ep}: omega -> {omega:.4f}")

        result = run_simulation(
            seed=ep, n_nodes=n_nodes, omega=omega,
            n_steps=n_steps_per_episode,
            agent=agent, train_agent=True, record_traces=False,
        )
        log.append({
            "episode": ep, "omega": omega,
            "mrr": result.mrr, "bpi": result.bpi, "ie": result.ie,
        })

        if ep % log_every == 0:
            recent = log[-log_every:]
            mrr_mean = float(np.mean([r["mrr"] for r in recent]))
            print(f"ep {ep:5d}  omega={omega:.3f}  "
                  f"MRR(last {log_every} avg) = {mrr_mean:6.2f}%  "
                  f"eps={agent.epsilon:.3f}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    agent.save(output_path)
    print(f"\nSaved checkpoint: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_nodes", type=int, default=1000)
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--n_steps_per_episode", type=int, default=200)
    parser.add_argument("--output", type=str, default="models/marin_n1000.pt")
    parser.add_argument("--quick", action="store_true",
                        help="Quick demo run (N=80, 30 episodes, 20 steps each)")
    args = parser.parse_args()

    if args.quick:
        args.n_nodes = 80
        args.episodes = 30
        args.n_steps_per_episode = 20
        args.output = "models/marin_demo.pt"

    train(
        n_nodes=args.n_nodes,
        episodes=args.episodes,
        n_steps_per_episode=args.n_steps_per_episode,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
