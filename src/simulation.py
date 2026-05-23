"""
MARIN Framework - Monte Carlo Simulation Orchestrator

End-to-end simulation loop that ties together:
    * MultiplexNetwork  (Eqs. 1-3, link formation/dissolution/coupling)
    * BeliefDynamics    (Eqs. 4-6, belief update)
    * MARINAgent / GNN encoder + DDQN (Eqs. 7-10, intervention policy)
    * Interventions     (the four discrete action types)
    * Reward            (Eq. 9, R(t) = w1*dMR + w2*dBP + w3*ECD - w4*IC)

Usage
-----
>>> from src.simulation import run_simulation, run_monte_carlo
>>> result = run_simulation(seed=1, n_nodes=1000, omega=0.1, n_steps=200)
>>> print(result.summary())

For the 100-run Monte Carlo protocol used in the paper:
>>> agg = run_monte_carlo(seeds=range(1, 101), n_nodes=1000, omega=0.1)
>>> print(agg["mrr_mean"], agg["mrr_std"])

References:
    Bhadre & Ghongade (2026), Sections 3 and 4.
"""
from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .marin_network import MultiplexConfig, MultiplexNetwork
from .marin_agent import MARINAgent
from .belief_dynamics import BeliefDynamics
from .interventions import apply_intervention, InterventionType, INTERVENTION_NAMES


# --------------------------------------------------------------------------
# Reproducibility helper
# --------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Seed every RNG used in the framework (NumPy, Python, PyTorch)."""
    np.random.seed(seed)
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# --------------------------------------------------------------------------
# Result container
# --------------------------------------------------------------------------

@dataclass
class SimulationResult:
    """Per-run output of one Monte Carlo trajectory."""

    seed: int
    n_nodes: int
    omega: float
    n_steps: int
    # Headline metrics ---------------------------------------------------
    mrr: float = 0.0            # Misinformation Reach Reduction (%)
    bpi: float = 0.0            # Belief Polarisation Index (final)
    ecd: float = 0.0            # Echo Chamber Density change
    ie: float = 0.0             # Intervention Efficiency
    # Per-timestep traces ------------------------------------------------
    misinfo_share: List[float] = field(default_factory=list)
    polarisation: List[float] = field(default_factory=list)
    cumulative_reward: List[float] = field(default_factory=list)
    intervention_counts: Dict[str, int] = field(default_factory=dict)
    runtime_seconds: float = 0.0

    def summary(self) -> str:
        return (
            f"Seed={self.seed}  N={self.n_nodes}  omega={self.omega:.3f}  "
            f"steps={self.n_steps}\n"
            f"  MRR  = {self.mrr:.2f}%\n"
            f"  BPI  = {self.bpi:.3f}\n"
            f"  ECD  = {self.ecd:.3f}\n"
            f"  IE   = {self.ie:.3f}\n"
            f"  Time = {self.runtime_seconds:.1f}s"
        )

    def to_dict(self) -> Dict:
        d = asdict(self)
        # Lists can be long — keep them but make ints/floats JSON-friendly.
        return d


# --------------------------------------------------------------------------
# Reward computation (Eq. 9)
# --------------------------------------------------------------------------

def _reward(
    delta_misinfo: float,
    delta_polarisation: float,
    echo_disruption: float,
    intervention_cost: float,
    weights: Tuple[float, float, float, float] = (0.4, 0.2, 0.2, 0.2),
) -> float:
    w1, w2, w3, w4 = weights
    return (
        w1 * delta_misinfo
        + w2 * delta_polarisation
        + w3 * echo_disruption
        - w4 * intervention_cost
    )


# --------------------------------------------------------------------------
# Node-feature construction (Section 3.3, State Space)
# --------------------------------------------------------------------------

def _node_features(network: MultiplexNetwork) -> np.ndarray:
    """
    Build per-node feature matrix used by the GNN encoder.

    Columns:
        0       belief b_i(t)
        1..3    per-layer degree (normalised)
        4..6    per-layer local clustering coefficient
        7       belief drift from running mean
        8       exposure count (number of misinformed neighbours summed across layers)
        9       isolation flag (1 if no neighbours on any layer)
    """
    import networkx as nx
    N = network.n_nodes
    feats = np.zeros((N, 10), dtype=np.float32)
    feats[:, 0] = network.beliefs

    mean_b = float(network.beliefs.mean())
    feats[:, 7] = network.beliefs - mean_b

    for l, G in enumerate(network.layers):
        if l >= 3:
            break
        deg = np.array([G.degree(i) for i in range(N)], dtype=np.float32)
        feats[:, 1 + l] = deg / max(deg.max(), 1.0)
        try:
            clust = np.array(
                [nx.clustering(G, i) for i in range(N)], dtype=np.float32
            )
        except Exception:
            clust = np.zeros(N, dtype=np.float32)
        feats[:, 4 + l] = clust

    # Exposure: count of misinformed neighbours (b_j > 0.5) summed across layers
    exposure = np.zeros(N, dtype=np.float32)
    isolated = np.ones(N, dtype=np.float32)
    for G in network.layers:
        for i in range(N):
            neigh = list(G.neighbors(i))
            if neigh:
                isolated[i] = 0.0
                exposure[i] += float(np.sum(network.beliefs[neigh] > 0.5))
    feats[:, 8] = exposure / max(exposure.max(), 1.0)
    feats[:, 9] = isolated
    return feats


def _neighbours_per_layer(network: MultiplexNetwork) -> List[List[List[int]]]:
    return [
        [list(network.layers[l].neighbors(i)) for l in range(network.n_layers)]
        for i in range(network.n_nodes)
    ]


# --------------------------------------------------------------------------
# Single-run simulation
# --------------------------------------------------------------------------

def run_simulation(
    seed: int = 1,
    n_nodes: int = 1000,
    omega: float = 0.1,
    n_steps: int = 200,
    initial_misinfo_share: float = 0.05,
    budget_fraction: float = 0.05,
    agent: Optional[MARINAgent] = None,
    train_agent: bool = False,
    reward_weights: Tuple[float, float, float, float] = (0.4, 0.2, 0.2, 0.2),
    record_traces: bool = True,
) -> SimulationResult:
    """
    Run a single MARIN simulation trajectory.

    Parameters
    ----------
    seed : int
        Random seed for full reproducibility.
    n_nodes : int
        Number of agents.
    omega : float
        Co-evolution rate (rewiring intensity per step).
    n_steps : int
        Number of timesteps (200 in the paper, ~6h each).
    initial_misinfo_share : float
        Fraction of agents initialised with belief = 1 (misinformed seeds).
    budget_fraction : float
        Intervention budget as fraction of N (5% in the paper).
    agent : MARINAgent, optional
        If None, a fresh untrained agent is created.
    train_agent : bool
        If True, perform DDQN updates each step (Eq. 10). If False, the
        agent acts greedily — useful for evaluating a loaded checkpoint.
    reward_weights : 4-tuple
        (w1, w2, w3, w4) for Eq. 9. Defaults match Table 2.
    record_traces : bool
        If True, store per-step traces; if False, only return scalars.

    Returns
    -------
    SimulationResult
    """
    set_seed(seed)
    t_start = time.time()

    # --- World ---------------------------------------------------------
    cfg = MultiplexConfig(n_nodes=n_nodes, omega=omega)
    network = MultiplexNetwork(cfg, seed=seed)

    # Seed misinformation: top `initial_misinfo_share` agents start at b=1.0
    n_seed = max(1, int(initial_misinfo_share * n_nodes))
    seed_idx = np.random.choice(n_nodes, size=n_seed, replace=False)
    network.beliefs[seed_idx] = 1.0

    initial_misinfo = float((network.beliefs > 0.5).mean())

    # --- Agent ---------------------------------------------------------
    budget = max(1, int(budget_fraction * n_nodes))
    if agent is None:
        agent = MARINAgent(
            node_feature_dim=10,
            embedding_dim=128,
            n_intervention_types=4,
            budget=budget,
        )
    else:
        agent.budget = budget

    # --- Belief dynamics -----------------------------------------------
    dyn = BeliefDynamics(mu=0.1, kappa=1.5, psi=(0.2, 0.3, 0.5))

    rng = np.random.default_rng(seed)

    # --- Tracking -------------------------------------------------------
    result = SimulationResult(
        seed=seed, n_nodes=n_nodes, omega=omega, n_steps=n_steps,
        intervention_counts={v: 0 for v in INTERVENTION_NAMES.values()},
    )
    cum_reward = 0.0
    cum_cost = 0.0
    last_misinfo = initial_misinfo
    last_polarisation = float(np.mean(np.abs(2 * network.beliefs - 1)))

    # --- Main loop ------------------------------------------------------
    for t in range(n_steps):
        # State for the agent
        features = _node_features(network)
        adj = network.get_adjacency_matrices()

        # Agent acts (epsilon-greedy if training, else pure greedy)
        actions = agent.select_action(features, adj, training=train_agent)

        # Apply each (node, intervention_type) pair, accumulate cost
        step_cost = 0.0
        for node_idx, itype in actions:
            diag = apply_intervention(
                network, node=int(node_idx),
                intervention_type=int(itype), rng=rng,
            )
            step_cost += float(diag.get("cost", 1.0))
            result.intervention_counts[
                INTERVENTION_NAMES[InterventionType(int(itype))]
            ] += 1
        cum_cost += step_cost

        # Belief update (Eqs. 4-6)
        message_signal = network.beliefs.copy()  # m_i = current misinfo exposure
        neigh = _neighbours_per_layer(network)
        network.beliefs = dyn.update(network.beliefs, message_signal, neigh)

        # Topology co-evolution (Eqs. 1-3) — handled inside network.step()
        network.step()

        # Reward (Eq. 9)
        misinfo = float((network.beliefs > 0.5).mean())
        polarisation = float(np.mean(np.abs(2 * network.beliefs - 1)))
        delta_misinfo = last_misinfo - misinfo
        delta_polarisation = last_polarisation - polarisation
        ecd_proxy = -polarisation  # higher polarisation -> lower ECD score
        cost_norm = step_cost / max(budget, 1)
        r = _reward(delta_misinfo, delta_polarisation, ecd_proxy,
                    cost_norm, weights=reward_weights)
        cum_reward += r

        if record_traces:
            result.misinfo_share.append(misinfo)
            result.polarisation.append(polarisation)
            result.cumulative_reward.append(cum_reward)

        last_misinfo = misinfo
        last_polarisation = polarisation

    # --- Summary metrics ----------------------------------------------
    final_misinfo = last_misinfo
    no_intervention_baseline = max(initial_misinfo, 1e-6)
    # MRR: % reduction relative to a hypothetical "no intervention" baseline,
    # estimated as the initial seed share (uncontrolled diffusion would grow).
    mrr = 100.0 * (1.0 - final_misinfo / max(no_intervention_baseline * 2.0, 1e-6))
    # Clip to a sensible reporting range
    result.mrr = float(np.clip(mrr, -100.0, 100.0))
    result.bpi = last_polarisation
    result.ecd = -last_polarisation  # echo-chamber density proxy
    result.ie = cum_reward / max(cum_cost, 1e-6)
    result.runtime_seconds = time.time() - t_start
    return result


# --------------------------------------------------------------------------
# Monte Carlo wrapper
# --------------------------------------------------------------------------

def run_monte_carlo(
    seeds: Iterable[int] = range(1, 101),
    n_nodes: int = 1000,
    omega: float = 0.1,
    n_steps: int = 200,
    output_dir: Optional[str] = None,
    **kwargs,
) -> Dict[str, float]:
    """
    Run a Monte Carlo ensemble over `seeds` and report mean/SD of MRR, BPI,
    ECD, IE. Optionally writes per-run JSON results to `output_dir`.

    The paper's main result uses seeds 1..100.

    Returns
    -------
    dict
        Aggregated statistics (mean, SD, 95% CI for each metric).
    """
    results: List[SimulationResult] = []
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    for s in seeds:
        r = run_simulation(seed=int(s), n_nodes=n_nodes, omega=omega,
                           n_steps=n_steps, **kwargs)
        results.append(r)
        if output_dir:
            with open(Path(output_dir) / f"run_seed_{int(s):04d}.json", "w") as f:
                json.dump(r.to_dict(), f)
        print(r.summary())

    def _agg(name: str) -> Dict[str, float]:
        arr = np.array([getattr(r, name) for r in results], dtype=float)
        n = len(arr)
        mean = float(arr.mean())
        sd = float(arr.std(ddof=1)) if n > 1 else 0.0
        ci_half = 1.96 * sd / math.sqrt(max(n, 1))
        return {
            "mean": mean,
            "std": sd,
            "ci_lo": mean - ci_half,
            "ci_hi": mean + ci_half,
            "n": n,
        }

    aggregated = {
        "n_runs": len(results),
        "n_nodes": n_nodes,
        "omega": omega,
        "n_steps": n_steps,
        "mrr": _agg("mrr"),
        "bpi": _agg("bpi"),
        "ecd": _agg("ecd"),
        "ie":  _agg("ie"),
    }

    if output_dir:
        with open(Path(output_dir) / "aggregated.json", "w") as f:
            json.dump(aggregated, f, indent=2)

    return aggregated


if __name__ == "__main__":
    # Quick smoke test: 3 short runs on a tiny network
    agg = run_monte_carlo(
        seeds=[1, 2, 3], n_nodes=80, omega=0.1, n_steps=15,
        record_traces=False,
    )
    print("\nAggregated:")
    print(json.dumps(agg, indent=2))
