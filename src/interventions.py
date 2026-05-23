"""
MARIN Framework - Intervention Action Types

Implements the four intervention mechanisms used by the DDQN agent:
    1. Truth injection           - direct correction of a target agent
    2. Bridge-node activation    - boosting cross-community information flow
    3. Echo-chamber disruption   - feed reranking to introduce viewpoint diversity
    4. Cross-layer amplification - promoting verified content across layers

Each intervention mutates the multiplex network and/or the belief vector
in-place and returns a small dict with diagnostics that the simulation
loop collects for reward computation.

References:
    Bhadre & Ghongade (2026), Section 3.3 (Action Space).
"""
from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING, Dict, Optional

import numpy as np

if TYPE_CHECKING:
    from .marin_network import MultiplexNetwork  # for type checkers only


class InterventionType(IntEnum):
    """Discrete intervention identifiers used in the agent's action space."""

    TRUTH_INJECTION = 0
    BRIDGE_NODE = 1
    ECHO_DISRUPTION = 2
    CROSS_LAYER_AMP = 3


INTERVENTION_NAMES = {
    InterventionType.TRUTH_INJECTION: "truth_injection",
    InterventionType.BRIDGE_NODE: "bridge_node_activation",
    InterventionType.ECHO_DISRUPTION: "echo_chamber_disruption",
    InterventionType.CROSS_LAYER_AMP: "cross_layer_amplification",
}


# --------------------------------------------------------------------------
# Individual intervention implementations
# --------------------------------------------------------------------------

def _truth_injection(
    network: "MultiplexNetwork",
    node: int,
    strength: float = 0.30,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, float]:
    """Push the target agent's belief toward 0 (rejection of misinformation)."""
    old = float(network.beliefs[node])
    network.beliefs[node] = float(max(0.0, old - strength))
    return {
        "type": int(InterventionType.TRUTH_INJECTION),
        "node": int(node),
        "delta_belief": old - float(network.beliefs[node]),
        "cost": 1.0,
    }


def _bridge_node_activation(
    network: "MultiplexNetwork",
    node: int,
    rng: Optional[np.random.Generator] = None,
    target_layer: int = 0,
) -> Dict[str, float]:
    """
    Create a small number of new long-range edges from the target node to
    nodes whose beliefs are below the median (i.e., the corrective community),
    on the chosen layer.
    """
    rng = rng or np.random.default_rng()
    n = network.n_nodes
    G = network.layers[target_layer]
    median_b = float(np.median(network.beliefs))
    corrective = np.where(network.beliefs < median_b)[0]
    if len(corrective) == 0:
        return {"type": int(InterventionType.BRIDGE_NODE), "node": int(node),
                "edges_added": 0, "cost": 1.0}
    k = min(3, len(corrective))
    chosen = rng.choice(corrective, size=k, replace=False)
    added = 0
    for j in chosen:
        if j == node:
            continue
        if not G.has_edge(node, int(j)):
            G.add_edge(node, int(j))
            added += 1
    return {
        "type": int(InterventionType.BRIDGE_NODE),
        "node": int(node),
        "edges_added": added,
        "cost": 1.0,
    }


def _echo_chamber_disruption(
    network: "MultiplexNetwork",
    node: int,
    rng: Optional[np.random.Generator] = None,
    target_layer: int = 1,
) -> Dict[str, float]:
    """
    Re-rank the focal agent's neighbourhood: drop one of the most belief-similar
    neighbours and add one belief-dissimilar candidate. Models a feed-ranking
    intervention that broadens viewpoint diversity.
    """
    rng = rng or np.random.default_rng()
    G = network.layers[target_layer]
    neighbours = list(G.neighbors(node))
    if not neighbours:
        return {"type": int(InterventionType.ECHO_DISRUPTION), "node": int(node),
                "swaps": 0, "cost": 1.0}
    b_i = float(network.beliefs[node])
    # Find the most-similar neighbour (smallest belief distance) to drop
    distances = np.array([abs(b_i - float(network.beliefs[j])) for j in neighbours])
    drop_j = neighbours[int(np.argmin(distances))]
    # Find a belief-dissimilar non-neighbour to add
    non_neigh = [j for j in range(network.n_nodes)
                 if j != node and not G.has_edge(node, j)]
    if not non_neigh:
        return {"type": int(InterventionType.ECHO_DISRUPTION), "node": int(node),
                "swaps": 0, "cost": 1.0}
    far_distances = np.array([abs(b_i - float(network.beliefs[j])) for j in non_neigh])
    add_j = non_neigh[int(np.argmax(far_distances))]
    G.remove_edge(node, drop_j)
    G.add_edge(node, add_j)
    return {
        "type": int(InterventionType.ECHO_DISRUPTION),
        "node": int(node),
        "swaps": 1,
        "cost": 1.0,
    }


def _cross_layer_amplification(
    network: "MultiplexNetwork",
    node: int,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, float]:
    """
    If the focal agent has a low belief (i.e., aligned with accurate info),
    copy a small set of its layer-1 (broadcast) edges to layer-2 and layer-3,
    amplifying the corrective signal through more influential channels.
    """
    rng = rng or np.random.default_rng()
    if float(network.beliefs[node]) > 0.5:
        # Only amplify agents who are themselves below the misinformation
        # threshold (i.e., carrying corrective content).
        return {"type": int(InterventionType.CROSS_LAYER_AMP), "node": int(node),
                "edges_added": 0, "cost": 1.0}
    G1 = network.layers[0]
    neighbours = list(G1.neighbors(node))
    if not neighbours:
        return {"type": int(InterventionType.CROSS_LAYER_AMP), "node": int(node),
                "edges_added": 0, "cost": 1.0}
    k = min(2, len(neighbours))
    chosen = rng.choice(neighbours, size=k, replace=False)
    added = 0
    for j in chosen:
        for layer in (1, 2):
            G = network.layers[layer]
            if not G.has_edge(node, int(j)):
                G.add_edge(node, int(j))
                added += 1
    return {
        "type": int(InterventionType.CROSS_LAYER_AMP),
        "node": int(node),
        "edges_added": added,
        "cost": 1.0,
    }


# --------------------------------------------------------------------------
# Dispatcher
# --------------------------------------------------------------------------

_DISPATCH = {
    InterventionType.TRUTH_INJECTION: _truth_injection,
    InterventionType.BRIDGE_NODE: _bridge_node_activation,
    InterventionType.ECHO_DISRUPTION: _echo_chamber_disruption,
    InterventionType.CROSS_LAYER_AMP: _cross_layer_amplification,
}


def apply_intervention(
    network: "MultiplexNetwork",
    node: int,
    intervention_type: int,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, float]:
    """
    Apply a single intervention to the multiplex network.

    Parameters
    ----------
    network : MultiplexNetwork
        The co-evolving multiplex network (mutated in-place).
    node : int
        Target agent index.
    intervention_type : int or InterventionType
        One of the four discrete action types.
    rng : np.random.Generator, optional
        Reproducibility hook.

    Returns
    -------
    dict
        Diagnostics: type, node, and intervention-specific metrics that
        feed into the reward computation.
    """
    itype = InterventionType(int(intervention_type))
    return _DISPATCH[itype](network, int(node), rng=rng)


if __name__ == "__main__":
    # Smoke test against a minimal multiplex network
    from marin_network import MultiplexConfig, MultiplexNetwork

    net = MultiplexNetwork(MultiplexConfig(n_nodes=40, omega=0.0), seed=0)
    rng = np.random.default_rng(0)
    for itype in InterventionType:
        out = apply_intervention(net, node=5, intervention_type=int(itype), rng=rng)
        print(f"{INTERVENTION_NAMES[itype]:30s} -> {out}")
