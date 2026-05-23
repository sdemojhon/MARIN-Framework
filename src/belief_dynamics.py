"""
MARIN Framework - Agent Belief Dynamics

Implements the belief update mechanism (Equations 4-6 of the manuscript).

Convention:
    b_i(t) in [0, 1] is agent i's susceptibility to/alignment with the
    misinformation narrative under study.
        b_i = 0  ->  agent firmly rejects the misinformation
        b_i = 1  ->  agent fully accepts the misinformation
    Exposure to misinformation pushes b_i toward 1; successful interventions
    push b_i toward 0.

References:
    Bhadre & Ghongade (2026), Section 3.2.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np


def bayesian_update(b_i: float, m: float, kappa: float = 1.5) -> float:
    """
    Bayesian update component (Equation 5):

        B_i(m, t) = b_i^kappa * m
                  / (b_i^kappa * m + (1 - b_i)^kappa * (1 - m))

    Parameters
    ----------
    b_i : float
        Current belief in [0, 1].
    m : float
        Message-content signal in [0, 1]. Higher m means the message carries
        stronger misinformation content under the convention above.
    kappa : float
        Confirmation-bias exponent. kappa = 1 reduces to standard Bayesian
        update; kappa > 1 produces confirmation bias.

    Returns
    -------
    float
        Updated belief contribution in [0, 1].
    """
    b_i = float(np.clip(b_i, 1e-12, 1.0 - 1e-12))
    m = float(np.clip(m, 1e-12, 1.0 - 1e-12))
    num = (b_i ** kappa) * m
    den = num + ((1.0 - b_i) ** kappa) * (1.0 - m)
    return num / max(den, 1e-12)


def social_reinforcement(
    beliefs: np.ndarray,
    neighbours_per_layer: Sequence[Sequence[int]],
    psi: Sequence[float],
) -> float:
    """
    Social reinforcement component (Equation 6):

        S_i(t) = sum_l  psi_l * (sum_{j in N_l_i} b_j(t))
                 / (sum_l  psi_l * |N_l_i(t)|)

    For totally isolated agents (no neighbours on any layer), the convention
    in the manuscript is to return b_i(t) itself; that case must be handled
    by the caller — here we return 0.0 if every layer is empty (callers in
    this codebase substitute b_i in that case).

    Parameters
    ----------
    beliefs : np.ndarray
        Belief vector of length N.
    neighbours_per_layer : sequence of sequences
        For each layer l, the list of neighbour indices of the focal agent i.
    psi : sequence of floats
        Layer-specific influence weights (must sum to 1).

    Returns
    -------
    float
        Social reinforcement contribution.
    """
    weighted_sum = 0.0
    weight_total = 0.0
    for l, neighbours in enumerate(neighbours_per_layer):
        if not neighbours:
            continue
        layer_sum = float(np.sum(beliefs[list(neighbours)]))
        weighted_sum += psi[l] * layer_sum
        weight_total += psi[l] * len(neighbours)
    if weight_total <= 0.0:
        return 0.0
    return weighted_sum / weight_total


@dataclass
class BeliefDynamics:
    """
    Belief-update engine for the MARIN co-evolving multiplex network.

    Parameters
    ----------
    mu : float
        Belief update rate (Eq. 4); (1 - mu) is the belief inertia.
    kappa : float
        Confirmation-bias parameter (Eq. 5).
    psi : sequence of floats
        Layer-specific influence weights, with sum 1 and psi_3 > psi_2 > psi_1
        per the manuscript (default: 0.2, 0.3, 0.5).
    """

    mu: float = 0.1
    kappa: float = 1.5
    psi: Sequence[float] = (0.2, 0.3, 0.5)

    def __post_init__(self):
        if abs(sum(self.psi) - 1.0) > 1e-6:
            raise ValueError(
                f"psi must sum to 1; got {self.psi} (sum={sum(self.psi):.4f})"
            )

    def update(
        self,
        beliefs: np.ndarray,
        message_signal: np.ndarray,
        neighbours_per_layer_per_node: List[List[List[int]]],
    ) -> np.ndarray:
        """
        Apply one timestep of belief updates (Equation 4) to every agent.

            b_i(t+1) = (1 - mu) * b_i(t)
                       + mu * (kappa * B_i(m, t) + (1 - kappa) * S_i(t))

        Parameters
        ----------
        beliefs : np.ndarray
            Belief vector of length N at time t.
        message_signal : np.ndarray
            Message-content signal m_i in [0, 1] for each agent.
        neighbours_per_layer_per_node : list of length N
            For each agent i, a list of length n_layers, each entry a list of
            neighbour indices on that layer.

        Returns
        -------
        np.ndarray
            New belief vector at time t+1, clipped to [0, 1].
        """
        n = len(beliefs)
        new_beliefs = np.empty_like(beliefs)
        for i in range(n):
            b_i = float(beliefs[i])
            m_i = float(message_signal[i])
            neighbours = neighbours_per_layer_per_node[i]

            B = bayesian_update(b_i, m_i, kappa=self.kappa)
            S = social_reinforcement(beliefs, neighbours, self.psi)

            # Convention from Section 3.2: isolated agents preserve belief
            if all(len(layer) == 0 for layer in neighbours):
                S = b_i

            new_b = (1.0 - self.mu) * b_i + self.mu * (
                self.kappa * B + (1.0 - self.kappa) * S
            )
            new_beliefs[i] = float(np.clip(new_b, 0.0, 1.0))
        return new_beliefs


if __name__ == "__main__":
    # Smoke test
    rng = np.random.default_rng(0)
    N = 50
    beliefs = rng.beta(2, 2, size=N)
    msg = rng.uniform(0, 1, size=N)
    neigh = [
        [list(rng.choice(N, size=rng.integers(0, 5), replace=False)) for _ in range(3)]
        for _ in range(N)
    ]
    dyn = BeliefDynamics()
    new_b = dyn.update(beliefs, msg, neigh)
    print(f"Belief update OK. mean(b) {beliefs.mean():.3f} -> {new_b.mean():.3f}")
