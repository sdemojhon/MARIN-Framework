"""
MARIN Framework
Multiplex Adaptive Reinforcement Intervention Network

Reference:
    Bhadre, A. and Ghongade, H.P. (2026).
    "Adaptive Intervention Strategies in Co-Evolving Multiplex Networks:
    A Reinforcement Learning Approach to Real-Time Misinformation Containment."
    Northeast Journal of Complex Systems (NEJCS).

Foundational paper:
    Ghongade, H.P., Bhadre, A.A., Agarwal, S., Pawar, H.U., Rane, H.S. (2026).
    "Emergent Dynamics in Multiplex Social Networks: Agent-Based Modeling of
    Information Diffusion for Misinformation Control."
    NEJCS 8(1), Article 12. DOI: 10.63562/2577-8439.1152
"""

from .marin_network import (
    MultiplexNetwork,
    MultiplexConfig,
    LayerConfig,
)
from .marin_agent import (
    MARINAgent,
    GNNEncoder,
    QNetwork,
    PrioritizedReplayBuffer,
)
from .belief_dynamics import (
    BeliefDynamics,
    bayesian_update,
    social_reinforcement,
)
from .interventions import (
    InterventionType,
    apply_intervention,
    INTERVENTION_NAMES,
)
from .gnn_encoder import build_encoder
from .simulation import (
    run_simulation,
    run_monte_carlo,
    SimulationResult,
)

__version__ = "1.0.0"
__all__ = [
    "MultiplexNetwork", "MultiplexConfig", "LayerConfig",
    "MARINAgent", "GNNEncoder", "QNetwork", "PrioritizedReplayBuffer",
    "BeliefDynamics", "bayesian_update", "social_reinforcement",
    "InterventionType", "apply_intervention", "INTERVENTION_NAMES",
    "build_encoder",
    "run_simulation", "run_monte_carlo", "SimulationResult",
]
