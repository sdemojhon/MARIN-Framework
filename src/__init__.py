"""
MARIN Framework - Multiplex Adaptive Reinforcement Intervention Network

A deep reinforcement learning framework for real-time misinformation containment
in co-evolving multiplex social networks.

Authors: Anjali Bhadre, Harshvardhan Ghongade
"""

from .marin_network import MultiplexNetwork, MultiplexConfig, LayerConfig
from .marin_agent import MARINAgent, GNNEncoder, QNetwork

__version__ = "1.0.0"
__author__ = "Anjali Bhadre, Harshvardhan Ghongade"

__all__ = [
    "MultiplexNetwork",
    "MultiplexConfig", 
    "LayerConfig",
    "MARINAgent",
    "GNNEncoder",
    "QNetwork"
]
