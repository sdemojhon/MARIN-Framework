"""
MARIN Framework - Multiplex Network Model
Co-Evolving Multiplex Network with Link Formation and Dissolution Dynamics

Authors: Anjali Bhadre, Harshvardhan Ghongade
Paper: "Adaptive Intervention Strategies in Co-Evolving Multiplex Networks"
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class LayerConfig:
    """Configuration for a single network layer."""
    name: str
    gamma: float = 0.8      # Link formation scaling
    lambda_: float = 1.0    # Homophily coefficient
    epsilon: float = 0.05   # Baseline connection rate
    psi: float = 0.2        # Influence weight
    delta: float = 1.0      # Disagreement sensitivity
    phi: float = 0.5        # Inactivity sensitivity
    theta: float = 0.5      # Dissolution threshold


@dataclass
class MultiplexConfig:
    """Configuration for the multiplex network."""
    n_nodes: int = 1000
    n_layers: int = 3
    omega: float = 0.1      # Co-evolution rate
    eta: float = 0.9        # Temporal discount factor
    coupling_matrix: Dict[Tuple[int, int], float] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.coupling_matrix:
            # Default coupling matrix from paper
            self.coupling_matrix = {
                (0, 1): 0.30, (0, 2): 0.10,
                (1, 0): 0.20, (1, 2): 0.25,
                (2, 0): 0.15, (2, 1): 0.35
            }


class MultiplexNetwork:
    """
    Co-Evolving Multiplex Network Model (Definition 1 from paper).
    
    Implements a three-layer multiplex network where:
    - Layer 1: Information Sharing (public broadcasts)
    - Layer 2: Social Reinforcement (comments, replies)
    - Layer 3: Private Communication (DMs, closed groups)
    
    The network topology evolves based on agent belief states through
    link formation (Eq. 1) and dissolution (Eq. 2) functions.
    """
    
    def __init__(
        self,
        config: Optional[MultiplexConfig] = None,
        layer_configs: Optional[List[LayerConfig]] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize the multiplex network.
        
        Args:
            config: Multiplex network configuration
            layer_configs: List of layer-specific configurations
            seed: Random seed for reproducibility
        """
        self.config = config or MultiplexConfig()
        self.n_nodes = self.config.n_nodes
        self.n_layers = self.config.n_layers
        
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize layer configurations
        if layer_configs is None:
            self.layer_configs = [
                LayerConfig("information_sharing", gamma=0.8, lambda_=1.0, epsilon=0.05, psi=0.2),
                LayerConfig("social_reinforcement", gamma=0.7, lambda_=1.5, epsilon=0.03, psi=0.3),
                LayerConfig("private_communication", gamma=0.6, lambda_=2.0, epsilon=0.02, psi=0.5),
            ]
        else:
            self.layer_configs = layer_configs
        
        # Initialize network layers
        self.layers: List[nx.Graph] = []
        self._initialize_topology()
        
        # Initialize agent beliefs - Beta(2,2) distribution (Definition 5)
        self.beliefs = np.random.beta(2, 2, size=self.n_nodes)
        
        # Disagreement history for link dissolution
        self.disagreement_history: Dict[Tuple[int, int, int], List[float]] = {}
        
        # Last interaction time
        self.last_interaction: Dict[Tuple[int, int, int], int] = {}
        
        # Current timestep
        self.t = 0
    
    def _initialize_topology(self, m: int = 5):
        """
        Initialize network topology using Barabási-Albert model.
        
        Args:
            m: Number of edges to attach from new node
        """
        for l in range(self.n_layers):
            G = nx.barabasi_albert_graph(self.n_nodes, m)
            self.layers.append(G)
    
    def link_formation_probability(
        self,
        i: int,
        j: int,
        layer: int
    ) -> float:
        """
        Calculate link formation probability (Equation 1).
        
        f_l(i,j,t) = γ_l · exp(-λ_l · |b_i(t) - b_j(t)|) + ε_l
        
        Args:
            i: First agent index
            j: Second agent index
            layer: Layer index
            
        Returns:
            Link formation probability
        """
        cfg = self.layer_configs[layer]
        belief_distance = abs(self.beliefs[i] - self.beliefs[j])
        
        prob = cfg.gamma * np.exp(-cfg.lambda_ * belief_distance) + cfg.epsilon
        
        # Apply cross-layer coupling (Equation 3)
        prob = self._apply_cross_layer_coupling(prob, i, j, layer)
        
        return min(prob, 1.0)  # Ensure valid probability
    
    def _apply_cross_layer_coupling(
        self,
        base_prob: float,
        i: int,
        j: int,
        layer: int
    ) -> float:
        """
        Apply cross-layer coupling (Equation 3).
        
        f̃_l(i,j,t) = f_l(i,j,t) + Σ_{l'≠l} C_{ll'} · A_{l'}^{ij}(t) · f_{l'}(i,j,t)
        """
        coupled_prob = base_prob
        
        for l_prime in range(self.n_layers):
            if l_prime == layer:
                continue
            
            # Check if edge exists on other layer
            if self.layers[l_prime].has_edge(i, j):
                coupling = self.config.coupling_matrix.get((layer, l_prime), 0.0)
                cfg_prime = self.layer_configs[l_prime]
                belief_distance = abs(self.beliefs[i] - self.beliefs[j])
                f_prime = cfg_prime.gamma * np.exp(-cfg_prime.lambda_ * belief_distance) + cfg_prime.epsilon
                coupled_prob += coupling * f_prime
        
        return coupled_prob
    
    def link_dissolution_probability(
        self,
        i: int,
        j: int,
        layer: int
    ) -> float:
        """
        Calculate link dissolution probability (Equation 2).
        
        g_l(i,j,t) = σ(δ_l · D_{ij}(t) + φ_l · ΔT_{ij}(t) - θ_l)
        
        where σ is sigmoid, D_{ij} is cumulative disagreement, ΔT is time since last interaction.
        """
        cfg = self.layer_configs[layer]
        
        # Get cumulative disagreement exposure
        D_ij = self._get_cumulative_disagreement(i, j, layer)
        
        # Get time since last interaction
        delta_T = self._get_time_since_interaction(i, j, layer)
        
        # Sigmoid function
        x = cfg.delta * D_ij + cfg.phi * delta_T - cfg.theta
        prob = 1 / (1 + np.exp(-x))
        
        return prob
    
    def _get_cumulative_disagreement(self, i: int, j: int, layer: int) -> float:
        """Calculate cumulative disagreement with temporal discounting."""
        key = (min(i, j), max(i, j), layer)
        history = self.disagreement_history.get(key, [])
        
        if not history:
            return 0.0
        
        # Apply temporal discounting
        D = 0.0
        for t_prime, disagreement in enumerate(history):
            D += (self.config.eta ** (self.t - t_prime)) * disagreement
        
        return D
    
    def _get_time_since_interaction(self, i: int, j: int, layer: int) -> int:
        """Get time steps since last interaction."""
        key = (min(i, j), max(i, j), layer)
        last_t = self.last_interaction.get(key, 0)
        return self.t - last_t
    
    def update_topology(self):
        """
        Update network topology based on co-evolution dynamics.
        Called at each timestep with probability omega.
        """
        for layer in range(self.n_layers):
            edges_to_remove = []
            edges_to_add = []
            
            # Check existing edges for dissolution
            for i, j in list(self.layers[layer].edges()):
                if np.random.random() < self.config.omega:
                    prob = self.link_dissolution_probability(i, j, layer)
                    if np.random.random() < prob:
                        edges_to_remove.append((i, j))
            
            # Check non-edges for formation
            for i in range(self.n_nodes):
                for j in range(i + 1, self.n_nodes):
                    if not self.layers[layer].has_edge(i, j):
                        if np.random.random() < self.config.omega:
                            prob = self.link_formation_probability(i, j, layer)
                            if np.random.random() < prob:
                                edges_to_add.append((i, j))
            
            # Apply changes
            self.layers[layer].remove_edges_from(edges_to_remove)
            self.layers[layer].add_edges_from(edges_to_add)
    
    def get_neighbors(self, node: int, layer: int) -> List[int]:
        """Get neighbors of a node on a specific layer."""
        return list(self.layers[layer].neighbors(node))
    
    def get_all_neighbors(self, node: int) -> Dict[int, List[int]]:
        """Get neighbors of a node across all layers."""
        return {l: self.get_neighbors(node, l) for l in range(self.n_layers)}
    
    def get_adjacency_matrices(self) -> List[np.ndarray]:
        """Get adjacency matrices for all layers."""
        return [nx.to_numpy_array(G) for G in self.layers]
    
    def get_network_statistics(self) -> Dict[str, float]:
        """Calculate network-level statistics."""
        stats = {}
        
        for l, G in enumerate(self.layers):
            prefix = f"layer_{l}_"
            stats[prefix + "density"] = nx.density(G)
            stats[prefix + "clustering"] = nx.average_clustering(G)
            stats[prefix + "modularity"] = self._calculate_modularity(G)
        
        # Cross-layer statistics
        stats["belief_mean"] = np.mean(self.beliefs)
        stats["belief_std"] = np.std(self.beliefs)
        stats["polarization"] = np.mean(np.abs(2 * self.beliefs - 1))
        
        return stats
    
    def _calculate_modularity(self, G: nx.Graph) -> float:
        """Calculate network modularity using Louvain communities."""
        try:
            from networkx.algorithms.community import louvain_communities
            communities = louvain_communities(G, seed=42)
            return nx.community.modularity(G, communities)
        except:
            return 0.0
    
    def step(self):
        """Advance simulation by one timestep."""
        # Update disagreement history
        for layer in range(self.n_layers):
            for i, j in self.layers[layer].edges():
                key = (min(i, j), max(i, j), layer)
                disagreement = abs(self.beliefs[i] - self.beliefs[j])
                
                if key not in self.disagreement_history:
                    self.disagreement_history[key] = []
                self.disagreement_history[key].append(disagreement)
                
                # Update last interaction
                self.last_interaction[key] = self.t
        
        # Update topology
        self.update_topology()
        
        # Increment time
        self.t += 1
    
    def reset(self, seed: Optional[int] = None):
        """Reset network to initial state."""
        if seed is not None:
            np.random.seed(seed)
        
        self.layers = []
        self._initialize_topology()
        self.beliefs = np.random.beta(2, 2, size=self.n_nodes)
        self.disagreement_history = {}
        self.last_interaction = {}
        self.t = 0


if __name__ == "__main__":
    # Example usage
    config = MultiplexConfig(n_nodes=100, omega=0.1)
    network = MultiplexNetwork(config, seed=42)
    
    print(f"Initialized multiplex network with {network.n_nodes} nodes and {network.n_layers} layers")
    print(f"Initial statistics: {network.get_network_statistics()}")
    
    # Run a few steps
    for _ in range(10):
        network.step()
    
    print(f"After 10 steps: {network.get_network_statistics()}")
