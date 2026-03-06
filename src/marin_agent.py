"""
MARIN Framework - Reinforcement Learning Agent
Double Deep Q-Network (DDQN) with GNN Encoder for Misinformation Intervention

Authors: Anjali Bhadre, Harshvardhan Ghongade
Paper: "Adaptive Intervention Strategies in Co-Evolving Multiplex Networks"
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from typing import Dict, List, Tuple, Optional
from collections import deque
import random


class GNNEncoder(nn.Module):
    """
    Graph Neural Network Encoder with Cross-Layer Attention (Section 3.3).
    
    Implements message-passing layers followed by attention-based
    cross-layer aggregation to produce node embeddings.
    
    h_i^(l) = ReLU(W^(l) · AGG({h_j^(l-1) : j ∈ N_i}) + U^(l) · h_i^(l-1))  (Eq. 7)
    z_i = Σ_l α_l · h_i^(l), where α_l = softmax(v^T · tanh(W_a · h_i^(l)))
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 128,
        n_layers: int = 3,
        n_network_layers: int = 3,
        attention_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.n_layers = n_layers
        self.n_network_layers = n_network_layers
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Message passing layers for each network layer
        self.message_layers = nn.ModuleList()
        self.update_layers = nn.ModuleList()
        
        for _ in range(n_layers):
            self.message_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.update_layers.append(nn.Linear(hidden_dim * 2, hidden_dim))
        
        # Cross-layer attention
        self.attention_W = nn.Linear(hidden_dim, attention_heads)
        self.attention_v = nn.Parameter(torch.randn(attention_heads))
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        node_features: torch.Tensor,
        adjacency_matrices: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through GNN encoder.
        
        Args:
            node_features: Node feature matrix [N, input_dim]
            adjacency_matrices: List of adjacency matrices for each layer
            
        Returns:
            node_embeddings: Node-level embeddings [N, output_dim]
            global_embedding: Global graph embedding [output_dim]
        """
        batch_size = node_features.size(0) if node_features.dim() == 3 else 1
        
        # Input projection
        h = self.input_proj(node_features)  # [N, hidden_dim]
        
        # Store embeddings from each network layer for attention
        layer_embeddings = []
        
        for net_layer, adj in enumerate(adjacency_matrices):
            h_layer = h.clone()
            
            # Message passing
            for l in range(self.n_layers):
                # Aggregate neighbor messages (mean pooling)
                if adj.dim() == 2:
                    # Normalize adjacency
                    deg = adj.sum(dim=1, keepdim=True).clamp(min=1)
                    adj_norm = adj / deg
                    messages = torch.matmul(adj_norm, h_layer)
                else:
                    messages = h_layer  # Fallback
                
                messages = self.message_layers[l](messages)
                
                # Update node representations
                h_combined = torch.cat([h_layer, messages], dim=-1)
                h_layer = F.relu(self.update_layers[l](h_combined))
                h_layer = self.dropout(h_layer)
            
            layer_embeddings.append(h_layer)
        
        # Cross-layer attention aggregation
        # α_l = softmax(v^T · tanh(W_a · h_i^(l)))
        stacked = torch.stack(layer_embeddings, dim=1)  # [N, n_network_layers, hidden_dim]
        
        attention_scores = torch.tanh(self.attention_W(stacked))  # [N, n_network_layers, heads]
        attention_scores = (attention_scores * self.attention_v).sum(dim=-1)  # [N, n_network_layers]
        attention_weights = F.softmax(attention_scores, dim=1)  # [N, n_network_layers]
        
        # Weighted combination
        node_embeddings = (stacked * attention_weights.unsqueeze(-1)).sum(dim=1)  # [N, hidden_dim]
        node_embeddings = self.output_proj(node_embeddings)
        
        # Global embedding (mean pooling)
        global_embedding = node_embeddings.mean(dim=0)  # [output_dim]
        
        return node_embeddings, global_embedding


class QNetwork(nn.Module):
    """
    Q-Network for node-wise intervention scoring (Equation 8).
    
    q_i = Q_φ(z_i, z_G; θ) ∈ ℝ^4
    
    Output dimensions correspond to intervention types:
    1. Truth injection
    2. Bridge-node activation
    3. Echo chamber disruption
    4. Cross-layer amplification
    """
    
    def __init__(
        self,
        node_dim: int = 128,
        global_dim: int = 128,
        hidden_dim: int = 256,
        n_intervention_types: int = 4
    ):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(node_dim + global_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_intervention_types)
        )
    
    def forward(
        self,
        node_embeddings: torch.Tensor,
        global_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Q-values for each node-intervention pair.
        
        Args:
            node_embeddings: [N, node_dim]
            global_embedding: [global_dim]
            
        Returns:
            q_values: [N, n_intervention_types]
        """
        N = node_embeddings.size(0)
        
        # Expand global embedding to match nodes
        global_expanded = global_embedding.unsqueeze(0).expand(N, -1)
        
        # Concatenate node and global embeddings
        combined = torch.cat([node_embeddings, global_expanded], dim=-1)
        
        # Compute Q-values
        q_values = self.network(combined)
        
        return q_values


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer for DDQN training.
    """
    
    def __init__(
        self,
        capacity: int = 100000,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        beta_frames: int = 100000
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_frames = beta_frames
        
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.frame = 0
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer."""
        max_priority = max(self.priorities) if self.priorities else 1.0
        
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(max_priority)
    
    def sample(self, batch_size: int) -> Tuple:
        """Sample batch with prioritized sampling."""
        self.frame += 1
        
        # Calculate beta for importance sampling
        beta = min(
            self.beta_end,
            self.beta_start + (self.beta_end - self.beta_start) * self.frame / self.beta_frames
        )
        
        # Calculate sampling probabilities
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        # Get samples
        samples = [self.buffer[idx] for idx in indices]
        
        states, actions, rewards, next_states, dones = zip(*samples)
        
        return (
            states, actions, rewards, next_states, dones,
            indices, torch.FloatTensor(weights)
        )
    
    def update_priorities(self, indices: List[int], td_errors: np.ndarray):
        """Update priorities based on TD errors."""
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = abs(td_error) + 1e-6
    
    def __len__(self):
        return len(self.buffer)


class MARINAgent:
    """
    MARIN Reinforcement Learning Agent.
    
    Combines GNN encoder with DDQN for adaptive misinformation intervention.
    Uses node-wise action decomposition to handle combinatorial action space.
    """
    
    def __init__(
        self,
        node_feature_dim: int = 10,
        embedding_dim: int = 128,
        n_intervention_types: int = 4,
        budget: int = 50,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        tau: int = 1000,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 100000,
        batch_size: int = 64,
        device: str = "cpu"
    ):
        self.device = torch.device(device)
        self.n_intervention_types = n_intervention_types
        self.budget = budget
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        
        # Exploration parameters
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Networks
        self.encoder = GNNEncoder(
            input_dim=node_feature_dim,
            output_dim=embedding_dim
        ).to(self.device)
        
        self.q_network = QNetwork(
            node_dim=embedding_dim,
            global_dim=embedding_dim,
            n_intervention_types=n_intervention_types
        ).to(self.device)
        
        self.target_q_network = QNetwork(
            node_dim=embedding_dim,
            global_dim=embedding_dim,
            n_intervention_types=n_intervention_types
        ).to(self.device)
        
        # Copy weights to target network
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = Adam(
            list(self.encoder.parameters()) + list(self.q_network.parameters()),
            lr=learning_rate
        )
        
        # Replay buffer
        self.replay_buffer = PrioritizedReplayBuffer(capacity=buffer_size)
        
        # Training step counter
        self.train_step = 0
    
    def select_action(
        self,
        node_features: np.ndarray,
        adjacency_matrices: List[np.ndarray],
        training: bool = True
    ) -> List[Tuple[int, int]]:
        """
        Select intervention actions using epsilon-greedy policy.
        
        Returns list of (node_index, intervention_type) pairs.
        """
        # Convert to tensors
        node_features_t = torch.FloatTensor(node_features).to(self.device)
        adj_matrices_t = [torch.FloatTensor(adj).to(self.device) for adj in adjacency_matrices]
        
        # Get Q-values
        with torch.no_grad():
            node_emb, global_emb = self.encoder(node_features_t, adj_matrices_t)
            q_values = self.q_network(node_emb, global_emb)  # [N, n_types]
        
        # Epsilon-greedy action selection
        if training and random.random() < self.epsilon:
            # Random action
            n_nodes = node_features.shape[0]
            indices = np.random.choice(n_nodes * self.n_intervention_types, 
                                       size=min(self.budget, n_nodes),
                                       replace=False)
            actions = [(idx // self.n_intervention_types, idx % self.n_intervention_types)
                      for idx in indices]
        else:
            # Greedy action: select top-k (node, type) pairs by Q-value
            q_flat = q_values.view(-1).cpu().numpy()
            top_k_indices = np.argsort(q_flat)[-self.budget:][::-1]
            
            actions = [(idx // self.n_intervention_types, idx % self.n_intervention_types)
                      for idx in top_k_indices]
        
        return actions
    
    def train_step_update(
        self,
        state: Dict,
        action: List[Tuple[int, int]],
        reward: float,
        next_state: Dict,
        done: bool
    ):
        """Perform one training step."""
        # Add to replay buffer
        self.replay_buffer.push(state, action, reward, next_state, done)
        
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones, indices, weights = batch
        
        # Compute loss and update
        loss = self._compute_loss(states, actions, rewards, next_states, dones, weights)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + list(self.q_network.parameters()),
            max_norm=1.0
        )
        self.optimizer.step()
        
        # Update target network (Equation 10)
        self.train_step += 1
        if self.train_step % self.tau == 0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def _compute_loss(
        self,
        states: List[Dict],
        actions: List[List[Tuple[int, int]]],
        rewards: List[float],
        next_states: List[Dict],
        dones: List[bool],
        weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute DDQN loss (Equation 10).
        
        Q(s,a;θ) ← R + γ · Q(s', argmax_a' Q(s',a';θ); θ⁻)
        """
        total_loss = 0.0
        
        for i in range(len(states)):
            state = states[i]
            action = actions[i]
            reward = rewards[i]
            next_state = next_states[i]
            done = dones[i]
            
            # Get current Q-values
            node_feat = torch.FloatTensor(state['node_features']).to(self.device)
            adj_mats = [torch.FloatTensor(adj).to(self.device) for adj in state['adjacency']]
            
            node_emb, global_emb = self.encoder(node_feat, adj_mats)
            q_values = self.q_network(node_emb, global_emb)
            
            # Get Q-values for selected actions
            q_selected = sum(q_values[node, interv_type] for node, interv_type in action)
            
            # Get target Q-values
            with torch.no_grad():
                next_node_feat = torch.FloatTensor(next_state['node_features']).to(self.device)
                next_adj_mats = [torch.FloatTensor(adj).to(self.device) for adj in next_state['adjacency']]
                
                next_node_emb, next_global_emb = self.encoder(next_node_feat, next_adj_mats)
                
                # DDQN: use online network to select action, target network to evaluate
                next_q_online = self.q_network(next_node_emb, next_global_emb)
                next_q_target = self.target_q_network(next_node_emb, next_global_emb)
                
                # Get best action indices from online network
                best_indices = next_q_online.view(-1).topk(self.budget).indices
                
                # Evaluate with target network
                next_q_value = sum(next_q_target.view(-1)[idx] for idx in best_indices)
                
                target = reward + (1 - done) * self.gamma * next_q_value
            
            # TD error
            td_error = (q_selected - target) ** 2
            total_loss += weights[i] * td_error
        
        return total_loss / len(states)
    
    def save(self, path: str):
        """Save model weights."""
        torch.save({
            'encoder': self.encoder.state_dict(),
            'q_network': self.q_network.state_dict(),
            'target_q_network': self.target_q_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'train_step': self.train_step
        }, path)
    
    def load(self, path: str):
        """Load model weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_q_network.load_state_dict(checkpoint['target_q_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.train_step = checkpoint['train_step']


if __name__ == "__main__":
    # Example usage
    agent = MARINAgent(
        node_feature_dim=10,
        embedding_dim=128,
        n_intervention_types=4,
        budget=50
    )
    
    # Dummy state
    n_nodes = 100
    node_features = np.random.randn(n_nodes, 10)
    adjacency_matrices = [np.random.randint(0, 2, (n_nodes, n_nodes)) for _ in range(3)]
    
    # Select action
    actions = agent.select_action(node_features, adjacency_matrices)
    print(f"Selected {len(actions)} interventions: {actions[:5]}...")
