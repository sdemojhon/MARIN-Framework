"""
MARIN Framework - GNN Encoder convenience module.

The full GNN encoder implementation (message passing + cross-layer attention)
lives in `marin_agent.py` as the `GNNEncoder` class.  This module is a thin
factory that exposes `build_encoder()` so that user code can follow the
import paths advertised in the README without depending on agent internals.

References:
    Bhadre & Ghongade (2026), Equation 7 and Section 3.3 (State Space).
"""
from __future__ import annotations

from typing import Optional

from .marin_agent import GNNEncoder

__all__ = ["GNNEncoder", "build_encoder"]


def build_encoder(
    node_feature_dim: int,
    hidden_dim: int = 128,
    output_dim: int = 128,
    n_message_passing_layers: int = 3,
    n_network_layers: int = 3,
    attention_heads: int = 4,
    dropout: float = 0.1,
    device: Optional[str] = None,
) -> GNNEncoder:
    """
    Construct a GNN encoder with cross-layer attention aggregation.

    Parameters
    ----------
    node_feature_dim : int
        Dimensionality of per-node input features.
    hidden_dim : int
        Hidden width of the message-passing layers.
    output_dim : int
        Final per-node embedding width (128 in the manuscript).
    n_message_passing_layers : int
        Number of message-passing iterations L (default 3, per Table 2).
    n_network_layers : int
        Number of multiplex network layers (default 3).
    attention_heads : int
        Number of attention heads for cross-layer aggregation.
    dropout : float
        Dropout rate inside message-passing blocks.
    device : str, optional
        "cpu" or "cuda"; if provided, the model is moved to that device.

    Returns
    -------
    GNNEncoder
        A ready-to-use encoder. Use as
            node_emb, global_emb = encoder(node_features, adjacency_matrices)
        where node_features is [N, node_feature_dim] and adjacency_matrices is
        a list of [N, N] tensors, one per multiplex layer.
    """
    encoder = GNNEncoder(
        input_dim=node_feature_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        n_layers=n_message_passing_layers,
        n_network_layers=n_network_layers,
        attention_heads=attention_heads,
        dropout=dropout,
    )
    if device is not None:
        encoder = encoder.to(device)
    return encoder


if __name__ == "__main__":
    import torch

    enc = build_encoder(node_feature_dim=10)
    x = torch.randn(50, 10)
    adj = [torch.randint(0, 2, (50, 50)).float() for _ in range(3)]
    z_i, z_G = enc(x, adj)
    print(f"GNN encoder OK. node embeddings {tuple(z_i.shape)}, global {tuple(z_G.shape)}")
