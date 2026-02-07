"""Graph Neural Network for collaborative swarm perception.

Implements message-passing GNN for multi-agent communication:
- Node updates: aggregate neighbor information
- Edge updates: compute pairwise interactions
- Global readout: optional swarm-level features
"""

from __future__ import annotations

from typing import Callable, Sequence

import jax
import jax.numpy as jnp
from jax import Array
import flax.linen as nn

from swarm.perception.graph_builder import GraphData


class MLPBlock(nn.Module):
    """Simple MLP with optional layer norm."""
    
    features: Sequence[int]
    activation: Callable = nn.relu
    use_layer_norm: bool = True
    
    @nn.compact
    def __call__(self, x: Array) -> Array:
        for feat in self.features[:-1]:
            x = nn.Dense(feat)(x)
            if self.use_layer_norm:
                x = nn.LayerNorm()(x)
            x = self.activation(x)
        x = nn.Dense(self.features[-1])(x)
        return x


class MessagePassingLayer(nn.Module):
    """Single message-passing layer.
    
    Updates node features by aggregating messages from neighbors:
    1. Compute edge messages from sender features + edge features
    2. Aggregate messages at receiver nodes (mean pooling)
    3. Update node features using aggregated messages
    
    Args:
        hidden_dim: Dimension of hidden layers
        output_dim: Dimension of output node features
    """
    
    hidden_dim: int = 128
    output_dim: int = 128
    
    @nn.compact
    def __call__(self, graph: GraphData) -> GraphData:
        nodes = graph.nodes
        edges = graph.edges
        senders = graph.senders
        receivers = graph.receivers
        
        num_nodes = nodes.shape[0]
        num_edges = edges.shape[0]
        
        # Edge message network: f(sender_features, edge_features)
        sender_features = nodes[senders]  # (E, node_dim)
        edge_input = jnp.concatenate([sender_features, edges], axis=-1)
        messages = MLPBlock([self.hidden_dim, self.hidden_dim])(edge_input)  # (E, hidden)
        
        # Aggregate messages at receivers (mean pooling)
        aggregated = jnp.zeros((num_nodes, self.hidden_dim))
        aggregated = aggregated.at[receivers].add(messages)
        
        # Count incoming edges per node for mean
        counts = jnp.zeros(num_nodes)
        counts = counts.at[receivers].add(1.0)
        counts = jnp.maximum(counts, 1.0)  # Avoid division by zero
        aggregated = aggregated / counts[:, None]
        
        # Node update network: f(node_features, aggregated_messages)
        node_input = jnp.concatenate([nodes, aggregated], axis=-1)
        new_nodes = MLPBlock([self.hidden_dim, self.output_dim])(node_input)
        
        # Residual connection if dimensions match
        if nodes.shape[-1] == self.output_dim:
            new_nodes = new_nodes + nodes
        
        return GraphData(
            nodes=new_nodes,
            edges=edges,
            senders=senders,
            receivers=receivers,
            n_node=graph.n_node,
            n_edge=graph.n_edge,
        )


class SwarmGNN(nn.Module):
    """Multi-layer GNN for swarm perception.
    
    Stacks multiple message-passing layers to enable multi-hop
    information propagation across the swarm.
    
    Args:
        hidden_dim: Hidden layer dimension
        output_dim: Final output dimension per agent
        num_layers: Number of message-passing layers
        use_edge_updates: Whether to update edge features (slower but more expressive)
    """
    
    hidden_dim: int = 128
    output_dim: int = 64
    num_layers: int = 3
    use_edge_updates: bool = False
    
    @nn.compact
    def __call__(self, graph: GraphData) -> Array:
        """Process graph and return per-agent features.
        
        Args:
            graph: Input graph with node/edge features
            
        Returns:
            Per-agent feature vectors (num_agents, output_dim)
        """
        # Initial node embedding
        x = graph.nodes
        x = MLPBlock([self.hidden_dim, self.hidden_dim])(x)
        
        # Update graph with embedded nodes
        current_graph = GraphData(
            nodes=x,
            edges=graph.edges,
            senders=graph.senders,
            receivers=graph.receivers,
            n_node=graph.n_node,
            n_edge=graph.n_edge,
        )
        
        # Apply message-passing layers
        for i in range(self.num_layers):
            current_graph = MessagePassingLayer(
                hidden_dim=self.hidden_dim,
                output_dim=self.hidden_dim,
                name=f"mp_layer_{i}",
            )(current_graph)
        
        # Final projection to output dimension
        output = MLPBlock([self.hidden_dim, self.output_dim])(current_graph.nodes)
        
        return output


class AttentionMessagePassing(nn.Module):
    """Graph Attention-style message passing.
    
    Uses attention weights to aggregate neighbor messages,
    allowing the network to focus on important neighbors.
    
    Args:
        hidden_dim: Hidden dimension
        num_heads: Number of attention heads
        output_dim: Output dimension
    """
    
    hidden_dim: int = 128
    num_heads: int = 4
    output_dim: int = 128
    
    @nn.compact
    def __call__(self, graph: GraphData) -> GraphData:
        nodes = graph.nodes
        edges = graph.edges
        senders = graph.senders
        receivers = graph.receivers
        
        num_nodes = nodes.shape[0]
        head_dim = self.hidden_dim // self.num_heads
        
        # Query, Key, Value projections
        queries = nn.Dense(self.hidden_dim)(nodes)  # (N, hidden)
        keys = nn.Dense(self.hidden_dim)(nodes)      # (N, hidden)
        values = nn.Dense(self.hidden_dim)(nodes)    # (N, hidden)
        
        # Reshape for multi-head attention
        queries = queries.reshape(num_nodes, self.num_heads, head_dim)
        keys = keys.reshape(num_nodes, self.num_heads, head_dim)
        values = values.reshape(num_nodes, self.num_heads, head_dim)
        
        # Compute attention scores for edges
        q_senders = queries[senders]      # (E, heads, head_dim)
        k_receivers = keys[receivers]      # (E, heads, head_dim)
        
        # Scaled dot-product attention
        scores = jnp.sum(q_senders * k_receivers, axis=-1) / jnp.sqrt(head_dim)  # (E, heads)
        
        # Softmax over edges per receiver (approximate with direct normalization)
        # Note: For exact softmax, would need sparse ops
        attn_weights = nn.softmax(scores, axis=-1)  # (E, heads)
        
        # Aggregate values weighted by attention
        v_senders = values[senders]  # (E, heads, head_dim)
        weighted_values = v_senders * attn_weights[:, :, None]  # (E, heads, head_dim)
        weighted_values = weighted_values.reshape(-1, self.hidden_dim)  # (E, hidden)
        
        # Sum aggregation at receivers
        aggregated = jnp.zeros((num_nodes, self.hidden_dim))
        aggregated = aggregated.at[receivers].add(weighted_values)
        
        # Output projection with residual
        output = nn.Dense(self.output_dim)(aggregated)
        if nodes.shape[-1] == self.output_dim:
            output = output + nodes
        
        return GraphData(
            nodes=output,
            edges=edges,
            senders=senders,
            receivers=receivers,
            n_node=graph.n_node,
            n_edge=graph.n_edge,
        )


def create_gnn(
    hidden_dim: int = 128,
    output_dim: int = 64,
    num_layers: int = 3,
) -> SwarmGNN:
    """Factory function to create SwarmGNN.
    
    Args:
        hidden_dim: Hidden layer dimension
        output_dim: Output feature dimension
        num_layers: Number of message-passing layers
        
    Returns:
        SwarmGNN module
    """
    return SwarmGNN(
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
    )


def init_gnn(
    gnn: SwarmGNN,
    key: Array,
    num_agents: int = 10,
    node_dim: int = 6,
    edge_dim: int = 7,
) -> dict:
    """Initialize GNN parameters.
    
    Args:
        gnn: SwarmGNN module
        key: JAX PRNG key
        num_agents: Number of agents for dummy input
        node_dim: Node feature dimension
        edge_dim: Edge feature dimension
        
    Returns:
        Initialized parameters
    """
    # Create dummy graph
    dummy_graph = GraphData(
        nodes=jnp.zeros((num_agents, node_dim)),
        edges=jnp.zeros((num_agents * 5, edge_dim)),  # ~5 edges per node
        senders=jnp.zeros(num_agents * 5, dtype=jnp.int32),
        receivers=jnp.zeros(num_agents * 5, dtype=jnp.int32),
        n_node=jnp.array([num_agents]),
        n_edge=jnp.array([num_agents * 5]),
    )
    
    return gnn.init(key, dummy_graph)
