"""Graph builder for swarm communication topology.

Constructs dynamic graphs based on agent proximity with support for:
- Radius-based connectivity
- k-nearest neighbors
- Curriculum learning (progressive packet loss/delays)
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array


class GraphData(NamedTuple):
    """Graph representation for GNN processing.
    
    Attributes:
        nodes: Node features (num_agents, node_dim)
        edges: Edge features (num_edges, edge_dim)
        senders: Source node indices (num_edges,)
        receivers: Target node indices (num_edges,)
        n_node: Number of nodes per graph (for batching)
        n_edge: Number of edges per graph (for batching)
    """
    nodes: Array
    edges: Array
    senders: Array
    receivers: Array
    n_node: Array
    n_edge: Array


def build_radius_graph(
    positions: Array,
    velocities: Array,
    radius: float = 50.0,
    max_edges_per_node: int = 10,
) -> GraphData:
    """Build graph with edges between agents within radius.
    
    Args:
        positions: Agent positions (N, 3)
        velocities: Agent velocities (N, 3)
        radius: Communication radius in meters
        max_edges_per_node: Maximum outgoing edges per agent
        
    Returns:
        GraphData with connectivity and features
    """
    num_agents = positions.shape[0]
    
    # Compute pairwise distances
    diff = positions[:, None, :] - positions[None, :, :]  # (N, N, 3)
    dists = jnp.linalg.norm(diff, axis=-1)  # (N, N)
    
    # Create adjacency (exclude self-loops)
    adjacency = (dists < radius) & (dists > 0)
    
    # Limit edges per node
    # Sort by distance and keep closest max_edges_per_node
    dists_masked = jnp.where(adjacency, dists, jnp.inf)
    sorted_indices = jnp.argsort(dists_masked, axis=-1)
    
    # Create mask for top-k neighbors
    k_mask = jnp.arange(num_agents) < max_edges_per_node
    valid_edges = jnp.zeros((num_agents, num_agents), dtype=bool)
    
    for i in range(num_agents):
        neighbor_indices = sorted_indices[i, :max_edges_per_node]
        neighbor_dists = dists_masked[i, neighbor_indices]
        valid = neighbor_dists < jnp.inf
        valid_edges = valid_edges.at[i, neighbor_indices].set(valid)
    
    # Extract edge lists
    senders, receivers = jnp.where(valid_edges)
    
    # Node features: position + velocity
    nodes = jnp.concatenate([positions, velocities], axis=-1)  # (N, 6)
    
    # Edge features: relative position, relative velocity, distance
    rel_pos = positions[receivers] - positions[senders]  # (E, 3)
    rel_vel = velocities[receivers] - velocities[senders]  # (E, 3)
    edge_dist = jnp.linalg.norm(rel_pos, axis=-1, keepdims=True)  # (E, 1)
    edges = jnp.concatenate([rel_pos, rel_vel, edge_dist], axis=-1)  # (E, 7)
    
    return GraphData(
        nodes=nodes,
        edges=edges,
        senders=senders,
        receivers=receivers,
        n_node=jnp.array([num_agents]),
        n_edge=jnp.array([len(senders)]),
    )


def build_knn_graph(
    positions: Array,
    velocities: Array,
    k: int = 5,
) -> GraphData:
    """Build k-nearest neighbors graph.
    
    Args:
        positions: Agent positions (N, 3)
        velocities: Agent velocities (N, 3)
        k: Number of neighbors per agent
        
    Returns:
        GraphData with k-NN connectivity
    """
    num_agents = positions.shape[0]
    k = min(k, num_agents - 1)
    
    # Compute pairwise distances
    diff = positions[:, None, :] - positions[None, :, :]
    dists = jnp.linalg.norm(diff, axis=-1)
    dists = dists + jnp.eye(num_agents) * 1e10  # Exclude self
    
    # Get k nearest neighbors for each agent
    neighbor_indices = jnp.argsort(dists, axis=-1)[:, :k]  # (N, k)
    
    # Create edge lists
    senders = jnp.repeat(jnp.arange(num_agents), k)
    receivers = neighbor_indices.ravel()
    
    # Node features
    nodes = jnp.concatenate([positions, velocities], axis=-1)
    
    # Edge features
    rel_pos = positions[receivers] - positions[senders]
    rel_vel = velocities[receivers] - velocities[senders]
    edge_dist = jnp.linalg.norm(rel_pos, axis=-1, keepdims=True)
    edges = jnp.concatenate([rel_pos, rel_vel, edge_dist], axis=-1)
    
    return GraphData(
        nodes=nodes,
        edges=edges,
        senders=senders,
        receivers=receivers,
        n_node=jnp.array([num_agents]),
        n_edge=jnp.array([num_agents * k]),
    )


def apply_packet_dropout(
    graph: GraphData,
    key: Array,
    dropout_rate: float = 0.0,
) -> GraphData:
    """Apply random packet loss to edges (curriculum learning).
    
    Args:
        graph: Input graph
        key: JAX PRNG key
        dropout_rate: Probability of dropping each edge (0-1)
        
    Returns:
        Graph with some edges removed
    """
    if dropout_rate <= 0:
        return graph
    
    num_edges = graph.edges.shape[0]
    
    # Generate dropout mask
    keep_mask = jax.random.uniform(key, (num_edges,)) > dropout_rate
    
    # Filter edges
    kept_indices = jnp.where(keep_mask, size=num_edges, fill_value=-1)[0]
    valid_count = keep_mask.sum()
    
    # Gather kept edges (pad with zeros for invalid)
    def safe_gather(arr, indices):
        return jnp.where(
            indices[:, None] >= 0,
            arr[jnp.clip(indices, 0, len(arr) - 1)],
            0.0
        )
    
    new_edges = safe_gather(graph.edges, kept_indices)
    new_senders = jnp.where(kept_indices >= 0, graph.senders[jnp.clip(kept_indices, 0, num_edges - 1)], 0)
    new_receivers = jnp.where(kept_indices >= 0, graph.receivers[jnp.clip(kept_indices, 0, num_edges - 1)], 0)
    
    return GraphData(
        nodes=graph.nodes,
        edges=new_edges[:valid_count],
        senders=new_senders[:valid_count],
        receivers=new_receivers[:valid_count],
        n_node=graph.n_node,
        n_edge=jnp.array([valid_count]),
    )


class CurriculumGraphBuilder:
    """Graph builder with curriculum learning for communication degradation.
    
    Progressively introduces:
    1. Range limitations
    2. Packet loss
    3. Latency (future)
    
    Args:
        initial_radius: Starting communication radius (large = easy)
        final_radius: Target communication radius
        initial_dropout: Starting dropout rate (0 = easy)
        final_dropout: Target dropout rate
        curriculum_steps: Steps to reach final difficulty
    """
    
    def __init__(
        self,
        initial_radius: float = 200.0,
        final_radius: float = 50.0,
        initial_dropout: float = 0.0,
        final_dropout: float = 0.2,
        curriculum_steps: int = 500_000,
        k_neighbors: int = 5,
    ):
        self.initial_radius = initial_radius
        self.final_radius = final_radius
        self.initial_dropout = initial_dropout
        self.final_dropout = final_dropout
        self.curriculum_steps = curriculum_steps
        self.k_neighbors = k_neighbors
    
    def get_params(self, step: int) -> tuple[float, float]:
        """Get current curriculum parameters.
        
        Args:
            step: Current training step
            
        Returns:
            Tuple of (radius, dropout_rate)
        """
        progress = min(1.0, step / self.curriculum_steps)
        
        # Linear interpolation
        radius = self.initial_radius + (self.final_radius - self.initial_radius) * progress
        dropout = self.initial_dropout + (self.final_dropout - self.initial_dropout) * progress
        
        return radius, dropout
    
    def build(
        self,
        positions: Array,
        velocities: Array,
        step: int,
        key: Array,
    ) -> GraphData:
        """Build graph with current curriculum difficulty.
        
        Args:
            positions: Agent positions
            velocities: Agent velocities
            step: Current training step
            key: JAX PRNG key
            
        Returns:
            GraphData with curriculum-appropriate connectivity
        """
        radius, dropout = self.get_params(step)
        
        # Build radius-limited k-NN graph
        graph = build_radius_graph(
            positions, velocities,
            radius=radius,
            max_edges_per_node=self.k_neighbors,
        )
        
        # Apply packet dropout
        graph = apply_packet_dropout(graph, key, dropout)
        
        return graph


# JIT-compiled graph builders
@jax.jit
def build_graph_jit(positions: Array, velocities: Array, k: int = 5) -> GraphData:
    """JIT-compiled k-NN graph builder."""
    return build_knn_graph(positions, velocities, k)
