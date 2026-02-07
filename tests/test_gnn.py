"""Tests for GNN perception modules."""

import pytest
import jax
import jax.numpy as jnp

from swarm.perception.graph_builder import (
    build_radius_graph,
    build_knn_graph,
    apply_packet_dropout,
    CurriculumGraphBuilder,
    GraphData,
)
from swarm.perception.gnn import SwarmGNN, create_gnn, init_gnn
from swarm.perception.encoders import StateEncoder, ObservationEncoder


class TestGraphBuilder:
    """Test graph construction."""
    
    def test_knn_graph(self):
        """Test k-NN graph construction."""
        positions = jax.random.normal(jax.random.PRNGKey(0), (10, 3))
        velocities = jnp.zeros((10, 3))
        
        graph = build_knn_graph(positions, velocities, k=3)
        
        assert graph.nodes.shape == (10, 6)  # pos + vel
        assert graph.edges.shape[0] == 10 * 3  # 3 edges per node
        assert graph.senders.shape[0] == 10 * 3
        assert graph.receivers.shape[0] == 10 * 3
    
    def test_radius_graph(self):
        """Test radius-based graph construction."""
        # Create agents in a line, spaced 10m apart
        positions = jnp.stack([
            jnp.arange(5) * 10.0,
            jnp.zeros(5),
            jnp.zeros(5),
        ], axis=-1)  # (5, 3)
        velocities = jnp.zeros((5, 3))
        
        # Radius 15 should connect adjacent agents only
        graph = build_radius_graph(positions, velocities, radius=15.0)
        
        assert graph.nodes.shape == (5, 6)
        # Each agent should connect to ~2 neighbors (except edges)
        assert graph.n_edge[0] > 0
    
    def test_packet_dropout(self):
        """Test packet dropout removes edges."""
        positions = jax.random.normal(jax.random.PRNGKey(0), (10, 3))
        velocities = jnp.zeros((10, 3))
        graph = build_knn_graph(positions, velocities, k=5)
        
        original_edges = graph.edges.shape[0]
        
        # Apply 50% dropout
        key = jax.random.PRNGKey(42)
        dropped_graph = apply_packet_dropout(graph, key, dropout_rate=0.5)
        
        # Should have fewer edges (approximately half)
        assert dropped_graph.edges.shape[0] < original_edges
    
    def test_curriculum_builder(self):
        """Test curriculum graph builder."""
        builder = CurriculumGraphBuilder(
            initial_radius=100.0,
            final_radius=50.0,
            initial_dropout=0.0,
            final_dropout=0.2,
            curriculum_steps=1000,
        )
        
        # Test curriculum progression
        radius_0, dropout_0 = builder.get_params(0)
        assert radius_0 == 100.0
        assert dropout_0 == 0.0
        
        radius_500, dropout_500 = builder.get_params(500)
        assert 70 < radius_500 < 80  # Midpoint
        assert 0.08 < dropout_500 < 0.12
        
        radius_1000, dropout_1000 = builder.get_params(1000)
        assert radius_1000 == 50.0
        assert dropout_1000 == 0.2


class TestGNN:
    """Test GNN modules."""
    
    def test_gnn_forward(self):
        """Test GNN forward pass."""
        gnn = create_gnn(hidden_dim=64, output_dim=32, num_layers=2)
        
        # Create dummy graph
        positions = jax.random.normal(jax.random.PRNGKey(0), (10, 3))
        velocities = jnp.zeros((10, 3))
        graph = build_knn_graph(positions, velocities, k=3)
        
        # Initialize and apply
        key = jax.random.PRNGKey(42)
        params = gnn.init(key, graph)
        output = gnn.apply(params, graph)
        
        assert output.shape == (10, 32)  # (num_agents, output_dim)
    
    def test_gnn_init_helper(self):
        """Test GNN initialization helper."""
        gnn = create_gnn(hidden_dim=64, output_dim=32, num_layers=2)
        key = jax.random.PRNGKey(0)
        
        params = init_gnn(gnn, key, num_agents=10)
        
        assert "params" in params


class TestEncoders:
    """Test encoder modules."""
    
    def test_state_encoder(self):
        """Test state encoder."""
        encoder = StateEncoder(hidden_dims=(32, 32), output_dim=16)
        
        state = jnp.zeros((5, 7))  # 5 agents, 7-dim state
        key = jax.random.PRNGKey(0)
        params = encoder.init(key, state)
        
        output = encoder.apply(params, state)
        assert output.shape == (5, 16)
    
    def test_observation_encoder(self):
        """Test full observation encoder."""
        encoder = ObservationEncoder(
            state_dim=16,
            neighbor_dim=16,
            output_dim=32,
            max_neighbors=5,
        )
        
        own_state = jnp.zeros((10, 7))
        neighbor_features = jnp.zeros((10, 5, 6))  # 5 neighbors, 6 features
        neighbor_mask = jnp.ones((10, 5), dtype=bool)
        
        key = jax.random.PRNGKey(0)
        params = encoder.init(key, own_state, neighbor_features, neighbor_mask)
        
        output = encoder.apply(params, own_state, neighbor_features, neighbor_mask)
        assert output.shape == (10, 32)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
