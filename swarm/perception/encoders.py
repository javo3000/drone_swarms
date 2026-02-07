"""Sensor encoders for processing agent observations."""

from __future__ import annotations

from typing import Sequence

import jax.numpy as jnp
from jax import Array
import flax.linen as nn


class StateEncoder(nn.Module):
    """Encode agent state (position, velocity, energy) to feature vector.
    
    Args:
        hidden_dims: Sequence of hidden layer dimensions
        output_dim: Output feature dimension
    """
    
    hidden_dims: Sequence[int] = (64, 64)
    output_dim: int = 32
    
    @nn.compact
    def __call__(self, state: Array) -> Array:
        """Encode agent state.
        
        Args:
            state: Agent state vector (pos, vel, energy) shape (7,) or (N, 7)
            
        Returns:
            Encoded features
        """
        x = state
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.LayerNorm()(x)
            x = nn.relu(x)
        x = nn.Dense(self.output_dim)(x)
        return x


class RelativeEncoder(nn.Module):
    """Encode relative observations (neighbor positions/velocities).
    
    Args:
        hidden_dims: Sequence of hidden layer dimensions
        output_dim: Output feature dimension
    """
    
    hidden_dims: Sequence[int] = (64, 64)
    output_dim: int = 32
    
    @nn.compact
    def __call__(self, rel_obs: Array) -> Array:
        """Encode relative observations.
        
        Args:
            rel_obs: Relative observation (rel_pos, rel_vel) shape (6,) or (N, K, 6)
            
        Returns:
            Encoded features
        """
        x = rel_obs
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.relu(x)
        x = nn.Dense(self.output_dim)(x)
        return x


class ObservationEncoder(nn.Module):
    """Full observation encoder combining self and neighbor features.
    
    Args:
        state_dim: Dimension of state encoder output
        neighbor_dim: Dimension of neighbor encoder output
        output_dim: Final output dimension
        max_neighbors: Maximum number of neighbors to process
    """
    
    state_dim: int = 32
    neighbor_dim: int = 32
    output_dim: int = 64
    max_neighbors: int = 5
    
    @nn.compact
    def __call__(
        self,
        own_state: Array,
        neighbor_features: Array,
        neighbor_mask: Array,
    ) -> Array:
        """Encode full observation.
        
        Args:
            own_state: Agent's own state (N, 7)
            neighbor_features: Neighbor features (N, K, feature_dim)
            neighbor_mask: Valid neighbor mask (N, K)
            
        Returns:
            Encoded observation (N, output_dim)
        """
        # Encode own state
        state_enc = StateEncoder(output_dim=self.state_dim)(own_state)
        
        # Encode and aggregate neighbor features
        neighbor_enc = RelativeEncoder(output_dim=self.neighbor_dim)(neighbor_features)
        
        # Masked mean aggregation
        mask = neighbor_mask[..., None].astype(jnp.float32)
        neighbor_enc = neighbor_enc * mask
        neighbor_agg = neighbor_enc.sum(axis=-2) / jnp.maximum(mask.sum(axis=-2), 1.0)
        
        # Combine
        combined = jnp.concatenate([state_enc, neighbor_agg], axis=-1)
        output = nn.Dense(self.output_dim)(combined)
        output = nn.relu(output)
        output = nn.Dense(self.output_dim)(output)
        
        return output
