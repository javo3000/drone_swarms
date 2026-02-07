"""Communication degradation models.

Simulates realistic communication challenges:
- Packet loss/dropout
- Latency/delays  
- Range limitations
- Jamming zones
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array


class CommsConfig(NamedTuple):
    """Communication configuration."""
    max_range: float = 100.0        # Maximum comm range (m)
    base_dropout: float = 0.0       # Base packet loss rate
    range_dropout_factor: float = 0.5  # Additional dropout at max range
    latency_mean: float = 0.0       # Mean latency (timesteps)
    latency_std: float = 0.0        # Latency std dev


def apply_range_dropout(
    adjacency: Array,
    distances: Array,
    max_range: float,
    dropout_factor: float = 0.5,
    key: Array = None,
) -> Array:
    """Apply distance-based dropout to communication graph.
    
    Packets more likely to drop at longer distances.
    
    Args:
        adjacency: Boolean adjacency matrix (N, N)
        distances: Distance matrix (N, N)
        max_range: Maximum communication range
        dropout_factor: Dropout probability at max range
        key: PRNG key
        
    Returns:
        Modified adjacency matrix
    """
    if key is None:
        return adjacency
    
    # Dropout probability increases linearly with distance
    dropout_prob = (distances / max_range) * dropout_factor
    dropout_prob = jnp.clip(dropout_prob, 0, dropout_factor)
    
    # Random dropout mask
    keep_mask = jax.random.uniform(key, adjacency.shape) > dropout_prob
    
    return adjacency & keep_mask


def apply_random_dropout(
    adjacency: Array,
    dropout_rate: float,
    key: Array,
) -> Array:
    """Apply random packet loss.
    
    Args:
        adjacency: Boolean adjacency matrix
        dropout_rate: Probability of dropping each edge
        key: PRNG key
        
    Returns:
        Modified adjacency matrix
    """
    keep_mask = jax.random.uniform(key, adjacency.shape) > dropout_rate
    return adjacency & keep_mask


def apply_jammer_zones(
    adjacency: Array,
    positions: Array,
    jammer_positions: Array,
    jammer_radii: Array,
) -> Array:
    """Block communication in jammer zones.
    
    Agents inside jammer radius cannot send OR receive.
    
    Args:
        adjacency: Boolean adjacency matrix (N, N)
        positions: Agent positions (N, 3)
        jammer_positions: Jammer locations (M, 3)
        jammer_radii: Jammer effect radii (M,)
        
    Returns:
        Modified adjacency matrix
    """
    num_agents = positions.shape[0]
    
    # Check which agents are jammed
    # Distance to each jammer
    dists_to_jammers = jnp.linalg.norm(
        positions[:, None, :] - jammer_positions[None, :, :],
        axis=-1
    )  # (N, M)
    
    # Agent is jammed if within any jammer radius
    jammed = jnp.any(dists_to_jammers < jammer_radii[None, :], axis=-1)  # (N,)
    
    # Block all edges from/to jammed agents
    jammed_mask = jammed[:, None] | jammed[None, :]  # (N, N)
    
    return adjacency & ~jammed_mask


class RealisticCommsModel:
    """Realistic communication model with multiple degradation effects.
    
    Args:
        config: Communication configuration
    """
    
    def __init__(self, config: CommsConfig | None = None):
        self.config = config or CommsConfig()
    
    def apply(
        self,
        adjacency: Array,
        distances: Array,
        positions: Array,
        key: Array,
        jammer_positions: Array | None = None,
        jammer_radii: Array | None = None,
    ) -> Array:
        """Apply all communication effects.
        
        Args:
            adjacency: Base adjacency matrix
            distances: Distance matrix
            positions: Agent positions
            key: PRNG key
            jammer_positions: Optional jammer locations
            jammer_radii: Optional jammer radii
            
        Returns:
            Degraded adjacency matrix
        """
        key1, key2 = jax.random.split(key)
        
        # Range-based dropout
        adjacency = apply_range_dropout(
            adjacency, distances,
            self.config.max_range,
            self.config.range_dropout_factor,
            key1,
        )
        
        # Random dropout
        if self.config.base_dropout > 0:
            adjacency = apply_random_dropout(
                adjacency,
                self.config.base_dropout,
                key2,
            )
        
        # Jammer zones
        if jammer_positions is not None and jammer_radii is not None:
            adjacency = apply_jammer_zones(
                adjacency, positions,
                jammer_positions, jammer_radii,
            )
        
        return adjacency


class CurriculumCommsModel:
    """Communication model with curriculum learning.
    
    Progressively increases communication difficulty during training.
    
    Args:
        initial_config: Easy starting configuration
        final_config: Hard final configuration
        curriculum_steps: Steps to reach final difficulty
    """
    
    def __init__(
        self,
        initial_config: CommsConfig | None = None,
        final_config: CommsConfig | None = None,
        curriculum_steps: int = 500_000,
    ):
        self.initial = initial_config or CommsConfig(
            max_range=200.0,
            base_dropout=0.0,
            range_dropout_factor=0.0,
        )
        self.final = final_config or CommsConfig(
            max_range=50.0,
            base_dropout=0.1,
            range_dropout_factor=0.3,
        )
        self.curriculum_steps = curriculum_steps
    
    def get_config(self, step: int) -> CommsConfig:
        """Get interpolated config at current step."""
        progress = min(1.0, step / self.curriculum_steps)
        
        return CommsConfig(
            max_range=self._lerp(self.initial.max_range, self.final.max_range, progress),
            base_dropout=self._lerp(self.initial.base_dropout, self.final.base_dropout, progress),
            range_dropout_factor=self._lerp(
                self.initial.range_dropout_factor,
                self.final.range_dropout_factor,
                progress
            ),
            latency_mean=self._lerp(self.initial.latency_mean, self.final.latency_mean, progress),
            latency_std=self._lerp(self.initial.latency_std, self.final.latency_std, progress),
        )
    
    def apply(
        self,
        adjacency: Array,
        distances: Array,
        positions: Array,
        key: Array,
        step: int,
        **kwargs,
    ) -> Array:
        """Apply curriculum communication effects."""
        config = self.get_config(step)
        model = RealisticCommsModel(config)
        return model.apply(adjacency, distances, positions, key, **kwargs)
    
    @staticmethod
    def _lerp(a: float, b: float, t: float) -> float:
        """Linear interpolation."""
        return a + (b - a) * t
