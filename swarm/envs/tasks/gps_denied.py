"""GPS-denied operation scenarios.

Simulates areas where GPS/external positioning is unavailable,
forcing agents to use relative localization.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from swarm.envs.dynamics import SwarmState


class GPSJammer(NamedTuple):
    """GPS jammer zone configuration."""
    position: Array     # (3,) jammer center
    radius: float       # Effect radius
    noise_std: float = 5.0  # Position noise std when jammed


def apply_gps_jamming(
    state: SwarmState,
    jammers: list[GPSJammer],
    key: Array,
) -> tuple[Array, Array]:
    """Apply GPS jamming to agent observations.
    
    Agents in jammed zones receive noisy/no position information.
    
    Args:
        state: True swarm state
        jammers: List of GPS jammers
        key: PRNG key for noise
        
    Returns:
        Tuple of (observed_positions, jammed_mask)
    """
    num_agents = state.pos.shape[0]
    
    # Check which agents are jammed
    jammed = jnp.zeros(num_agents, dtype=bool)
    
    for jammer in jammers:
        dists = jnp.linalg.norm(state.pos - jammer.position, axis=-1)
        in_zone = dists < jammer.radius
        jammed = jammed | in_zone
    
    # Add noise to jammed agents' position observations
    noise = jax.random.normal(key, state.pos.shape)
    
    # Scale noise by jammer strength
    max_noise_std = max(j.noise_std for j in jammers) if jammers else 5.0
    noise = noise * max_noise_std
    
    # Apply noise only to jammed agents
    observed_pos = jnp.where(
        jammed[:, None],
        state.pos + noise,
        state.pos
    )
    
    return observed_pos, jammed


class GPSDeniedTask:
    """Task for operating in GPS-denied environments.
    
    Agents must:
    1. Detect they're in a jammed zone (position uncertainty high)
    2. Switch to relative navigation using neighbors
    3. Maintain formation/coverage despite uncertainty
    
    Args:
        jammers: List of GPS jammers in environment
        uncertainty_threshold: Threshold to detect jamming
    """
    
    def __init__(
        self,
        jammers: list[GPSJammer] | None = None,
        uncertainty_threshold: float = 3.0,
    ):
        self.jammers = jammers or []
        self.uncertainty_threshold = uncertainty_threshold
    
    def add_jammer(
        self,
        position: Array,
        radius: float = 20.0,
        noise_std: float = 5.0,
    ):
        """Add a GPS jammer to the environment."""
        self.jammers.append(GPSJammer(
            position=jnp.array(position),
            radius=radius,
            noise_std=noise_std,
        ))
    
    def get_observations(
        self,
        state: SwarmState,
        key: Array,
    ) -> tuple[Array, dict]:
        """Get GPS-affected observations.
        
        Args:
            state: True swarm state
            key: PRNG key
            
        Returns:
            Tuple of (observed_positions, info)
        """
        if not self.jammers:
            return state.pos, {"jammed_agents": 0}
        
        observed_pos, jammed = apply_gps_jamming(state, self.jammers, key)
        
        info = {
            "jammed_agents": jammed.sum(),
            "jammed_mask": jammed,
            "position_error": jnp.linalg.norm(observed_pos - state.pos, axis=-1),
        }
        
        return observed_pos, info
    
    def compute_reward(
        self,
        state: SwarmState,
        observed_positions: Array,
        key: Array,
    ) -> tuple[Array, dict]:
        """Compute reward for GPS-denied operation.
        
        Rewards agents for:
        - Maintaining formation despite uncertainty
        - Staying connected to unjammed neighbors
        - Operating effectively in jammed zones
        
        Args:
            state: True state
            observed_positions: What agents think their positions are
            key: PRNG key
            
        Returns:
            Tuple of (per-agent rewards, info)
        """
        num_agents = state.pos.shape[0]
        
        # Check which agents are jammed
        jammed = jnp.zeros(num_agents, dtype=bool)
        for jammer in self.jammers:
            dists = jnp.linalg.norm(state.pos - jammer.position, axis=-1)
            jammed = jammed | (dists < jammer.radius)
        
        # Reward for maintaining connectivity to unjammed neighbors
        # (This helps jammed agents localize using relative info)
        
        # Pairwise distances
        dists = jnp.linalg.norm(
            state.pos[:, None, :] - state.pos[None, :, :],
            axis=-1
        )
        dists = dists + jnp.eye(num_agents) * 1e10
        
        # Jammed agents get reward for being near unjammed agents
        unjammed_mask = ~jammed
        dist_to_unjammed = jnp.where(
            unjammed_mask[None, :],
            dists,
            jnp.inf
        )
        min_dist_to_unjammed = jnp.min(dist_to_unjammed, axis=-1)
        
        # Reward for jammed agents staying connected
        connectivity_reward = jnp.where(
            jammed & (min_dist_to_unjammed < 30.0),
            1.0 - min_dist_to_unjammed / 30.0,
            0.0
        )
        
        # Penalty for large position errors (encourages uncertainty awareness)
        position_error = jnp.linalg.norm(observed_positions - state.pos, axis=-1)
        
        reward = connectivity_reward
        
        info = {
            "connectivity_reward": connectivity_reward.mean(),
            "position_error": position_error.mean(),
            "jammed_ratio": jammed.mean(),
        }
        
        return reward, info


def create_gps_denied_scenario(
    arena_size: float = 100.0,
    num_jammers: int = 2,
    jammer_radius: float = 20.0,
    key: Array = None,
) -> GPSDeniedTask:
    """Create a GPS-denied scenario with random jammer placement.
    
    Args:
        arena_size: Size of arena
        num_jammers: Number of jammers
        jammer_radius: Radius of each jammer
        key: PRNG key for placement
        
    Returns:
        Configured GPS denied task
    """
    task = GPSDeniedTask()
    
    if key is None:
        key = jax.random.PRNGKey(0)
    
    # Random jammer positions
    half = arena_size / 2 * 0.6  # Keep inside arena
    positions = jax.random.uniform(key, (num_jammers, 2)) * 2 * half - half
    
    for i in range(num_jammers):
        task.add_jammer(
            position=jnp.array([positions[i, 0], positions[i, 1], 0.0]),
            radius=jammer_radius,
            noise_std=5.0,
        )
    
    return task
