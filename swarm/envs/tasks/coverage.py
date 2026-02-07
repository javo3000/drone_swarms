"""Area coverage task for swarm coordination.

The coverage task rewards agents for spreading out to maximize
area coverage while avoiding collisions and staying in bounds.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from swarm.envs.dynamics import SwarmState


def compute_coverage_reward(
    state: SwarmState,
    arena_size: float = 100.0,
    grid_resolution: int = 20,
) -> tuple[Array, dict[str, Array]]:
    """Compute coverage reward based on Voronoi-like partitioning.
    
    Divides arena into grid and measures how well agents cover it.
    
    Args:
        state: Current swarm state
        arena_size: Size of arena in meters
        grid_resolution: Number of grid cells per side
        
    Returns:
        Tuple of (reward, info_dict)
    """
    num_agents = state.pos.shape[0]
    
    # Create grid points
    x = jnp.linspace(-arena_size/2, arena_size/2, grid_resolution)
    y = jnp.linspace(-arena_size/2, arena_size/2, grid_resolution)
    grid_x, grid_y = jnp.meshgrid(x, y)
    grid_points = jnp.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)  # (G, 2)
    
    # Distance from each grid point to each agent
    agent_xy = state.pos[:, :2]  # (N, 2)
    dists = jnp.linalg.norm(
        grid_points[:, None, :] - agent_xy[None, :, :],
        axis=-1
    )  # (G, N)
    
    # Coverage = sum of inverse distances (soft coverage)
    min_dists = jnp.min(dists, axis=-1)  # (G,)
    coverage_score = jnp.mean(1.0 / (1.0 + min_dists))
    
    # Per-agent contribution: how many grid points is this agent closest to?
    closest_agent = jnp.argmin(dists, axis=-1)  # (G,)
    agent_coverage = jax.vmap(
        lambda i: jnp.sum(closest_agent == i)
    )(jnp.arange(num_agents))  # (N,)
    
    # Normalize by ideal coverage (uniform distribution)
    ideal_coverage = grid_resolution ** 2 / num_agents
    coverage_balance = 1.0 - jnp.std(agent_coverage) / ideal_coverage
    
    # Per-agent reward based on local coverage
    reward = agent_coverage / ideal_coverage * coverage_balance
    
    info = {
        "coverage_score": coverage_score,
        "coverage_balance": coverage_balance,
        "min_agent_coverage": agent_coverage.min(),
        "max_agent_coverage": agent_coverage.max(),
    }
    
    return reward, info


def compute_voronoi_targets(
    state: SwarmState,
    arena_size: float = 100.0,
    grid_resolution: int = 20,
) -> Array:
    """Compute Voronoi centroids as target positions.
    
    Uses Lloyd's algorithm iteration: each agent should move
    towards the centroid of its Voronoi cell.
    
    Args:
        state: Current swarm state
        arena_size: Size of arena
        grid_resolution: Grid resolution for centroid computation
        
    Returns:
        Target positions (num_agents, 3)
    """
    num_agents = state.pos.shape[0]
    
    # Create grid points
    x = jnp.linspace(-arena_size/2, arena_size/2, grid_resolution)
    y = jnp.linspace(-arena_size/2, arena_size/2, grid_resolution)
    grid_x, grid_y = jnp.meshgrid(x, y)
    grid_points = jnp.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)  # (G, 2)
    
    # Distance from each grid point to each agent
    agent_xy = state.pos[:, :2]  # (N, 2)
    dists = jnp.linalg.norm(
        grid_points[:, None, :] - agent_xy[None, :, :],
        axis=-1
    )  # (G, N)
    
    # Assign grid points to nearest agent
    closest_agent = jnp.argmin(dists, axis=-1)  # (G,)
    
    # Compute centroid of each agent's Voronoi cell
    def compute_centroid(agent_idx):
        mask = closest_agent == agent_idx
        cell_points = jnp.where(mask[:, None], grid_points, 0.0)
        count = jnp.maximum(mask.sum(), 1)
        centroid = cell_points.sum(axis=0) / count
        return centroid
    
    centroids_xy = jax.vmap(compute_centroid)(jnp.arange(num_agents))  # (N, 2)
    
    # Keep current height
    centroids = jnp.concatenate([centroids_xy, state.pos[:, 2:3]], axis=-1)
    
    return centroids


def coverage_task_reward(
    state: SwarmState,
    new_state: SwarmState,
    actions: Array,
    config: dict | None = None,
) -> tuple[Array, dict]:
    """Complete coverage task reward function.
    
    Components:
    - Coverage: Reward for good area coverage
    - Centroid: Reward for moving towards Voronoi centroid
    - Smoothness: Penalty for jerky movements
    """
    cfg = config or {}
    arena_size = cfg.get("arena_size", 100.0)
    
    # Coverage component
    coverage_reward, coverage_info = compute_coverage_reward(new_state, arena_size)
    
    # Centroid tracking component
    targets = compute_voronoi_targets(new_state, arena_size)
    dist_to_target = jnp.linalg.norm(new_state.pos - targets, axis=-1)
    centroid_reward = -dist_to_target / arena_size  # Normalized
    
    # Smoothness (penalize large velocity changes)
    vel_change = jnp.linalg.norm(new_state.vel - state.vel, axis=-1)
    smoothness_penalty = -0.1 * vel_change
    
    # Total reward
    reward = (
        1.0 * coverage_reward
        + 0.5 * centroid_reward
        + 0.1 * smoothness_penalty
    )
    
    info = {
        **coverage_info,
        "centroid_dist": dist_to_target.mean(),
        "velocity_change": vel_change.mean(),
    }
    
    return reward, info
