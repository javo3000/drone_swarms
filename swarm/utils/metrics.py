"""Metrics utilities for swarm evaluation."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from swarm.envs.dynamics import SwarmState


def compute_coverage_percentage(
    state: SwarmState,
    arena_size: float = 100.0,
    cell_size: float = 5.0,
) -> float:
    """Compute percentage of arena cells covered by agents.
    
    Args:
        state: Current swarm state
        arena_size: Size of arena
        cell_size: Size of each grid cell
        
    Returns:
        Coverage percentage (0-100)
    """
    num_cells = int(arena_size / cell_size)
    half = arena_size / 2
    
    # Create grid
    x = jnp.linspace(-half + cell_size/2, half - cell_size/2, num_cells)
    y = jnp.linspace(-half + cell_size/2, half - cell_size/2, num_cells)
    grid_x, grid_y = jnp.meshgrid(x, y)
    grid_points = jnp.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)
    
    # Check coverage
    agent_xy = state.pos[:, :2]
    dists = jnp.linalg.norm(
        grid_points[:, None, :] - agent_xy[None, :, :],
        axis=-1
    )
    min_dists = jnp.min(dists, axis=-1)
    covered = min_dists < cell_size
    
    return 100.0 * float(covered.mean())


def compute_min_separation(state: SwarmState) -> float:
    """Compute minimum separation between any two agents."""
    from swarm.utils.geometry import pairwise_distances
    
    dists = pairwise_distances(state.pos)
    dists = dists + jnp.eye(state.pos.shape[0]) * 1e10
    return float(jnp.min(dists))


def compute_mean_speed(state: SwarmState) -> float:
    """Compute mean agent speed."""
    speeds = jnp.linalg.norm(state.vel, axis=-1)
    return float(jnp.mean(speeds))


def compute_energy_remaining(state: SwarmState) -> float:
    """Compute mean remaining energy."""
    return float(jnp.mean(state.energy))


def compute_all_metrics(
    state: SwarmState,
    arena_size: float = 100.0,
) -> dict[str, float]:
    """Compute all evaluation metrics.
    
    Returns dictionary with:
    - coverage_pct: Percentage of arena covered
    - min_separation: Minimum agent separation
    - mean_speed: Average agent speed
    - energy_remaining: Mean remaining energy
    """
    return {
        "coverage_pct": compute_coverage_percentage(state, arena_size),
        "min_separation": compute_min_separation(state),
        "mean_speed": compute_mean_speed(state),
        "energy_remaining": compute_energy_remaining(state),
    }
