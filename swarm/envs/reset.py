"""Vectorized reset utilities for swarm environments."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from swarm.envs.dynamics import SwarmState


def reset_grid(
    num_agents: int,
    spacing: float = 5.0,
    height: float = 10.0,
) -> SwarmState:
    """Initialize swarm in a grid formation.
    
    Args:
        num_agents: Number of agents
        spacing: Distance between agents in meters
        height: Flight altitude
        
    Returns:
        Initial swarm state in grid formation
    """
    # Compute grid dimensions
    grid_size = int(jnp.ceil(jnp.sqrt(num_agents)))
    
    # Create grid positions
    indices = jnp.arange(num_agents)
    x = (indices % grid_size - grid_size / 2) * spacing
    y = (indices // grid_size - grid_size / 2) * spacing
    z = jnp.full(num_agents, height)
    pos = jnp.stack([x, y, z], axis=-1)
    
    return SwarmState(
        pos=pos,
        vel=jnp.zeros((num_agents, 3)),
        quat=jnp.tile(jnp.array([1.0, 0.0, 0.0, 0.0]), (num_agents, 1)),
        omega=jnp.zeros((num_agents, 3)),
        energy=jnp.ones(num_agents),
        time=jnp.array(0.0),
    )


def reset_random_circle(
    key: Array,
    num_agents: int,
    radius: float = 50.0,
    height: float = 10.0,
) -> SwarmState:
    """Initialize swarm randomly distributed in a circle.
    
    Args:
        key: JAX PRNG key
        num_agents: Number of agents
        radius: Circle radius in meters
        height: Flight altitude
        
    Returns:
        Initial swarm state
    """
    k1, k2 = jax.random.split(key)
    
    # Random angles and radii (sqrt for uniform distribution)
    angles = jax.random.uniform(k1, (num_agents,), minval=0, maxval=2 * jnp.pi)
    radii = radius * jnp.sqrt(jax.random.uniform(k2, (num_agents,)))
    
    x = radii * jnp.cos(angles)
    y = radii * jnp.sin(angles)
    z = jnp.full(num_agents, height)
    pos = jnp.stack([x, y, z], axis=-1)
    
    return SwarmState(
        pos=pos,
        vel=jnp.zeros((num_agents, 3)),
        quat=jnp.tile(jnp.array([1.0, 0.0, 0.0, 0.0]), (num_agents, 1)),
        omega=jnp.zeros((num_agents, 3)),
        energy=jnp.ones(num_agents),
        time=jnp.array(0.0),
    )


def reset_formation(
    formation: str,
    num_agents: int,
    key: Array | None = None,
    **kwargs,
) -> SwarmState:
    """Initialize swarm in specified formation.
    
    Args:
        formation: One of "grid", "circle", "random"
        num_agents: Number of agents
        key: JAX PRNG key (required for "random" and "circle")
        **kwargs: Formation-specific parameters
        
    Returns:
        Initial swarm state
    """
    if formation == "grid":
        return reset_grid(num_agents, **kwargs)
    elif formation == "circle" and key is not None:
        return reset_random_circle(key, num_agents, **kwargs)
    elif formation == "random" and key is not None:
        from swarm.envs.dynamics import PointMassDynamics
        dynamics = PointMassDynamics()
        return dynamics.reset_swarm(key, num_agents, **kwargs)
    else:
        raise ValueError(f"Unknown formation: {formation} or missing key")
