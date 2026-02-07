"""Cost functions for MPC and reward shaping."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from swarm.envs.dynamics import SwarmState


def position_tracking_cost(
    state: SwarmState,
    target_positions: Array,
    weight: float = 1.0,
) -> Array:
    """Cost for tracking target positions.
    
    Args:
        state: Current swarm state
        target_positions: Target positions (N, 3)
        weight: Cost weight
        
    Returns:
        Per-agent cost (N,)
    """
    error = state.pos - target_positions
    return weight * jnp.sum(error ** 2, axis=-1)


def velocity_tracking_cost(
    state: SwarmState,
    target_velocities: Array,
    weight: float = 0.1,
) -> Array:
    """Cost for tracking target velocities."""
    error = state.vel - target_velocities
    return weight * jnp.sum(error ** 2, axis=-1)


def control_effort_cost(
    actions: Array,
    weight: float = 0.01,
) -> Array:
    """Cost for control effort (thrust magnitude).
    
    Args:
        actions: Control actions (N, 3)
        weight: Cost weight
        
    Returns:
        Per-agent cost (N,)
    """
    return weight * jnp.sum(actions ** 2, axis=-1)


def energy_efficiency_cost(
    state: SwarmState,
    actions: Array,
    weight: float = 0.5,
) -> Array:
    """Cost for energy consumption.
    
    Penalizes high thrust when moving fast (inefficient).
    """
    speed = jnp.linalg.norm(state.vel, axis=-1)
    thrust = jnp.linalg.norm(actions, axis=-1)
    return weight * thrust * speed


def collision_avoidance_cost(
    state: SwarmState,
    min_distance: float = 3.0,
    weight: float = 10.0,
) -> Array:
    """Cost for collision avoidance.
    
    Exponential penalty when agents get too close.
    
    Args:
        state: Current swarm state
        min_distance: Minimum safe distance
        weight: Cost weight
        
    Returns:
        Per-agent cost (N,)
    """
    num_agents = state.pos.shape[0]
    
    # Pairwise distances
    diff = state.pos[:, None, :] - state.pos[None, :, :]
    dists = jnp.linalg.norm(diff, axis=-1)
    dists = dists + jnp.eye(num_agents) * 1e10  # Exclude self
    
    # Minimum distance to any other agent
    min_dists = jnp.min(dists, axis=-1)
    
    # Exponential penalty below threshold
    violation = jnp.maximum(0, min_distance - min_dists)
    return weight * jnp.exp(violation) - weight


def boundary_cost(
    state: SwarmState,
    arena_size: float = 100.0,
    weight: float = 5.0,
) -> Array:
    """Cost for leaving arena boundaries."""
    half = arena_size / 2
    
    # Distance beyond boundary (0 if inside)
    x_violation = jnp.maximum(0, jnp.abs(state.pos[:, 0]) - half)
    y_violation = jnp.maximum(0, jnp.abs(state.pos[:, 1]) - half)
    
    return weight * (x_violation ** 2 + y_violation ** 2)


def threat_response_cost(
    state: SwarmState,
    threat_positions: Array | None,
    detection_radius: float = 20.0,
    intercept_reward: float = 10.0,
) -> Array:
    """Cost (negative reward) for threat response.
    
    Args:
        state: Current swarm state
        threat_positions: Threat locations (M, 3) or None
        detection_radius: Radius for threat detection
        intercept_reward: Reward for intercepting threat
        
    Returns:
        Per-agent cost (N,) - negative when near threats
    """
    if threat_positions is None or len(threat_positions) == 0:
        return jnp.zeros(state.pos.shape[0])
    
    # Distance to each threat
    dists_to_threats = jnp.linalg.norm(
        state.pos[:, None, :] - threat_positions[None, :, :],
        axis=-1
    )  # (N, M)
    
    # Reward for being close to threats (within detection radius)
    close_to_threat = dists_to_threats < detection_radius
    threat_reward = jnp.sum(close_to_threat.astype(jnp.float32) * intercept_reward, axis=-1)
    
    return -threat_reward  # Negative cost = reward


def total_mpc_cost(
    state: SwarmState,
    actions: Array,
    target_positions: Array | None = None,
    target_velocities: Array | None = None,
    arena_size: float = 100.0,
    weights: dict | None = None,
) -> Array:
    """Compute total MPC cost combining all components.
    
    Args:
        state: Current swarm state
        actions: Control actions
        target_positions: Target positions (optional)
        target_velocities: Target velocities (optional)
        arena_size: Arena size
        weights: Dictionary of cost weights
        
    Returns:
        Total per-agent cost (N,)
    """
    w = weights or {}
    cost = jnp.zeros(state.pos.shape[0])
    
    if target_positions is not None:
        cost += position_tracking_cost(state, target_positions, w.get("position", 1.0))
    
    if target_velocities is not None:
        cost += velocity_tracking_cost(state, target_velocities, w.get("velocity", 0.1))
    
    cost += control_effort_cost(actions, w.get("control", 0.01))
    cost += energy_efficiency_cost(state, actions, w.get("energy", 0.5))
    cost += collision_avoidance_cost(state, w.get("min_distance", 3.0), w.get("collision", 10.0))
    cost += boundary_cost(state, arena_size, w.get("boundary", 5.0))
    
    return cost
