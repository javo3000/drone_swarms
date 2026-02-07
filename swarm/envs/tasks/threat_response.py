"""Threat response task.

Defines scenarios with moving intruders that the swarm must:
- Detect
- Track
- Intercept
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from swarm.envs.dynamics import SwarmState


class ThreatState(NamedTuple):
    """State of threats/intruders."""
    positions: Array     # (M, 3) threat positions
    velocities: Array    # (M, 3) threat velocities
    active: Array        # (M,) boolean active flags
    detected: Array      # (M,) boolean detected flags
    intercepted: Array   # (M,) boolean intercepted flags


class ThreatConfig(NamedTuple):
    """Threat scenario configuration."""
    num_threats: int = 3
    spawn_radius: float = 80.0      # Spawn at edge of arena
    threat_speed: float = 5.0       # Threat movement speed
    detection_radius: float = 20.0  # Range to detect threat
    intercept_radius: float = 5.0   # Range to intercept
    respawn_on_intercept: bool = True


def spawn_threats(
    key: Array,
    num_threats: int,
    spawn_radius: float = 80.0,
    target_center: Array | None = None,
) -> ThreatState:
    """Spawn threats at random edge positions.
    
    Args:
        key: PRNG key
        num_threats: Number of threats
        spawn_radius: Distance from center to spawn
        target_center: Where threats should move toward (default: origin)
        
    Returns:
        Initial threat state
    """
    key1, key2 = jax.random.split(key)
    
    # Random angles for spawn positions
    angles = jax.random.uniform(key1, (num_threats,)) * 2 * jnp.pi
    
    # Spawn at edge
    spawn_x = spawn_radius * jnp.cos(angles)
    spawn_y = spawn_radius * jnp.sin(angles)
    spawn_z = jax.random.uniform(key2, (num_threats,)) * 20 + 10  # 10-30m altitude
    
    positions = jnp.stack([spawn_x, spawn_y, spawn_z], axis=-1)
    
    # Velocities toward center
    target = target_center if target_center is not None else jnp.zeros(3)
    directions = target - positions
    directions = directions / jnp.linalg.norm(directions, axis=-1, keepdims=True)
    velocities = directions * 5.0  # Default speed
    
    return ThreatState(
        positions=positions,
        velocities=velocities,
        active=jnp.ones(num_threats, dtype=bool),
        detected=jnp.zeros(num_threats, dtype=bool),
        intercepted=jnp.zeros(num_threats, dtype=bool),
    )


def update_threats(
    threat_state: ThreatState,
    swarm_state: SwarmState,
    config: ThreatConfig,
    dt: float = 0.02,
    key: Array | None = None,
) -> ThreatState:
    """Update threat positions and check for detection/interception.
    
    Args:
        threat_state: Current threat state
        swarm_state: Current swarm state
        config: Threat configuration
        dt: Time step
        key: PRNG key for respawning
        
    Returns:
        Updated threat state
    """
    # Move threats
    new_positions = threat_state.positions + threat_state.velocities * dt
    
    # Check detection (any agent within detection radius)
    dists_to_agents = jnp.linalg.norm(
        threat_state.positions[:, None, :] - swarm_state.pos[None, :, :],
        axis=-1
    )  # (M, N)
    
    min_dist_to_agent = jnp.min(dists_to_agents, axis=-1)  # (M,)
    
    newly_detected = min_dist_to_agent < config.detection_radius
    detected = threat_state.detected | newly_detected
    
    # Check interception
    newly_intercepted = min_dist_to_agent < config.intercept_radius
    intercepted = threat_state.intercepted | newly_intercepted
    
    # Deactivate intercepted threats
    active = threat_state.active & ~intercepted
    
    # Respawn intercepted threats
    if config.respawn_on_intercept and key is not None:
        needs_respawn = intercepted & threat_state.active
        
        if jnp.any(needs_respawn):
            respawn_key = jax.random.split(key, num=int(needs_respawn.sum()))
            # Note: In practice, would need to handle dynamic respawning
            pass
    
    return ThreatState(
        positions=new_positions,
        velocities=threat_state.velocities,
        active=active,
        detected=detected,
        intercepted=intercepted,
    )


def compute_threat_reward(
    swarm_state: SwarmState,
    threat_state: ThreatState,
    config: ThreatConfig,
) -> tuple[Array, dict]:
    """Compute reward for threat response.
    
    Rewards:
    - Detection: Small reward for getting close to threats
    - Interception: Large reward for intercepting
    - Tracking: Continuous reward for maintaining coverage
    
    Args:
        swarm_state: Swarm state
        threat_state: Threat state
        config: Threat configuration
        
    Returns:
        Tuple of (per-agent rewards, info dict)
    """
    num_agents = swarm_state.pos.shape[0]
    
    # Distance from each agent to each threat
    dists = jnp.linalg.norm(
        swarm_state.pos[:, None, :] - threat_state.positions[None, :, :],
        axis=-1
    )  # (N, M)
    
    # Only consider active threats
    dists = jnp.where(threat_state.active[None, :], dists, jnp.inf)
    
    # Closest threat for each agent
    min_dist_to_threat = jnp.min(dists, axis=-1)  # (N,)
    
    # Detection reward: smooth reward for approaching threats
    detection_reward = jnp.where(
        min_dist_to_threat < config.detection_radius * 2,
        (config.detection_radius * 2 - min_dist_to_threat) / config.detection_radius,
        0.0
    )
    
    # Interception reward: bonus for very close
    intercept_reward = jnp.where(
        min_dist_to_threat < config.intercept_radius * 2,
        5.0 * (config.intercept_radius * 2 - min_dist_to_threat) / config.intercept_radius,
        0.0
    )
    
    # Total reward
    reward = detection_reward + intercept_reward
    
    info = {
        "detection_reward": detection_reward.mean(),
        "intercept_reward": intercept_reward.mean(),
        "threats_detected": threat_state.detected.sum(),
        "threats_intercepted": threat_state.intercepted.sum(),
        "threats_active": threat_state.active.sum(),
    }
    
    return reward, info


class ForbiddenZone(NamedTuple):
    """A forbidden zone that agents must avoid."""
    center: Array       # (3,) center position
    radius: float       # Zone radius
    penalty: float = 10.0  # Penalty for entering


def compute_forbidden_zone_penalty(
    state: SwarmState,
    zones: list[ForbiddenZone],
) -> Array:
    """Compute penalty for entering forbidden zones.
    
    Args:
        state: Swarm state
        zones: List of forbidden zones
        
    Returns:
        Per-agent penalty (N,)
    """
    penalty = jnp.zeros(state.pos.shape[0])
    
    for zone in zones:
        dists = jnp.linalg.norm(state.pos - zone.center, axis=-1)
        in_zone = dists < zone.radius
        penalty = penalty + in_zone * zone.penalty
    
    return penalty
