"""MJX-based multi-agent swarm environment.

This module provides a gymnasium-style environment for multi-robot
swarm simulation using MuJoCo's MJX backend for JAX acceleration.
"""

from __future__ import annotations

from typing import Any, NamedTuple
from functools import partial

import jax
import jax.numpy as jnp
from jax import Array

from swarm.envs.dynamics import PointMassDynamics, SwarmState


class EnvConfig(NamedTuple):
    """Environment configuration."""
    num_agents: int = 10
    arena_size: float = 100.0
    init_height: float = 10.0
    max_steps: int = 1000
    dt: float = 0.02
    
    # Reward weights
    coverage_weight: float = 1.0
    collision_weight: float = -10.0
    energy_weight: float = 0.01
    boundary_weight: float = -5.0
    
    # Physical parameters
    collision_radius: float = 1.0
    min_separation: float = 3.0


class Observation(NamedTuple):
    """Agent observations.
    
    Attributes:
        own_state: Agent's own state (pos, vel, energy)
        relative_positions: Positions of neighbors relative to self
        relative_velocities: Velocities of neighbors relative to self
        neighbor_mask: Boolean mask for valid neighbors
    """
    own_state: Array       # (N, 7) - pos(3) + vel(3) + energy(1)
    relative_positions: Array  # (N, K, 3) - K nearest neighbors
    relative_velocities: Array # (N, K, 3)
    neighbor_mask: Array   # (N, K) - valid neighbor flags


class StepResult(NamedTuple):
    """Result of environment step."""
    state: SwarmState
    obs: Observation
    reward: Array
    done: Array
    info: dict[str, Any]


class SwarmEnv:
    """Multi-agent swarm environment with JAX acceleration.
    
    A vectorized environment where all agents are simulated in parallel
    using JAX primitives. Supports vmap for batch simulation and shard_map
    for multi-GPU scaling.
    
    Args:
        config: Environment configuration
    """
    
    def __init__(self, config: EnvConfig | None = None):
        self.config = config or EnvConfig()
        self.dynamics = PointMassDynamics(dt=self.config.dt)
        
        # Observation dimensions
        self.obs_dim = 7 + 6 * 5  # own_state + 5 neighbors * (rel_pos + rel_vel)
        self.action_dim = 3  # 3D thrust vector
        
    def reset(self, key: Array) -> tuple[SwarmState, Observation]:
        """Reset environment to initial state.
        
        Args:
            key: JAX PRNG key
            
        Returns:
            Initial state and observations
        """
        state = self.dynamics.reset_swarm(
            key,
            self.config.num_agents,
            self.config.arena_size,
            self.config.init_height,
        )
        obs = self._get_observations(state)
        return state, obs
    
    def step(
        self,
        state: SwarmState,
        actions: Array,
    ) -> StepResult:
        """Execute one environment step.
        
        Args:
            state: Current swarm state
            actions: Thrust commands (num_agents, 3)
            
        Returns:
            StepResult with new state, observations, rewards, done flags, and info
        """
        # Clip actions to valid range
        actions = jnp.clip(actions, -self.dynamics.max_thrust, self.dynamics.max_thrust)
        
        # Step dynamics
        new_state = self.dynamics.step_swarm(state, actions)
        
        # Compute observations
        obs = self._get_observations(new_state)
        
        # Compute rewards
        reward, reward_info = self._compute_reward(state, new_state, actions)
        
        # Check termination
        done = self._check_done(new_state)
        
        info = {
            "step": new_state.time / self.config.dt,
            **reward_info,
        }
        
        return StepResult(new_state, obs, reward, done, info)
    
    def _get_observations(self, state: SwarmState) -> Observation:
        """Compute observations for all agents.
        
        Each agent observes:
        - Its own state (position, velocity, energy)
        - Relative positions and velocities of K nearest neighbors
        """
        num_agents = self.config.num_agents
        k_neighbors = min(5, num_agents - 1)  # 5 nearest neighbors
        
        # Own state
        own_state = jnp.concatenate([
            state.pos,
            state.vel,
            state.energy[:, None],
        ], axis=-1)  # (N, 7)
        
        # Compute pairwise distances
        pos = state.pos  # (N, 3)
        diff = pos[:, None, :] - pos[None, :, :]  # (N, N, 3)
        dists = jnp.linalg.norm(diff, axis=-1)  # (N, N)
        
        # Mask self-distances
        dists = dists + jnp.eye(num_agents) * 1e10
        
        # Get K nearest neighbors
        neighbor_indices = jnp.argsort(dists, axis=-1)[:, :k_neighbors]  # (N, K)
        
        # Gather relative positions and velocities
        def gather_neighbors(agent_idx, neighbor_idx):
            rel_pos = state.pos[neighbor_idx] - state.pos[agent_idx]
            rel_vel = state.vel[neighbor_idx] - state.vel[agent_idx]
            return rel_pos, rel_vel
        
        # Vectorize over agents and neighbors
        agent_indices = jnp.arange(num_agents)[:, None]  # (N, 1)
        agent_indices = jnp.broadcast_to(agent_indices, (num_agents, k_neighbors))
        
        rel_pos = state.pos[neighbor_indices] - state.pos[:, None, :]  # (N, K, 3)
        rel_vel = state.vel[neighbor_indices] - state.vel[:, None, :]  # (N, K, 3)
        
        # All neighbors are valid (mask = True)
        neighbor_mask = jnp.ones((num_agents, k_neighbors), dtype=bool)
        
        return Observation(
            own_state=own_state,
            relative_positions=rel_pos,
            relative_velocities=rel_vel,
            neighbor_mask=neighbor_mask,
        )
    
    def _compute_reward(
        self,
        state: SwarmState,
        new_state: SwarmState,
        actions: Array,
    ) -> tuple[Array, dict[str, Array]]:
        """Compute reward for all agents.
        
        Reward components:
        - Coverage: Reward for spreading out (maximizing pairwise distances)
        - Collision: Penalty for getting too close to other agents
        - Energy: Small penalty for energy expenditure
        - Boundary: Penalty for leaving arena
        """
        cfg = self.config
        num_agents = cfg.num_agents
        
        # Coverage reward: mean distance to nearest neighbor (normalized)
        pos = new_state.pos
        diff = pos[:, None, :] - pos[None, :, :]
        dists = jnp.linalg.norm(diff, axis=-1) + jnp.eye(num_agents) * 1e10
        min_dists = jnp.min(dists, axis=-1)
        coverage_reward = jnp.tanh(min_dists / cfg.min_separation - 1.0)  # (N,)
        
        # Collision penalty
        collision_mask = min_dists < cfg.collision_radius
        collision_penalty = collision_mask.astype(jnp.float32)  # (N,)
        
        # Energy penalty
        thrust_magnitude = jnp.linalg.norm(actions, axis=-1)
        energy_penalty = thrust_magnitude / self.dynamics.max_thrust  # (N,)
        
        # Boundary penalty
        xy_dist = jnp.linalg.norm(pos[:, :2], axis=-1)
        boundary_violation = xy_dist > cfg.arena_size / 2
        boundary_penalty = boundary_violation.astype(jnp.float32)  # (N,)
        
        # Total reward
        reward = (
            cfg.coverage_weight * coverage_reward
            + cfg.collision_weight * collision_penalty
            + cfg.energy_weight * (-energy_penalty)
            + cfg.boundary_weight * boundary_penalty
        )
        
        info = {
            "coverage_reward": coverage_reward.mean(),
            "collision_penalty": collision_penalty.mean(),
            "energy_penalty": energy_penalty.mean(),
            "boundary_penalty": boundary_penalty.mean(),
        }
        
        return reward, info
    
    def _check_done(self, state: SwarmState) -> Array:
        """Check if episode should terminate.
        
        Termination conditions:
        - Maximum steps reached
        - All agents out of energy
        - All agents out of bounds
        """
        cfg = self.config
        
        # Time limit
        time_done = state.time >= cfg.max_steps * cfg.dt
        
        # Energy depletion (all agents)
        energy_done = jnp.all(state.energy < 0.01)
        
        # Out of bounds (all agents)
        xy_dist = jnp.linalg.norm(state.pos[:, :2], axis=-1)
        oob_done = jnp.all(xy_dist > cfg.arena_size)
        
        return time_done | energy_done | oob_done


# Factory function for creating jitted environment
def make_env(config: EnvConfig | None = None) -> SwarmEnv:
    """Create a swarm environment instance."""
    return SwarmEnv(config)


@partial(jax.jit, static_argnums=(0,))
def env_step(env: SwarmEnv, state: SwarmState, actions: Array) -> StepResult:
    """JIT-compiled environment step."""
    return env.step(state, actions)


@partial(jax.jit, static_argnums=(0,))
def env_reset(env: SwarmEnv, key: Array) -> tuple[SwarmState, Observation]:
    """JIT-compiled environment reset."""
    return env.reset(key)
