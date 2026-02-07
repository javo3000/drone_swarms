"""Batched rollout collection for training."""

from __future__ import annotations

from typing import NamedTuple
from functools import partial

import jax
import jax.numpy as jnp
from jax import Array

from swarm.envs.mjx_env import SwarmEnv, EnvConfig
from swarm.envs.dynamics import SwarmState
from swarm.control.rl_policy import PPOAgent, Trajectory


class RolloutConfig(NamedTuple):
    """Rollout configuration."""
    num_envs: int = 8
    num_steps: int = 128


def collect_rollout(
    env: SwarmEnv,
    agent: PPOAgent,
    initial_state: SwarmState,
    key: Array,
    num_steps: int = 128,
) -> tuple[Trajectory, SwarmState]:
    """Collect a rollout trajectory.
    
    Args:
        env: Environment instance
        agent: PPO agent
        initial_state: Initial environment state
        key: PRNG key
        num_steps: Number of steps to collect
        
    Returns:
        Tuple of (trajectory, final_state)
    """
    def step_fn(carry, _):
        state, key = carry
        key, action_key = jax.random.split(key)
        
        # Get observations from state
        obs = env._get_observations(state)
        obs_flat = _flatten_observation(obs)
        
        # Get action from agent
        action, log_prob, value = agent.get_action(obs_flat, action_key)
        
        # Step environment
        result = env.step(state, action)
        
        transition = {
            "obs": obs_flat,
            "action": action,
            "log_prob": log_prob,
            "value": value,
            "reward": result.reward,
            "done": result.done,
        }
        
        return (result.state, key), transition
    
    # Collect trajectory
    (final_state, _), transitions = jax.lax.scan(
        step_fn,
        (initial_state, key),
        None,
        length=num_steps,
    )
    
    trajectory = Trajectory(
        obs=transitions["obs"],
        actions=transitions["action"],
        log_probs=transitions["log_prob"],
        values=transitions["value"],
        rewards=transitions["reward"],
        dones=transitions["done"],
    )
    
    return trajectory, final_state


def _flatten_observation(obs) -> Array:
    """Flatten observation named tuple to array."""
    # Concatenate own_state with flattened neighbor features
    neighbor_flat = obs.relative_positions.reshape(obs.relative_positions.shape[0], -1)
    neighbor_vel_flat = obs.relative_velocities.reshape(obs.relative_velocities.shape[0], -1)
    
    return jnp.concatenate([
        obs.own_state,
        neighbor_flat,
        neighbor_vel_flat,
    ], axis=-1)


class RolloutManager:
    """Manages rollout collection across multiple environments.
    
    Args:
        env_config: Environment configuration
        num_envs: Number of parallel environments
        num_steps: Steps per rollout
    """
    
    def __init__(
        self,
        env_config: EnvConfig,
        num_envs: int = 8,
        num_steps: int = 128,
    ):
        self.env_config = env_config
        self.num_envs = num_envs
        self.num_steps = num_steps
        
        # Create environment
        self.env = SwarmEnv(env_config)
        
        # Compute observation dimension
        obs_dim = 7 + 5 * 3 + 5 * 3  # own_state + 5 neighbors * (rel_pos + rel_vel)
        self.obs_dim = obs_dim
    
    def collect(
        self,
        agent: PPOAgent,
        states: list[SwarmState],
        key: Array,
    ) -> tuple[list[Trajectory], list[SwarmState]]:
        """Collect rollouts from multiple environments.
        
        Args:
            agent: PPO agent
            states: Initial states for each env
            key: PRNG key
            
        Returns:
            Tuple of (trajectories, final_states)
        """
        trajectories = []
        final_states = []
        
        for i, state in enumerate(states):
            key, rollout_key = jax.random.split(key)
            traj, final_state = collect_rollout(
                self.env, agent, state, rollout_key, self.num_steps
            )
            trajectories.append(traj)
            final_states.append(final_state)
        
        return trajectories, final_states
    
    def reset_all(self, key: Array) -> list[SwarmState]:
        """Reset all environments."""
        states = []
        for i in range(self.num_envs):
            key, reset_key = jax.random.split(key)
            state, _ = self.env.reset(reset_key)
            states.append(state)
        return states


def merge_trajectories(trajectories: list[Trajectory]) -> Trajectory:
    """Merge multiple trajectories into one.
    
    Args:
        trajectories: List of trajectories
        
    Returns:
        Merged trajectory with all data concatenated
    """
    return Trajectory(
        obs=jnp.concatenate([t.obs for t in trajectories], axis=0),
        actions=jnp.concatenate([t.actions for t in trajectories], axis=0),
        log_probs=jnp.concatenate([t.log_probs for t in trajectories], axis=0),
        values=jnp.concatenate([t.values for t in trajectories], axis=0),
        rewards=jnp.concatenate([t.rewards for t in trajectories], axis=0),
        dones=jnp.concatenate([t.dones for t in trajectories], axis=0),
    )
