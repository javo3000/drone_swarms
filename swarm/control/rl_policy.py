"""PPO Actor-Critic for swarm control.

Implements Proximal Policy Optimization (PPO) with:
- Separate actor and critic networks
- GAE advantage estimation
- Clipped surrogate objective
- Value function clipping
"""

from __future__ import annotations

from typing import NamedTuple, Sequence, Callable

import jax
import jax.numpy as jnp
from jax import Array
import flax.linen as nn
import optax


class PPOConfig(NamedTuple):
    """PPO hyperparameters."""
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    num_minibatches: int = 4


class Trajectory(NamedTuple):
    """Collected trajectory data."""
    obs: Array          # (T, N, obs_dim)
    actions: Array      # (T, N, act_dim)
    log_probs: Array    # (T, N)
    values: Array       # (T, N)
    rewards: Array      # (T, N)
    dones: Array        # (T,)
    

class ActorCritic(nn.Module):
    """Combined actor-critic network.
    
    Actor: Outputs mean and log_std for Gaussian policy
    Critic: Outputs state value estimate
    
    Args:
        action_dim: Dimension of action space
        hidden_dims: Hidden layer dimensions
    """
    
    action_dim: int
    hidden_dims: Sequence[int] = (256, 256)
    
    def setup(self):
        self.log_std = self.param(
            'log_std',
            nn.initializers.zeros,
            (self.action_dim,)
        )
    
    @nn.compact
    def __call__(self, obs: Array) -> tuple[Array, Array, Array]:
        """Forward pass.
        
        Args:
            obs: Observations (batch, obs_dim) or (obs_dim,)
            
        Returns:
            Tuple of (action_mean, log_std, value)
        """
        # Shared backbone
        x = obs
        for dim in self.hidden_dims:
            x = nn.Dense(dim, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(x)
            x = nn.LayerNorm()(x)
            x = nn.tanh(x)
        
        # Actor head
        action_mean = nn.Dense(
            self.action_dim,
            kernel_init=nn.initializers.orthogonal(0.01)
        )(x)
        
        # Critic head (separate layer)
        value = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))(x)
        value = value.squeeze(-1)
        
        return action_mean, self.log_std, value
    
    def get_action(
        self,
        params: dict,
        obs: Array,
        key: Array,
        deterministic: bool = False,
    ) -> tuple[Array, Array]:
        """Sample action from policy.
        
        Args:
            params: Network parameters
            obs: Observation
            key: PRNG key
            deterministic: If True, return mean action
            
        Returns:
            Tuple of (action, log_prob)
        """
        mean, log_std, _ = self.apply(params, obs)
        std = jnp.exp(log_std)
        
        if deterministic:
            action = mean
        else:
            action = mean + std * jax.random.normal(key, mean.shape)
        
        log_prob = self._log_prob(action, mean, log_std)
        return action, log_prob
    
    def get_value(self, params: dict, obs: Array) -> Array:
        """Get value estimate.
        
        Args:
            params: Network parameters
            obs: Observation
            
        Returns:
            Value estimate
        """
        _, _, value = self.apply(params, obs)
        return value
    
    def evaluate_actions(
        self,
        params: dict,
        obs: Array,
        actions: Array,
    ) -> tuple[Array, Array, Array]:
        """Evaluate actions for PPO update.
        
        Args:
            params: Network parameters
            obs: Observations
            actions: Actions taken
            
        Returns:
            Tuple of (log_probs, entropy, values)
        """
        mean, log_std, value = self.apply(params, obs)
        log_prob = self._log_prob(actions, mean, log_std)
        entropy = self._entropy(log_std)
        return log_prob, entropy, value
    
    @staticmethod
    def _log_prob(action: Array, mean: Array, log_std: Array) -> Array:
        """Compute log probability of action under Gaussian."""
        std = jnp.exp(log_std)
        log_prob = -0.5 * (
            jnp.sum(((action - mean) / std) ** 2, axis=-1)
            + jnp.sum(2 * log_std)
            + mean.shape[-1] * jnp.log(2 * jnp.pi)
        )
        return log_prob
    
    @staticmethod
    def _entropy(log_std: Array) -> Array:
        """Compute entropy of Gaussian policy."""
        return 0.5 * jnp.sum(log_std + 0.5 * jnp.log(2 * jnp.pi * jnp.e))


def compute_gae(
    rewards: Array,
    values: Array,
    dones: Array,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[Array, Array]:
    """Compute Generalized Advantage Estimation.
    
    Args:
        rewards: Rewards (T, N)
        values: Value estimates (T+1, N) - includes bootstrap value
        dones: Done flags (T,)
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        
    Returns:
        Tuple of (advantages, returns)
    """
    T = rewards.shape[0]
    advantages = jnp.zeros_like(rewards)
    last_gae = 0.0
    
    for t in reversed(range(T)):
        done_mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * values[t + 1] * done_mask - values[t]
        advantages = advantages.at[t].set(delta + gamma * gae_lambda * done_mask * last_gae)
        last_gae = advantages[t]
    
    returns = advantages + values[:-1]
    return advantages, returns


def ppo_loss(
    params: dict,
    actor_critic: ActorCritic,
    obs: Array,
    actions: Array,
    old_log_probs: Array,
    advantages: Array,
    returns: Array,
    config: PPOConfig,
) -> tuple[Array, dict]:
    """Compute PPO loss.
    
    Args:
        params: Network parameters
        actor_critic: ActorCritic module
        obs: Observations (B, obs_dim)
        actions: Actions (B, act_dim)
        old_log_probs: Old log probabilities (B,)
        advantages: Advantages (B,)
        returns: Returns (B,)
        config: PPO configuration
        
    Returns:
        Tuple of (loss, info_dict)
    """
    log_probs, entropy, values = actor_critic.evaluate_actions(params, obs, actions)
    
    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # Policy loss (clipped surrogate)
    ratio = jnp.exp(log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = jnp.clip(ratio, 1 - config.clip_epsilon, 1 + config.clip_epsilon) * advantages
    policy_loss = -jnp.mean(jnp.minimum(surr1, surr2))
    
    # Value loss (clipped)
    value_loss = 0.5 * jnp.mean((values - returns) ** 2)
    
    # Entropy bonus
    entropy_loss = -jnp.mean(entropy)
    
    # Total loss
    loss = policy_loss + config.vf_coef * value_loss + config.ent_coef * entropy_loss
    
    info = {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": -entropy_loss,
        "approx_kl": jnp.mean((ratio - 1) - jnp.log(ratio)),
        "clip_fraction": jnp.mean(jnp.abs(ratio - 1) > config.clip_epsilon),
    }
    
    return loss, info


class PPOAgent:
    """PPO agent for swarm control.
    
    Args:
        obs_dim: Observation dimension
        action_dim: Action dimension
        config: PPO configuration
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        config: PPOConfig | None = None,
    ):
        self.config = config or PPOConfig()
        self.actor_critic = ActorCritic(action_dim=action_dim)
        
        # Initialize
        dummy_obs = jnp.zeros((1, obs_dim))
        self.params = self.actor_critic.init(jax.random.PRNGKey(0), dummy_obs)
        
        # Optimizer with gradient clipping
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(self.config.max_grad_norm),
            optax.adam(self.config.learning_rate),
        )
        self.opt_state = self.optimizer.init(self.params)
    
    def get_action(
        self,
        obs: Array,
        key: Array,
        deterministic: bool = False,
    ) -> tuple[Array, Array, Array]:
        """Sample action for all agents.
        
        Args:
            obs: Observations (N, obs_dim)
            key: PRNG key
            deterministic: Use mean action
            
        Returns:
            Tuple of (actions, log_probs, values)
        """
        mean, log_std, value = self.actor_critic.apply(self.params, obs)
        std = jnp.exp(log_std)
        
        if deterministic:
            action = mean
        else:
            action = mean + std * jax.random.normal(key, mean.shape)
        
        log_prob = self.actor_critic._log_prob(action, mean, log_std)
        return action, log_prob, value
    
    def update(self, trajectory: Trajectory) -> dict:
        """Update policy from collected trajectory.
        
        Args:
            trajectory: Collected trajectory data
            
        Returns:
            Dictionary of training metrics
        """
        T, N, _ = trajectory.obs.shape
        
        # Compute advantages
        bootstrap_value = self.actor_critic.get_value(self.params, trajectory.obs[-1])
        values_with_bootstrap = jnp.concatenate([
            trajectory.values,
            bootstrap_value[None, :]
        ], axis=0)
        
        advantages, returns = compute_gae(
            trajectory.rewards,
            values_with_bootstrap,
            trajectory.dones,
            self.config.gamma,
            self.config.gae_lambda,
        )
        
        # Flatten for batching
        obs_flat = trajectory.obs.reshape(-1, trajectory.obs.shape[-1])
        actions_flat = trajectory.actions.reshape(-1, trajectory.actions.shape[-1])
        log_probs_flat = trajectory.log_probs.reshape(-1)
        advantages_flat = advantages.reshape(-1)
        returns_flat = returns.reshape(-1)
        
        # PPO update epochs
        all_metrics = []
        batch_size = len(obs_flat)
        minibatch_size = batch_size // self.config.num_minibatches
        
        for _ in range(self.config.update_epochs):
            # Shuffle
            perm = jax.random.permutation(jax.random.PRNGKey(0), batch_size)
            
            for i in range(self.config.num_minibatches):
                idx = perm[i * minibatch_size : (i + 1) * minibatch_size]
                
                # Compute gradients
                (loss, info), grads = jax.value_and_grad(ppo_loss, has_aux=True)(
                    self.params,
                    self.actor_critic,
                    obs_flat[idx],
                    actions_flat[idx],
                    log_probs_flat[idx],
                    advantages_flat[idx],
                    returns_flat[idx],
                    self.config,
                )
                
                # Update parameters
                updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
                self.params = optax.apply_updates(self.params, updates)
                
                all_metrics.append(info)
        
        # Average metrics
        return jax.tree.map(lambda *xs: jnp.mean(jnp.stack(xs)), *all_metrics)


def create_ppo_agent(
    obs_dim: int,
    action_dim: int,
    config: PPOConfig | None = None,
) -> PPOAgent:
    """Factory function to create PPO agent."""
    return PPOAgent(obs_dim, action_dim, config)
