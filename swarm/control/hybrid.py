"""Hybrid RL-MPC controller.

Combines high-level RL policy with low-level MPC for:
- RL: Long-horizon planning, exploration, coordination
- MPC: Short-horizon tracking, constraint satisfaction, safety
"""

from __future__ import annotations

from typing import NamedTuple
from functools import partial

import jax
import jax.numpy as jnp
from jax import Array

from swarm.envs.dynamics import SwarmState, PointMassDynamics
from swarm.control.mpc import DifferentiableMPC, MPCConfig, RecedingHorizonMPC
from swarm.control.rl_policy import PPOAgent


class HybridConfig(NamedTuple):
    """Hybrid controller configuration."""
    # Blending parameters
    rl_weight: float = 0.7          # Weight for RL output (vs MPC)
    use_rl_as_reference: bool = True  # Use RL output as MPC reference
    
    # MPC settings
    mpc_horizon: int = 10
    mpc_iterations: int = 3
    
    # Safety override
    safety_override: bool = True     # Let MPC override for safety
    collision_threshold: float = 5.0  # Distance threshold for override


class HybridController:
    """Hybrid RL-MPC controller.
    
    Three modes of operation:
    1. RL-as-reference: RL outputs target positions, MPC tracks them
    2. Blended: Weighted combination of RL and MPC actions  
    3. Hierarchical: RL outputs high-level goals, MPC executes
    
    Args:
        rl_agent: Trained PPO agent
        dynamics: Physics model
        config: Hybrid configuration
    """
    
    def __init__(
        self,
        rl_agent: PPOAgent,
        dynamics: PointMassDynamics | None = None,
        config: HybridConfig | None = None,
    ):
        self.rl_agent = rl_agent
        self.config = config or HybridConfig()
        
        # Create MPC
        mpc_config = MPCConfig(
            horizon=self.config.mpc_horizon,
            num_iterations=self.config.mpc_iterations,
        )
        self.mpc = RecedingHorizonMPC(dynamics, mpc_config)
    
    def get_action(
        self,
        obs: Array,
        state: SwarmState,
        key: Array,
    ) -> tuple[Array, dict]:
        """Get hybrid action.
        
        Args:
            obs: Flattened observations for RL
            state: Full swarm state for MPC
            key: PRNG key
            
        Returns:
            Tuple of (action, info_dict)
        """
        # Get RL action
        rl_action, log_prob, value = self.rl_agent.get_action(obs, key)
        
        if self.config.use_rl_as_reference:
            # Mode 1: RL outputs velocity targets, MPC tracks
            # Interpret RL action as desired velocity direction
            target_velocities = rl_action * 5.0  # Scale factor
            
            # Compute target positions (current + velocity * dt)
            dt = 0.02 * self.config.mpc_horizon
            target_positions = state.pos + target_velocities * dt
            
            # MPC tracks the target
            mpc_action, mpc_result = self.mpc.step(state, target_positions)
            
            final_action = mpc_action
            mode = "rl_as_reference"
            
        else:
            # Mode 2: Weighted blend of RL and MPC
            # MPC tries to maintain formation / avoid collisions
            mpc_action, mpc_result = self.mpc.step(state, None)
            
            # Blend actions
            w = self.config.rl_weight
            final_action = w * rl_action + (1 - w) * mpc_action
            mode = "blended"
        
        # Safety override: let MPC take over if collision imminent
        if self.config.safety_override:
            final_action = self._apply_safety_override(
                final_action, rl_action, mpc_action, state
            )
        
        info = {
            "rl_action": rl_action,
            "mpc_action": mpc_action if 'mpc_action' in dir() else None,
            "mode": mode,
            "value": value,
            "log_prob": log_prob,
        }
        
        return final_action, info
    
    def _apply_safety_override(
        self,
        current_action: Array,
        rl_action: Array,
        mpc_action: Array,
        state: SwarmState,
    ) -> Array:
        """Apply MPC safety override when collision is imminent.
        
        Args:
            current_action: Current action choice
            rl_action: RL's action
            mpc_action: MPC's action
            state: Current state
            
        Returns:
            Potentially modified action
        """
        # Check minimum distance to neighbors
        num_agents = state.pos.shape[0]
        
        # Pairwise distances
        diff = state.pos[:, None, :] - state.pos[None, :, :]
        dists = jnp.linalg.norm(diff, axis=-1)
        dists = dists + jnp.eye(num_agents) * 1e10  # Exclude self
        
        min_dist_per_agent = jnp.min(dists, axis=-1)
        
        # Override with MPC where collision risk is high
        collision_risk = min_dist_per_agent < self.config.collision_threshold
        
        # Blend towards MPC for risky agents
        override_weight = jnp.where(collision_risk, 1.0, 0.0)[:, None]
        safe_action = override_weight * mpc_action + (1 - override_weight) * current_action
        
        return safe_action
    
    def reset(self):
        """Reset MPC state."""
        self.mpc.reset()


class AdaptiveHybridController(HybridController):
    """Hybrid controller with adaptive blending.
    
    Adjusts RL/MPC blend based on:
    - Uncertainty (high uncertainty → more MPC)
    - Energy levels (low energy → more MPC for efficiency)
    - Proximity to threats (near threat → more RL for flexibility)
    """
    
    def __init__(
        self,
        rl_agent: PPOAgent,
        dynamics: PointMassDynamics | None = None,
        config: HybridConfig | None = None,
    ):
        super().__init__(rl_agent, dynamics, config)
        self.base_rl_weight = self.config.rl_weight
    
    def get_action(
        self,
        obs: Array,
        state: SwarmState,
        key: Array,
        uncertainty: Array | None = None,
    ) -> tuple[Array, dict]:
        """Get action with adaptive blending.
        
        Args:
            obs: Observations
            state: Swarm state
            key: PRNG key
            uncertainty: Optional per-agent uncertainty estimate
            
        Returns:
            Tuple of (action, info)
        """
        # Compute adaptive weights
        if uncertainty is not None:
            # High uncertainty → reduce RL weight
            adapted_weight = self.base_rl_weight * (1 - 0.5 * uncertainty)
        else:
            adapted_weight = self.base_rl_weight
        
        # Adjust for energy
        energy_factor = state.energy  # 0 to 1
        adapted_weight = adapted_weight * (0.5 + 0.5 * energy_factor)
        
        # Temporarily update config
        original_weight = self.config.rl_weight
        self.config = self.config._replace(rl_weight=float(adapted_weight.mean()))
        
        # Get action with adapted weights
        action, info = super().get_action(obs, state, key)
        
        # Restore config
        self.config = self.config._replace(rl_weight=original_weight)
        
        info["adapted_rl_weight"] = adapted_weight
        return action, info


def create_hybrid_controller(
    rl_agent: PPOAgent,
    mode: str = "reference",
) -> HybridController:
    """Factory function for hybrid controllers.
    
    Args:
        rl_agent: Trained RL agent
        mode: "reference", "blended", or "adaptive"
        
    Returns:
        Configured hybrid controller
    """
    if mode == "reference":
        config = HybridConfig(use_rl_as_reference=True)
        return HybridController(rl_agent, config=config)
    elif mode == "blended":
        config = HybridConfig(use_rl_as_reference=False, rl_weight=0.6)
        return HybridController(rl_agent, config=config)
    elif mode == "adaptive":
        config = HybridConfig(use_rl_as_reference=False)
        return AdaptiveHybridController(rl_agent, config=config)
    else:
        raise ValueError(f"Unknown mode: {mode}")
