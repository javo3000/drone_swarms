"""Differentiable Model Predictive Control for trajectory optimization.

Implements shooting-based MPC that can be differentiated through for
gradient-based optimization and integration with RL.
"""

from __future__ import annotations

from typing import Callable, NamedTuple
from functools import partial

import jax
import jax.numpy as jnp
from jax import Array

from swarm.envs.dynamics import PointMassDynamics, SwarmState, DroneState
from swarm.control.costs import total_mpc_cost


class MPCConfig(NamedTuple):
    """MPC configuration."""
    horizon: int = 10
    num_iterations: int = 5
    learning_rate: float = 0.1
    control_min: float = -20.0
    control_max: float = 20.0


class MPCResult(NamedTuple):
    """MPC optimization result."""
    actions: Array          # Optimal action sequence (H, N, 3)
    predicted_states: list  # Predicted state trajectory
    costs: Array           # Cost at each iteration
    success: bool          # Whether optimization converged


class DifferentiableMPC:
    """Differentiable MPC using gradient descent on action sequences.
    
    Uses shooting method: optimize over action sequences directly,
    rolling out dynamics to compute costs.
    
    Args:
        dynamics: Point-mass dynamics model
        config: MPC configuration
        cost_fn: Cost function (state, action) -> cost
    """
    
    def __init__(
        self,
        dynamics: PointMassDynamics | None = None,
        config: MPCConfig | None = None,
    ):
        self.dynamics = dynamics or PointMassDynamics()
        self.config = config or MPCConfig()
    
    def solve(
        self,
        initial_state: SwarmState,
        target_positions: Array | None = None,
        warm_start: Array | None = None,
    ) -> MPCResult:
        """Solve MPC optimization problem.
        
        Args:
            initial_state: Current swarm state
            target_positions: Target positions for tracking (optional)
            warm_start: Initial action sequence guess (optional)
            
        Returns:
            MPCResult with optimal actions and diagnostics
        """
        num_agents = initial_state.pos.shape[0]
        H = self.config.horizon
        
        # Initialize action sequence
        if warm_start is not None:
            actions = warm_start
        else:
            # Initialize with hover thrust
            hover_thrust = self.dynamics.mass * 9.81
            actions = jnp.zeros((H, num_agents, 3))
            actions = actions.at[:, :, 2].set(hover_thrust)
        
        # Gradient descent optimization
        costs = []
        
        for i in range(self.config.num_iterations):
            # Compute cost and gradients
            cost, grads = jax.value_and_grad(self._trajectory_cost)(
                actions, initial_state, target_positions
            )
            costs.append(cost)
            
            # Update actions
            actions = actions - self.config.learning_rate * grads
            
            # Clip to control limits
            actions = jnp.clip(
                actions,
                self.config.control_min,
                self.config.control_max
            )
        
        # Get predicted trajectory with final actions
        predicted_states = self._rollout(actions, initial_state)
        
        return MPCResult(
            actions=actions,
            predicted_states=predicted_states,
            costs=jnp.array(costs),
            success=True,
        )
    
    def _trajectory_cost(
        self,
        actions: Array,
        initial_state: SwarmState,
        target_positions: Array | None,
    ) -> Array:
        """Compute total cost over trajectory.
        
        Args:
            actions: Action sequence (H, N, 3)
            initial_state: Starting state
            target_positions: Target positions
            
        Returns:
            Total scalar cost
        """
        H = actions.shape[0]
        state = initial_state
        total_cost = 0.0
        
        for t in range(H):
            action_t = actions[t]
            
            # Step dynamics
            state = self.dynamics.step_swarm(state, action_t)
            
            # Accumulate cost
            step_cost = total_mpc_cost(
                state, action_t,
                target_positions=target_positions,
            )
            total_cost = total_cost + step_cost.sum()
        
        return total_cost
    
    def _rollout(
        self,
        actions: Array,
        initial_state: SwarmState,
    ) -> list[SwarmState]:
        """Roll out trajectory with given actions.
        
        Args:
            actions: Action sequence (H, N, 3)
            initial_state: Starting state
            
        Returns:
            List of states [s0, s1, ..., sH]
        """
        states = [initial_state]
        state = initial_state
        
        for t in range(actions.shape[0]):
            state = self.dynamics.step_swarm(state, actions[t])
            states.append(state)
        
        return states
    
    def get_action(
        self,
        state: SwarmState,
        target_positions: Array | None = None,
    ) -> Array:
        """Get single-step action from MPC.
        
        Solves MPC and returns first action in sequence.
        
        Args:
            state: Current state
            target_positions: Target positions
            
        Returns:
            Action for current timestep (N, 3)
        """
        result = self.solve(state, target_positions)
        return result.actions[0]


class RecedingHorizonMPC:
    """Receding horizon MPC with warm-starting.
    
    Maintains action sequence buffer, shifts on each step,
    and warm-starts next optimization.
    """
    
    def __init__(
        self,
        dynamics: PointMassDynamics | None = None,
        config: MPCConfig | None = None,
    ):
        self.mpc = DifferentiableMPC(dynamics, config)
        self.config = config or MPCConfig()
        self._action_buffer: Array | None = None
    
    def step(
        self,
        state: SwarmState,
        target_positions: Array | None = None,
    ) -> tuple[Array, MPCResult]:
        """Execute one MPC step.
        
        Args:
            state: Current state
            target_positions: Target positions
            
        Returns:
            Tuple of (action, mpc_result)
        """
        # Warm start from shifted previous solution
        warm_start = None
        if self._action_buffer is not None:
            # Shift buffer: drop first, append zero
            warm_start = jnp.concatenate([
                self._action_buffer[1:],
                jnp.zeros((1,) + self._action_buffer.shape[1:])
            ], axis=0)
        
        # Solve MPC
        result = self.mpc.solve(state, target_positions, warm_start)
        
        # Store for warm-starting
        self._action_buffer = result.actions
        
        # Return first action
        return result.actions[0], result
    
    def reset(self):
        """Reset action buffer."""
        self._action_buffer = None


# JIT-compiled single-step MPC
@partial(jax.jit, static_argnums=(0,))
def mpc_step_jit(
    mpc: DifferentiableMPC,
    state: SwarmState,
    target_positions: Array,
) -> Array:
    """JIT-compiled MPC step."""
    return mpc.get_action(state, target_positions)
