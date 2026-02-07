"""Control modules for PPO and MPC."""

from swarm.control.rl_policy import (
    PPOConfig,
    PPOAgent,
    ActorCritic,
    Trajectory,
    compute_gae,
    ppo_loss,
    create_ppo_agent,
)
from swarm.control.costs import (
    position_tracking_cost,
    velocity_tracking_cost,
    control_effort_cost,
    collision_avoidance_cost,
    boundary_cost,
    total_mpc_cost,
)
from swarm.control.mpc import (
    MPCConfig,
    MPCResult,
    DifferentiableMPC,
    RecedingHorizonMPC,
)
from swarm.control.hybrid import (
    HybridConfig,
    HybridController,
    AdaptiveHybridController,
    create_hybrid_controller,
)

__all__ = [
    "PPOConfig",
    "PPOAgent",
    "ActorCritic",
    "Trajectory",
    "compute_gae",
    "ppo_loss",
    "create_ppo_agent",
    "position_tracking_cost",
    "velocity_tracking_cost",
    "control_effort_cost",
    "collision_avoidance_cost",
    "boundary_cost",
    "total_mpc_cost",
    "MPCConfig",
    "MPCResult",
    "DifferentiableMPC",
    "RecedingHorizonMPC",
    "HybridConfig",
    "HybridController",
    "AdaptiveHybridController",
    "create_hybrid_controller",
]
