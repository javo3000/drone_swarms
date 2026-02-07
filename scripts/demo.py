"""Demo script to visualize the swarm environment.

Run this to see the environment in action with random actions.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches

from swarm.envs.mjx_env import SwarmEnv, EnvConfig
from swarm.envs.dynamics import SwarmState


def run_demo(
    num_agents: int = 10,
    num_steps: int = 200,
    arena_size: float = 100.0,
    save_path: str | None = None,
    seed: int = 42,
):
    """Run a demo of the swarm environment with random actions.
    
    Args:
        num_agents: Number of agents
        num_steps: Number of simulation steps
        arena_size: Size of arena
        save_path: Optional path to save animation
        seed: Random seed
    """
    print(f"🐝 Drone Swarms Demo - {num_agents} agents, {num_steps} steps")
    print("=" * 50)
    
    # Create environment
    config = EnvConfig(
        num_agents=num_agents,
        arena_size=arena_size,
        max_steps=num_steps,
    )
    env = SwarmEnv(config)
    
    # Initialize
    key = jax.random.PRNGKey(seed)
    key, reset_key = jax.random.split(key)
    state, obs = env.reset(reset_key)
    
    # Collect trajectory
    trajectory = [state]
    rewards_history = []
    
    print("Running simulation...")
    for step in range(num_steps):
        # Random actions (for demo - replace with trained policy)
        key, action_key = jax.random.split(key)
        actions = jax.random.normal(action_key, (num_agents, 3)) * 5.0
        
        # Simple heuristic: move towards center if too far
        center_dir = -state.pos / jnp.maximum(jnp.linalg.norm(state.pos, axis=-1, keepdims=True), 1.0)
        actions = actions + center_dir * 2.0
        
        # Step environment
        result = env.step(state, actions)
        state = result.state
        trajectory.append(state)
        rewards_history.append(float(result.reward.mean()))
        
        if step % 50 == 0:
            print(f"  Step {step}: mean reward = {result.reward.mean():.3f}")
    
    print(f"Simulation complete! Final time: {float(state.time):.2f}s")
    
    # Visualize
    print("\nCreating visualization...")
    _animate_demo(trajectory, arena_size, rewards_history, save_path)
    
    return trajectory, rewards_history


def _animate_demo(
    trajectory: list[SwarmState],
    arena_size: float,
    rewards: list[float],
    save_path: str | None = None,
):
    """Create animated visualization of trajectory."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax_swarm, ax_reward = axes
    
    # Initialize swarm plot
    pos = trajectory[0].pos
    scatter = ax_swarm.scatter(
        pos[:, 0], pos[:, 1],
        c=trajectory[0].energy,
        cmap='RdYlGn',
        s=100,
        edgecolors='black',
        vmin=0, vmax=1,
    )
    
    # Arena boundary
    half = arena_size / 2
    rect = patches.Rectangle(
        (-half, -half), arena_size, arena_size,
        linewidth=2, edgecolor='red', facecolor='none', linestyle='--'
    )
    ax_swarm.add_patch(rect)
    
    ax_swarm.set_xlim(-half * 1.2, half * 1.2)
    ax_swarm.set_ylim(-half * 1.2, half * 1.2)
    ax_swarm.set_aspect('equal')
    ax_swarm.set_xlabel('X (m)')
    ax_swarm.set_ylabel('Y (m)')
    ax_swarm.set_title('Swarm Positions')
    ax_swarm.grid(True, alpha=0.3)
    
    time_text = ax_swarm.text(
        0.02, 0.98, '', transform=ax_swarm.transAxes,
        va='top', fontsize=12, fontweight='bold'
    )
    
    # Initialize reward plot
    reward_line, = ax_reward.plot([], [], 'b-', linewidth=2)
    ax_reward.set_xlim(0, len(trajectory))
    ax_reward.set_ylim(min(rewards) - 0.5, max(rewards) + 0.5)
    ax_reward.set_xlabel('Step')
    ax_reward.set_ylabel('Mean Reward')
    ax_reward.set_title('Reward Over Time')
    ax_reward.grid(True, alpha=0.3)
    
    plt.colorbar(scatter, ax=ax_swarm, label='Energy', shrink=0.8)
    
    def update(frame):
        state = trajectory[frame]
        pos = state.pos
        
        # Update scatter
        scatter.set_offsets(pos[:, :2])
        scatter.set_array(state.energy)
        
        time_text.set_text(f'Time: {float(state.time):.2f}s | Agents: {pos.shape[0]}')
        
        # Update reward line
        if frame > 0:
            reward_line.set_data(range(frame), rewards[:frame])
        
        return scatter, time_text, reward_line
    
    anim = FuncAnimation(
        fig, update,
        frames=len(trajectory),
        interval=50,
        blit=True
    )
    
    if save_path:
        print(f"Saving animation to {save_path}...")
        anim.save(save_path, writer='pillow', fps=20)
        print("Done!")
    else:
        plt.tight_layout()
        plt.show()
    
    return anim


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Drone Swarms Demo")
    parser.add_argument("--agents", type=int, default=10, help="Number of agents")
    parser.add_argument("--steps", type=int, default=200, help="Number of steps")
    parser.add_argument("--arena", type=float, default=100.0, help="Arena size")
    parser.add_argument("--save", type=str, default=None, help="Save animation path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    run_demo(
        num_agents=args.agents,
        num_steps=args.steps,
        arena_size=args.arena,
        save_path=args.save,
        seed=args.seed,
    )
