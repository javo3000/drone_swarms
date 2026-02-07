"""Run trained policy and visualize results.

Loads a checkpoint and runs the trained PPO agent.
"""

import argparse
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches

from swarm.envs.mjx_env import SwarmEnv, EnvConfig
from swarm.control.rl_policy import ActorCritic


def load_checkpoint(checkpoint_path: str) -> dict:
    """Load trained checkpoint."""
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)
    print(f"Loaded checkpoint from {checkpoint_path}")
    return checkpoint


def run_trained_policy(
    checkpoint_path: str,
    num_agents: int = 10,
    num_steps: int = 300,
    arena_size: float = 100.0,
    seed: int = 42,
    save_path: str | None = None,
):
    """Run trained policy and visualize.
    
    Args:
        checkpoint_path: Path to .pkl checkpoint
        num_agents: Number of agents
        num_steps: Simulation steps
        arena_size: Arena size
        seed: Random seed
        save_path: Path to save animation (optional)
    """
    print(f"🐝 Running Trained Policy - {num_agents} agents, {num_steps} steps")
    print("=" * 50)
    
    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_path)
    params = checkpoint["params"]
    
    # Create environment
    config = EnvConfig(
        num_agents=num_agents,
        arena_size=arena_size,
        max_steps=num_steps,
    )
    env = SwarmEnv(config)
    
    # Create actor-critic with same architecture
    # Compute observation dimension
    obs_dim = 7 + 5 * 3 + 5 * 3  # own_state + 5 neighbors * (rel_pos + rel_vel)
    action_dim = 3
    actor_critic = ActorCritic(action_dim=action_dim)
    
    # Initialize
    key = jax.random.PRNGKey(seed)
    key, reset_key = jax.random.split(key)
    state, obs = env.reset(reset_key)
    
    # Flatten observation helper
    def flatten_obs(obs):
        neighbor_flat = obs.relative_positions.reshape(obs.relative_positions.shape[0], -1)
        neighbor_vel_flat = obs.relative_velocities.reshape(obs.relative_velocities.shape[0], -1)
        return jnp.concatenate([obs.own_state, neighbor_flat, neighbor_vel_flat], axis=-1)
    
    # Collect trajectory with trained policy
    trajectory = [state]
    rewards_history = []
    
    print("Running with trained policy...")
    for step in range(num_steps):
        key, action_key = jax.random.split(key)
        
        # Get observation
        obs = env._get_observations(state)
        obs_flat = flatten_obs(obs)
        
        # Get action from trained policy
        mean, log_std, value = actor_critic.apply(params, obs_flat)
        
        # Use mean action (deterministic) for visualization
        action = mean
        
        # Clip actions to reasonable range
        action = jnp.clip(action, -20, 20)
        
        # Step environment
        result = env.step(state, action)
        state = result.state
        trajectory.append(state)
        rewards_history.append(float(result.reward.mean()))
        
        if step % 50 == 0:
            print(f"  Step {step}: mean reward = {result.reward.mean():.3f}")
    
    print(f"\nSimulation complete! Final time: {float(state.time):.2f}s")
    print(f"Mean reward over episode: {sum(rewards_history)/len(rewards_history):.3f}")
    
    # Visualize
    print("\nCreating visualization...")
    _animate_trained(trajectory, arena_size, rewards_history, save_path)
    
    return trajectory, rewards_history


def _animate_trained(trajectory, arena_size, rewards, save_path=None):
    """Animate the trained policy trajectory."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax_swarm, ax_reward = axes
    
    fig.suptitle("Trained Swarm Policy", fontsize=14, fontweight='bold')
    
    # Initialize swarm plot
    pos = trajectory[0].pos
    scatter = ax_swarm.scatter(
        pos[:, 0], pos[:, 1],
        c=trajectory[0].energy,
        cmap='RdYlGn',
        s=120,
        edgecolors='black',
        linewidth=1.5,
        vmin=0, vmax=1,
    )
    
    # Add trails (will update)
    trail_lines = []
    for i in range(pos.shape[0]):
        line, = ax_swarm.plot([], [], 'b-', alpha=0.3, linewidth=1)
        trail_lines.append(line)
    
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
    ax_swarm.set_title('Agent Positions (Trained Policy)')
    ax_swarm.grid(True, alpha=0.3)
    
    time_text = ax_swarm.text(
        0.02, 0.98, '', transform=ax_swarm.transAxes,
        va='top', fontsize=11, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    )
    
    # Initialize reward plot
    reward_line, = ax_reward.plot([], [], 'b-', linewidth=2)
    ax_reward.set_xlim(0, len(trajectory))
    if rewards:
        ax_reward.set_ylim(min(rewards) - 0.5, max(max(rewards), 0.5) + 0.5)
    else:
        ax_reward.set_ylim(-1, 1)
    ax_reward.set_xlabel('Step')
    ax_reward.set_ylabel('Mean Reward')
    ax_reward.set_title('Reward Over Time')
    ax_reward.grid(True, alpha=0.3)
    ax_reward.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.colorbar(scatter, ax=ax_swarm, label='Energy', shrink=0.8)
    
    # Store history for trails
    trail_length = 30
    history = {i: [] for i in range(pos.shape[0])}
    
    def update(frame):
        state = trajectory[frame]
        pos = state.pos
        
        # Update scatter
        scatter.set_offsets(pos[:, :2])
        scatter.set_array(state.energy)
        
        # Update trails
        for i in range(pos.shape[0]):
            history[i].append(pos[i, :2])
            if len(history[i]) > trail_length:
                history[i] = history[i][-trail_length:]
            
            if len(history[i]) > 1:
                trail_data = jnp.array(history[i])
                trail_lines[i].set_data(trail_data[:, 0], trail_data[:, 1])
        
        time_text.set_text(f'Time: {float(state.time):.2f}s | Agents: {pos.shape[0]}')
        
        # Update reward line
        if frame > 0:
            reward_line.set_data(range(frame), rewards[:frame])
        
        return [scatter, time_text, reward_line] + trail_lines
    
    anim = FuncAnimation(
        fig, update,
        frames=len(trajectory),
        interval=50,
        blit=True
    )
    
    if save_path:
        print(f"Saving animation to {save_path}...")
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        anim.save(save_path, writer='pillow', fps=20)
        print("Done!")
    else:
        plt.tight_layout()
        plt.show()
    
    return anim


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Trained Swarm Policy")
    parser.add_argument("--checkpoint", type=str, 
                        default="logs/quick_test/final_checkpoint.pkl",
                        help="Path to checkpoint file")
    parser.add_argument("--agents", type=int, default=10, help="Number of agents")
    parser.add_argument("--steps", type=int, default=300, help="Number of steps")
    parser.add_argument("--arena", type=float, default=100.0, help="Arena size")
    parser.add_argument("--save", type=str, default=None, help="Save animation path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    run_trained_policy(
        checkpoint_path=args.checkpoint,
        num_agents=args.agents,
        num_steps=args.steps,
        arena_size=args.arena,
        seed=args.seed,
        save_path=args.save,
    )
