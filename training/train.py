"""Main training loop for swarm PPO."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import yaml

from swarm.envs.mjx_env import SwarmEnv, EnvConfig
from swarm.control.rl_policy import PPOAgent, PPOConfig, create_ppo_agent
from swarm.utils.logging import MetricsLogger, load_config
from training.rollout import RolloutManager, merge_trajectories


def train(
    env_config_path: str = "configs/env/10_agents.yaml",
    train_config_path: str = "configs/training/ppo.yaml",
    total_steps: int = 1_000_000,
    log_dir: str = "logs",
    seed: int = 42,
):
    """Main training loop.
    
    Args:
        env_config_path: Path to environment config
        train_config_path: Path to training config
        total_steps: Total environment steps
        log_dir: Directory for logs and checkpoints
        seed: Random seed
    """
    print("=" * 60)
    print("🐝 Drone Swarms - PPO Training")
    print("=" * 60)
    
    # Load configs
    env_cfg = load_config(env_config_path)
    train_cfg = load_config(train_config_path)
    
    print(f"Environment: {env_cfg['env']['num_agents']} agents")
    print(f"Training steps: {total_steps:,}")
    
    # Create environment config
    env_config = EnvConfig(
        num_agents=env_cfg['env']['num_agents'],
        arena_size=env_cfg['env']['arena_size'],
        max_steps=env_cfg['env']['max_steps'],
        dt=env_cfg['env']['dt'],
    )
    
    # Create PPO config
    ppo_config = PPOConfig(
        learning_rate=train_cfg['ppo']['learning_rate'],
        gamma=train_cfg['ppo']['gamma'],
        gae_lambda=train_cfg['ppo']['gae_lambda'],
        clip_epsilon=train_cfg['ppo']['clip_epsilon'],
        vf_coef=train_cfg['ppo']['vf_coef'],
        ent_coef=train_cfg['ppo']['ent_coef'],
        max_grad_norm=train_cfg['ppo']['max_grad_norm'],
        update_epochs=train_cfg['ppo']['update_epochs'],
        num_minibatches=train_cfg['ppo']['num_minibatches'],
    )
    
    # Setup
    key = jax.random.PRNGKey(seed)
    num_envs = train_cfg['ppo']['num_envs']
    num_steps = train_cfg['ppo']['num_steps']
    
    # Create rollout manager
    rollout_manager = RolloutManager(env_config, num_envs, num_steps)
    obs_dim = rollout_manager.obs_dim
    action_dim = 3  # 3D thrust
    
    # Create agent
    agent = create_ppo_agent(obs_dim, action_dim, ppo_config)
    
    # Logger
    logger = MetricsLogger(log_dir=log_dir)
    
    # Initialize environments
    key, reset_key = jax.random.split(key)
    states = rollout_manager.reset_all(reset_key)
    
    # Training loop
    steps_per_update = num_envs * num_steps * env_config.num_agents
    num_updates = total_steps // steps_per_update
    
    print(f"\nStarting training: {num_updates} updates")
    print(f"Steps per update: {steps_per_update:,}")
    print("-" * 60)
    
    start_time = time.time()
    total_env_steps = 0
    
    for update in range(num_updates):
        # Collect rollouts
        key, rollout_key = jax.random.split(key)
        trajectories, states = rollout_manager.collect(agent, states, rollout_key)
        
        # Merge trajectories
        trajectory = merge_trajectories(trajectories)
        total_env_steps += steps_per_update
        
        # Update agent
        metrics = agent.update(trajectory)
        
        # Log
        mean_reward = float(trajectory.rewards.mean())
        metrics["mean_reward"] = mean_reward
        metrics["total_steps"] = total_env_steps
        logger.log(metrics)
        
        # Print progress
        if update % 10 == 0:
            elapsed = time.time() - start_time
            sps = total_env_steps / elapsed
            print(
                f"Update {update:4d}/{num_updates} | "
                f"Steps: {total_env_steps:8,} | "
                f"Reward: {mean_reward:7.3f} | "
                f"Policy Loss: {float(metrics['policy_loss']):7.4f} | "
                f"SPS: {sps:,.0f}"
            )
        
        # Reset envs if done
        if update % 50 == 0 and update > 0:
            key, reset_key = jax.random.split(key)
            states = rollout_manager.reset_all(reset_key)
    
    # Save final checkpoint
    checkpoint_path = Path(log_dir) / "final_checkpoint.pkl"
    import pickle
    with open(checkpoint_path, 'wb') as f:
        pickle.dump({"params": agent.params, "config": ppo_config}, f)
    print(f"\nSaved checkpoint to {checkpoint_path}")
    
    # Save logs
    logger.save()
    
    print("=" * 60)
    print("Training complete!")
    print(f"Total time: {time.time() - start_time:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train swarm PPO agent")
    parser.add_argument("--env-config", type=str, default="configs/env/10_agents.yaml")
    parser.add_argument("--train-config", type=str, default="configs/training/ppo.yaml")
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true", help="Run quick test (1k steps)")
    
    args = parser.parse_args()
    
    if args.smoke_test:
        args.steps = 1000
        print("Running smoke test with 1000 steps...")
    
    train(
        env_config_path=args.env_config,
        train_config_path=args.train_config,
        total_steps=args.steps,
        log_dir=args.log_dir,
        seed=args.seed,
    )
