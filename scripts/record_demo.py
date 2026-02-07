"""Record demo videos for different scenarios.

Creates polished demo videos for:
1. Coverage task
2. Threat response
3. GPS-denied operation
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp

from swarm.envs.mjx_env import SwarmEnv, EnvConfig
from swarm.envs.tasks.threat_response import ThreatConfig, spawn_threats, update_threats
from swarm.envs.tasks.gps_denied import create_gps_denied_scenario
from eval.visualizer import create_demo_animation, SwarmVisualizer, VisualizerConfig


def record_coverage_demo(
    num_agents: int = 15,
    num_steps: int = 300,
    save_path: str = "demos/coverage_demo.gif",
):
    """Record coverage task demo.
    
    Shows swarm spreading out to cover area.
    """
    print(f"🎬 Recording Coverage Demo ({num_agents} agents)")
    print("-" * 50)
    
    config = EnvConfig(num_agents=num_agents, arena_size=100.0)
    env = SwarmEnv(config)
    
    key = jax.random.PRNGKey(42)
    state, _ = env.reset(key)
    
    trajectory = [state]
    rewards = []
    
    for step in range(num_steps):
        key, action_key = jax.random.split(key)
        
        # Random exploration actions
        actions = jax.random.normal(action_key, (num_agents, 3)) * 8.0
        
        result = env.step(state, actions)
        state = result.state
        trajectory.append(state)
        rewards.append(float(result.reward.mean()))
        
        if step % 50 == 0:
            print(f"  Step {step}: reward = {result.reward.mean():.3f}")
    
    # Create animation
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    anim = create_demo_animation(
        trajectory, rewards,
        arena_size=100.0,
        save_path=save_path,
        title="Coverage Task - Swarm Area Exploration"
    )
    
    print(f"\n✅ Saved to {save_path}")
    return anim


def record_threat_demo(
    num_agents: int = 12,
    num_steps: int = 400,
    save_path: str = "demos/threat_demo.gif",
):
    """Record threat response demo.
    
    Shows swarm detecting and intercepting threats.
    """
    print(f"🎬 Recording Threat Response Demo ({num_agents} agents)")
    print("-" * 50)
    
    config = EnvConfig(num_agents=num_agents, arena_size=100.0)
    env = SwarmEnv(config)
    
    key = jax.random.PRNGKey(123)
    state, _ = env.reset(key)
    
    # Spawn threats
    threat_config = ThreatConfig(num_threats=3, spawn_radius=80.0)
    key, threat_key = jax.random.split(key)
    threats = spawn_threats(threat_key, 3, spawn_radius=80.0)
    
    trajectory = [state]
    rewards = []
    threat_positions = [(threats.positions[i].tolist(), 15.0) for i in range(3)]
    
    for step in range(num_steps):
        key, action_key = jax.random.split(key)
        
        # Move towards nearest threat (simple heuristic)
        agent_pos = state.pos
        threat_pos = threats.positions
        
        # Find nearest threat for each agent
        dists = jnp.linalg.norm(
            agent_pos[:, None, :] - threat_pos[None, :, :],
            axis=-1
        )
        nearest = jnp.argmin(dists, axis=-1)
        
        # Action towards threat
        targets = threat_pos[nearest]
        directions = targets - agent_pos
        directions = directions / (jnp.linalg.norm(directions, axis=-1, keepdims=True) + 1e-6)
        actions = directions * 10.0 + jax.random.normal(action_key, (num_agents, 3)) * 2.0
        
        result = env.step(state, actions)
        state = result.state
        
        # Update threats (they move too)
        threats = threats._replace(
            positions=threats.positions + threats.velocities * 0.02
        )
        
        trajectory.append(state)
        rewards.append(float(result.reward.mean()))
        
        if step % 100 == 0:
            print(f"  Step {step}")
    
    # Create animation with threat zones
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    anim = create_demo_animation(
        trajectory, rewards,
        arena_size=100.0,
        save_path=save_path,
        title="Threat Response - Intercept & Track",
        threats=threat_positions,
    )
    
    print(f"\n✅ Saved to {save_path}")
    return anim


def record_gps_denied_demo(
    num_agents: int = 10,
    num_steps: int = 300,
    save_path: str = "demos/gps_denied_demo.gif",
):
    """Record GPS-denied operation demo.
    
    Shows swarm operating with localization uncertainty.
    """
    print(f"🎬 Recording GPS-Denied Demo ({num_agents} agents)")
    print("-" * 50)
    
    config = EnvConfig(num_agents=num_agents, arena_size=100.0)
    env = SwarmEnv(config)
    
    key = jax.random.PRNGKey(456)
    state, _ = env.reset(key)
    
    # Create GPS denied scenario
    gps_task = create_gps_denied_scenario(
        arena_size=100.0,
        num_jammers=2,
        jammer_radius=25.0,
        key=key,
    )
    
    trajectory = [state]
    rewards = []
    gps_zones = [(j.position.tolist(), j.radius) for j in gps_task.jammers]
    
    for step in range(num_steps):
        key, action_key, obs_key = jax.random.split(key, 3)
        
        # Get degraded observations
        observed_pos, info = gps_task.get_observations(state, obs_key)
        
        # Random exploration
        actions = jax.random.normal(action_key, (num_agents, 3)) * 6.0
        
        result = env.step(state, actions)
        state = result.state
        trajectory.append(state)
        rewards.append(float(result.reward.mean()))
        
        if step % 50 == 0:
            jammed = int(info["jammed_agents"])
            print(f"  Step {step}: {jammed}/{num_agents} agents jammed")
    
    # Create animation with GPS denied zones
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    anim = create_demo_animation(
        trajectory, rewards,
        arena_size=100.0,
        save_path=save_path,
        title="GPS-Denied Operation",
        gps_denied=gps_zones,
    )
    
    print(f"\n✅ Saved to {save_path}")
    return anim


def record_all_demos():
    """Record all demo scenarios."""
    print("=" * 60)
    print("🎬 Recording All Artemis Demo Videos")
    print("=" * 60)
    print()
    
    demos = [
        ("Coverage", record_coverage_demo),
        ("Threat Response", record_threat_demo),
        ("GPS-Denied", record_gps_denied_demo),
    ]
    
    for name, record_fn in demos:
        print(f"\n### {name} ###")
        try:
            record_fn()
        except Exception as e:
            print(f"❌ Error recording {name}: {e}")
        print()
    
    print("=" * 60)
    print("✅ All demos recorded!")
    print("   Check the demos/ folder for output files.")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record demo videos")
    parser.add_argument("--demo", type=str, default="all",
                        choices=["all", "coverage", "threat", "gps"],
                        help="Demo to record")
    parser.add_argument("--agents", type=int, default=12, help="Number of agents")
    parser.add_argument("--steps", type=int, default=300, help="Number of steps")
    
    args = parser.parse_args()
    
    if args.demo == "all":
        record_all_demos()
    elif args.demo == "coverage":
        record_coverage_demo(args.agents, args.steps)
    elif args.demo == "threat":
        record_threat_demo(args.agents, args.steps)
    elif args.demo == "gps":
        record_gps_denied_demo(args.agents, args.steps)
